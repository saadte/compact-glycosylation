import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .checkpoints import cpu_state, save_checkpoint
from .compact import CompactProtein, pretraining_loss
from .sequence import residue_codes


class CorpusSampler:
    def __init__(
        self, records: list[dict], *, device: str, track_sequences: bool = False
    ):
        if not records or any(not r["sequence"] for r in records):
            raise ValueError("Corpus requires nonempty sequences")
        self.device = device
        self.tokens = [
            (residue_codes(r["sequence"]) + 3).astype(np.uint8) for r in records
        ]
        self.sources = ["+".join(r["sources"]) for r in records]
        self.observed = np.zeros(len(records), dtype=bool) if track_sequences else None
        membership = {}
        for i, record in enumerate(records):
            membership.setdefault(record["cluster"], []).append(i)
        self.members = list(membership.values())
        self.reviewed_clusters = []
        self.plant_clusters = []
        for cluster, indices in enumerate(self.members):
            pool = (
                self.reviewed_clusters
                if any("reviewed_uniprot" in records[i]["sources"] for i in indices)
                else self.plant_clusters
            )
            pool.append(cluster)
        if not self.reviewed_clusters or not self.plant_clusters:
            raise ValueError(
                "The source-balanced corpus requires reviewed-bearing and plant-only clusters"
            )

    def batch(
        self, random: np.random.Generator, size: int
    ) -> tuple[torch.Tensor, Counter]:
        if size <= 0:
            raise ValueError("Batch size must be positive")
        n = min(size, len(self.members))
        if (
            len(self.reviewed_clusters) < n // 2
            or len(self.plant_clusters) < n - n // 2
        ):
            raise ValueError(
                "A distinct source-balanced batch exceeds its available clusters"
            )
        clusters = random.permutation(
            np.concatenate(
                [
                    random.choice(self.reviewed_clusters, n // 2, replace=False),
                    random.choice(self.plant_clusters, n - n // 2, replace=False),
                ]
            )
        )
        indices = [int(random.choice(self.members[int(c)])) for c in clusters]
        if self.observed is not None:
            self.observed[indices] = True
        sequences = []
        for i in indices:
            sequence = self.tokens[i]
            start = int(random.integers(max(1, len(sequence) - 512 + 1)))
            sequence = sequence[start : start + 512]
            sequences.append(sequence)
        batch = np.zeros((len(indices), max(map(len, sequences))), dtype=np.int64)
        for i, sequence in enumerate(sequences):
            batch[i, : len(sequence)] = sequence
        exposure = Counter()
        for i, sequence in zip(indices, sequences):
            exposure[self.sources[i]] += len(sequence)
        return torch.as_tensor(batch, device=self.device), exposure


def read_corpus(path: Path, *, device: str) -> tuple[CorpusSampler, CorpusSampler]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = json.load(stream)
    if any(r["split"] not in ("train", "validation") for r in records):
        raise ValueError("Corpus split must be train or validation")
    training = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    if not training or not validation:
        raise ValueError("Corpus requires training and validation records")
    if {r["cluster"] for r in training} & {r["cluster"] for r in validation}:
        raise ValueError("Training and validation share a cluster")
    return CorpusSampler(training, device=device), CorpusSampler(
        validation, device=device
    )


def validation_loss(model: CompactProtein, sampler: CorpusSampler) -> tuple[float, int]:
    model.eval()
    random, masking = [np.random.default_rng(s) for s in (1901, 1902)]
    weighted, total = 0.0, 0
    device = next(model.parameters()).device.type
    with (
        torch.inference_mode(),
        torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"),
    ):
        for _ in range(32):
            batch, _ = sampler.batch(random, 32)
            _, stats = pretraining_loss(
                model, batch, random=masking, coherence_weight=0.0
            )
            weighted += stats["masked_cross_entropy"] * stats["masked_tokens"]
            total += stats["masked_tokens"]
    model.train()
    return weighted / total, total


def select_pretraining_checkpoint(history: list[dict]) -> dict:
    return min(
        (r for r in history if r["marks"]), key=lambda r: r["validation_masked_ce"]
    )


def train(
    output: Path,
    training: CorpusSampler,
    validation: CorpusSampler,
    *,
    batch_size: int,
    seed: int = 1801,
) -> dict:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("mlm_6m_*.pt")):
        raise FileExistsError("Output already contains pretraining checkpoints")
    torch.manual_seed(seed)
    device = torch.device(training.device)
    model = CompactProtein(512, 16).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    random, masking = [np.random.default_rng(s) for s in (1801, 1802)]
    marks = (1_000_000, 25_000_000, 50_000_000, 100_000_000)
    step, tokens, history, exposure = 0, 0, [], Counter()
    while tokens < marks[-1]:
        previous_tokens = tokens
        batch, source_exposure = training.batch(random, batch_size)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            loss, stats = pretraining_loss(
                model, batch, random=masking, coherence_weight=0.0
            )
        if not np.isfinite(stats["masked_cross_entropy"]):
            raise FloatingPointError("Nonfinite pretraining loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        tokens += stats["input_tokens"]
        exposure.update(source_exposure)
        reached = [mark for mark in marks if previous_tokens < mark <= tokens]
        if not reached:
            continue
        ce, nmasked = validation_loss(model, validation)
        history.append(
            {
                "step": step,
                "tokens": tokens,
                "validation_masked_ce": ce,
                "validation_masked_tokens": nmasked,
                "marks": reached,
                **stats,
            }
        )
        save_checkpoint(
            output / f"mlm_6m_step{step}.pt",
            {
                "model": cpu_state(model),
                "width": 512,
                "layers": 16,
                "configuration": "mlm_6m",
                "seed": seed,
                "step": step,
                "tokens": tokens,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "sampling_rng": random.bit_generator.state,
                "mask_rng": masking.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else [],
                "batch_size": batch_size,
                "history": history,
                "exposure": dict(exposure),
            },
        )
        print(json.dumps(history[-1]), flush=True)
    selected = select_pretraining_checkpoint(history)
    saved = torch.load(output / f"mlm_6m_step{selected['step']}.pt", weights_only=True)
    save_checkpoint(
        output / "mlm_6m_best.pt",
        {
            key: saved[key]
            for key in (
                "model",
                "width",
                "layers",
                "configuration",
                "seed",
                "step",
                "tokens",
            )
        },
    )
    report = {
        "seed": seed,
        "batch_size": batch_size,
        "history": history,
        "selected_checkpoint": selected,
        "source_input_tokens": dict(exposure),
    }
    (output / "pretraining.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
