import numpy as np
import torch
from torch.nn import functional as F

from .compact import CompactProtein
from .low_rank import factorize_linear
from .pretraining import CorpusSampler


def calibration_batch(sampler: CorpusSampler) -> torch.Tensor:
    if len(sampler.members) < 64:
        raise ValueError("Calibration requires at least 64 training clusters")
    tokens, _ = sampler.batch(np.random.default_rng(20261108), 64)
    return tokens


def output_moments(
    model: CompactProtein, tokens: torch.Tensor
) -> dict[str, torch.Tensor]:
    if model.training:
        raise ValueError("Calibration requires an encoder in evaluation mode")
    if tokens.ndim != 2 or not len(tokens) or torch.any((tokens != 0).sum(1) == 0):
        raise ValueError("Calibration requires nonempty token sequences")
    moments, counts, hooks = {}, {}, []
    context = {}
    for index, block in enumerate(model.blocks):
        for name in ("inward", "outward"):
            key = f"blocks.{index}.{name}"

            def capture(layer, inputs, key=key):
                # Exclude padding before collecting bias-free output moments.
                values = inputs[0][context["valid"]].float()
                outputs = F.linear(values, layer.weight.float(), None)
                moment = (outputs.T @ outputs).double().cpu()
                moments[key] = moments.get(key, torch.zeros_like(moment)) + moment
                counts[key] = counts.get(key, 0) + len(values)

            hooks.append(getattr(block, name).register_forward_pre_hook(capture))
    try:
        with torch.inference_mode(), torch.autocast(tokens.device.type, enabled=False):
            for start in range(0, len(tokens), 8):
                batch = tokens[start : start + 8]
                context["valid"] = batch != 0
                model.residues(batch)
    finally:
        for hook in hooks:
            hook.remove()
    expected = int((tokens != 0).sum())
    if set(counts.values()) != {expected} or len(moments) != 2 * model.layers:
        raise ValueError(
            "Layer moments do not cover exactly the valid calibration tokens"
        )
    return {key: value / expected for key, value in moments.items()}


def compress(
    model: CompactProtein, *, moments: dict[str, torch.Tensor], rank: int = 96
) -> None:
    # All moments come from the unchanged parent, before replacing any projection.
    for index, block in enumerate(model.blocks):
        for name in ("inward", "outward"):
            layer = getattr(block, name)
            reduced = factorize_linear(
                layer, rank=rank, output_second_moment=moments[f"blocks.{index}.{name}"]
            )
            setattr(block, name, reduced)
