import argparse
from pathlib import Path

import numpy as np
import torch

from gtb_predict.checkpoints import load_encoder
from gtb_predict.compact import token_batch
from gtb_predict.data import parse_fasta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen full-protein features."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", default="full")
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    device = torch.device(args.device)
    if args.precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 extraction requires CUDA")
    records = parse_fasta(args.fasta.read_text(encoding="utf-8"))
    ids = list(records)
    model = load_encoder(args.checkpoint, device=args.device).requires_grad_(False)
    outputs = []
    torch.set_num_threads(2)
    with (
        torch.inference_mode(),
        torch.autocast(
            device.type, dtype=torch.float16, enabled=args.precision == "fp16"
        ),
    ):
        for start in range(0, len(ids), args.batch_size):
            sequences = [records[key] for key in ids[start : start + args.batch_size]]
            tokens = token_batch(sequences, device=args.device)
            hidden, valid = model.residues(tokens)
            outputs.append(model.representation(hidden, valid).float().cpu().numpy())
    features = np.concatenate(outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=np.asarray(ids),
        **{args.view: features, args.view + "_mean": features[:, : model.width]},
    )


if __name__ == "__main__":
    main()
