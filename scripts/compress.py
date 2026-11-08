import argparse
from pathlib import Path

import torch

from gtb_predict.checkpoints import cpu_state, load_encoder, save_checkpoint
from gtb_predict.compression import calibration_batch, compress, output_moments
from gtb_predict.pretraining import read_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress encoder projections to rank 96."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_encoder(args.checkpoint, device=args.device)
    if (model.width, model.layers) != (512, 16):
        raise ValueError("Compression requires the width512, layers16 parent encoder")
    training, _ = read_corpus(args.corpus, device=args.device)
    tokens = calibration_batch(training)
    moments = output_moments(model, tokens)
    compress(model, moments=moments)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != 2_824_192:
        raise ValueError("Compressed encoder parameter count differs from the protocol")
    save_checkpoint(
        args.output,
        {
            "model": cpu_state(model),
            "rank": 96,
            "width": 512,
            "layers": 16,
            "heads_stripped": True,
            "calibration_seed": 20261108,
            "calibration_tokens": int((tokens != 0).sum()),
        },
    )


if __name__ == "__main__":
    main()
