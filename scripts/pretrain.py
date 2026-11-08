import argparse
from pathlib import Path

import torch

from gtb_predict.checkpoints import save_random_encoder
from gtb_predict.pretraining import read_corpus, train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the 6.36M-residue encoder from scratch."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--corpus", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--device", required=True)
    training.add_argument("--batch-size", type=int, required=True)
    training.add_argument("--seed", type=int, default=1801)
    random = commands.add_parser("random")
    random.add_argument("--output", type=Path, required=True)
    random.add_argument("--seed", type=int, default=1801)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.command == "random":
        save_random_encoder(args.output, seed=args.seed)
        return
    training, validation = read_corpus(args.corpus, device=args.device)
    train(args.output, training, validation, batch_size=args.batch_size, seed=args.seed)


if __name__ == "__main__":
    main()
