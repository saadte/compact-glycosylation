import argparse
import json
from pathlib import Path

from gtb_predict.data import parse_fasta
from gtb_predict.inference import DonorPredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a donor annotation with the exported full encoder."
    )
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    args = parser.parse_args()
    records = parse_fasta(args.fasta.read_text(encoding="utf-8"))
    if len(records) != 1:
        raise ValueError("Provide exactly one protein sequence")
    accession, sequence = next(iter(records.items()))
    predictor = DonorPredictor(encoder_path=args.encoder, readout_path=args.readout)
    result = predictor.predict(sequence)
    print(json.dumps({"id": accession, **result}, indent=2))


if __name__ == "__main__":
    main()
