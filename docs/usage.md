# Installation and usage

Use Python 3.11 or later. Install the package in a separate environment:

```sh
python -m pip install -e ".[neural]"
```

The numerical dependencies used by the original study are pinned in
`pyproject.toml`. Install a PyTorch build suitable for the selected machine
before running GPU work. Full experiments require a provisioned remote machine;
do not run training, dataset preparation or large inference jobs on a laptop.

## Inputs

The public repository contains the encoder, compression, frozen donor inference,
grouped evaluation and source data readers. Model weights, the sequence corpus
and cached features must be supplied separately under their source licenses.

The fixed donor development folds are in `data/donor_folds.json`. To retrieve
the three released training files on the experiment machine:

```python
from pathlib import Path
from gtb_predict.data import prepare_training

prepare_training(Path("data/donor_training"))
```

This reader requests only the 413-row released training partition. Do not replace
it with automatic discovery of other partitions.

## Pretraining

The corpus is a gzip-compressed JSON list. Each record contains `sequence`,
`sources`, `cluster` and `split`. Split is `train` or `validation`;
`sources` is a list, with `reviewed_uniprot` marking reviewed-source membership.
Prepare and qualify the corpus separately. Reproduction needs the same ordered
records and cluster assignments used in the study.

```sh
python scripts/pretrain.py train --corpus data/corpus.json.gz --output outputs/seed1801 --device cuda --batch-size 128 --seed 1801
python scripts/pretrain.py random --output outputs/random1801.pt --seed 1801
```

Training starts fresh and saves checkpoints at four token milestones. It writes
`mlm_6m_best.pt` using validation-only selection. There is no resume command.
The other evaluated initialization seeds are 2801 and 3801.

## Compression and features

```sh
python scripts/compress.py --checkpoint outputs/seed1801/mlm_6m_best.pt --corpus data/corpus.json.gz --output outputs/compact1801.pt --device cuda
python scripts/encode.py --checkpoint outputs/compact1801.pt --fasta data/donor_training/full.fasta --output outputs/compact_features.npz --device cuda --precision fp16 --view full
```

Compression uses fixed rank 96 and source-training calibration. Encoding uses
16 proteins per batch by default and retains full sequences. The study's feature
extraction used CUDA FP16 autocast, then converted pooled features to FP32.
Changing batching, precision or runtime can change numerical results.

The output contains accession IDs, global-plus-third features under `full`,
and global means under `full_mean`. The `--view` option names another explicitly
supplied sequence view, such as `domain`; it does not identify or crop a domain.

See [evaluation inputs](evaluation.md) for the grouped donor and acceptor commands.
The 73.85% reconstructed structural-feature baseline is separate from the public
embedding SVC profile. Its alignment and feature primitives are included, but
this repository does not automate its full structural-data preparation.

## Frozen donor prediction

For the original full-encoder export and its matching all-development classifier:

```sh
python scripts/predict_donor.py --encoder checkpoints/own6m_encoder.pt --readout checkpoints/own6m_all_development_readout.joblib --fasta input.fasta
```

The input must contain exactly one protein. This command performs no fitting.
Only use trusted serialized model files. The output is the predicted source
annotation and input metadata. See [model details](model.md) for its scientific
scope, input limits and class imbalance.
