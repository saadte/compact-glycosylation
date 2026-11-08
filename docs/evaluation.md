# Evaluation workflows

These commands evaluate the released development/source partitions. They do not
read held-out donor endpoints or the separate external acceptor assay. Run real
feature extraction and model fitting on an appropriately provisioned remote
machine, not a laptop. No dataset or checkpoint is downloaded by these commands.

Install the package first with `python -m pip install -e .`. Evaluation uses
NumPy, SciPy, scikit-learn 1.6 or later, joblib, and threadpoolctl. The shuffled grouped inner folds
require the newer GroupKFold API. Use a separate output directory for each arm;
files in that directory are replaced on a rerun.

## Donor annotation classification

Required inputs:

- A training directory containing `train.csv`, `full.fasta`, and `domain.fasta`.
  CSV columns are `ID` and `Donor`. The released partition contains 413 unique
  accessions, each with exactly one of the eight annotation categories.
- Frozen roles JSON with `ids`, `groups`, and `outer_folds` arrays in CSV order.
  The development protocol has 171 similarity components and five outer folds.
- A feature NPZ containing a string `ids` array in CSV order and a finite 2-D
  matrix under the requested view name. Supply frozen encoder features, never
  features from an encoder fitted using the outer query labels.

The original roles use connected components at 70% identity, taking the maximum
of full-length and domain alignment identity. Five-fold GroupKFold used shuffle
and seed 20261107. Supply those frozen roles rather than silently regrouping the
data. Each fitting partition selects C in 0.1, 1, 10, 100 and kernel in linear,
RBF using three-fold shuffled GroupKFold with seed 20261108 and mean validation
accuracy. Equal scores retain the first configuration in that order. Query rows
never enter inner selection or fitting. SVC uses random state 0.

Example public protein-embedding comparator:

```sh
python scripts/evaluate_donors.py --training data/donor_training --roles data/donor_folds.json --features data/esm_means.npz --view domain_mean --profile public_embedding --output results/donor_public_embedding
```

This profile uses L2-normalized ESM mean embeddings, the calibrated SVC grid,
and an unbounded solver. Native `predict` supplies annotation decisions;
calibrated probabilities supply ranking scores. This is the public protein-embedding comparator on grouped development folds.
It does not reproduce the separate residue/structure KNN reconstruction reported
at 73.85%; that baseline is not implemented by this command. Full-length means are another explicitly named view.

Example compact frozen-feature readout:

```sh
python scripts/evaluate_donors.py --training data/donor_training --roles data/donor_folds.json --features data/compact_features.npz --view full --profile compact --output results/donor_compact
```

Use global-plus-relative-thirds pooled representations from the fixed compact
encoder. Rows are first L2-normalized, then each inner fitting partition fits
its own StandardScaler followed by L2 normalization. The selected outer model
refits that preprocessing on its fitting rows. This profile uses uncalibrated
SVC decisions and a 1,000,000-iteration solver limit. Nonconverged inner choices
are ineligible; a failed selected refit is an error. This matches the standardized
compact readout, not earlier unstandardized experiments. Feature extraction and
pretraining remain separate from this evaluation command.

Both profiles save five fitted heads, `predictions.npz`, and `evaluation.json`.
The predictions contain IDs, labels, groups, folds, ranking values under
`decision`, and native one-hot decisions under `native`. The report includes
per-fold selection results and pooled metrics. On this single-annotation
endpoint, native samples F1 equals accuracy. These annotations do not establish
biochemical exclusivity or true multi-donor activity. Uncalibrated decision
values are not probabilities and are not interpreted with a 0.5 threshold.

## GASP fixed-compound enzyme ranking

This is the joint kernel-ridge endpoint used for the submission's whole-protein
comparisons, not the earlier pair-classification random forests or later
per-compound/positional readout experiments.

Required inputs:

- Source-only JSON with `source_pairs`, records containing string `enzyme` and
  `cid` identifiers and binary `label`, and `chemistry`, mapping each CID to its
  153 fixed published descriptors. There are 3,990 measured pairs, 793 positives,
  73 enzymes and 77 compounds. Missing pairs are never made negative.
- Frozen roles JSON with `enzymes`, mapping each enzyme ID to `group` and
  `outer_fold`. There are 32 components and five outer folds. Original roles use
  detected MMseqs2 edges at >=50% identity and >=80% bidirectional coverage,
  plus exact-sequence matches, followed by shuffled GroupKFold seed 20261108.
- Protein feature NPZ with `ids` in sorted source enzyme order and one finite
  matrix under the chosen representation name. `mlm_6m__whole`,
  `random_6m__whole` and `esm2_8m__whole` contain the frozen full-protein mean
  features from the matched extraction. These are not global-plus-thirds
  features or positional features. The full encoder checkpoint was trained with
  100M residue-token exposures; the arm name is a historical model-size label.
- `ordered_blosum_reference` requires the released ordered substitution-vector
  encoding for the first 235 fixed profile positions: 4,935 coordinates, in
  channel-by-position order, with the published '*' gap column. It must not be
  replaced by mean BLOSUM or amino-acid composition. `encode_consensus` retains
  that matrix layout; users supply the qualified alignment and feature bank.

```sh
python scripts/evaluate_acceptors.py --inputs data/gasp_source.json --roles data/gasp_roles.json --features data/gasp_features.npz --representation mlm_6m__whole --output results/gasp_compact
```

Repeat with the explicit random, ESM or ordered-reference key and a separate
output directory. For chemistry-only, omit `--features` and pass
`--representation chemistry_only`.

Each actual fitting role fits StandardScaler on unique fitting proteins and,
separately, unique fitting compounds. Row L2 normalization follows. RBF gamma
is inverse median positive squared distance among fitting objects only. The
joint pair kernel is the elementwise product of chemical and protein kernels.
Chemistry-only omits the protein factor and evaluates each unique query compound
once to preserve exact within-compound ties.

Ridge has a free intercept and penalty n_fitting_pairs times lambda. Four fixed
lambdas are 1e-5, 1e-4, 1e-3, 1e-2. Three shuffled grouped inner folds use seed
20261108. Select using the equal-compound/equal-fold enzyme-ranking AP below;
exact objective ties choose the stronger penalty. Fit the chosen model on the
outer fitting rows. Transform statistics and bandwidth are refitted within every
inner role, never using query observations. Computation uses FP64 and two BLAS
threads. Material PSD violations and failed ridge normal equations raise errors.

The primary endpoint ranks assayed enzymes separately within each held-out fold
and compound. Eligible groups have at least five enzymes and both classes.
Average eligible compound AP equally within each fold, then average all five
folds equally. The original endpoint has 152 eligible and 213 excluded groups.
A fold lacking eligible groups makes the aggregate unavailable; it is not
silently dropped. AUROC and expected precision at 1, 3 and 5 use the same groups.
Boundary ties receive their expected positive fraction, not arbitrary row order.
These unrestricted ridge scores are not calibrated probabilities.

Outputs contain five compact coefficient/preprocessing NPZ files, inner score
paths and selected lambdas, row-aligned out-of-fold `predictions.npz`, and
`evaluation.json` with all eligible/excluded groups, fold metrics and solver
checks. Retain input feature banks and row order with these model states.

The reported whole-feature comparison is AP 0.5718 random, 0.5900 ESM, 0.6048
owned encoder, and 0.6290 ordered BLOSUM. This is repeated source development,
not an independent external assay, new-compound validation or evidence that the
owned representation beats the ordered reference. Published alignment/chemical
priors and benchmark overlap with pretraining require disclosure. No confidence
interval is manufactured when the original uncertainty procedure lacks support.

The public workflow extraction is not a fresh run of those experiments. Actual
reproduction requires the original frozen roles, descriptors, checkpoint-derived
feature banks and compatible numerical dependencies. No real dataset fitting
was run locally while preparing this code.
