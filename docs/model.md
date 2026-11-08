# Model and scientific scope

The residue encoder has 6,363,136 parameters: width 512, 16 gated residual
convolution blocks, kernel width 5, and dilations cycling through 1, 2, 4, 8,
16 and 32. The training graph also contains reconstruction, coherence and donor
heads. They are excluded from the reported encoder count and frozen features.

Each protein produces 2,048 features: the global residue mean followed by means
over its three relative sequence regions. The first 512 features are the global
mean. Padding is excluded. Downstream encoding retains every residue; the
512-residue window applies only during pretraining.

## Pretraining

The qualified corpus contains 134,129 training and 14,280 validation sequences.
It combines plant UGT and reviewed glycosyltransferase sequences. Clusters are
kept separate across the two partitions. Exact development overlaps and detected
development homologs were excluded. Overlap with the reserved final donor set
has not been certified.

Sampling chooses equal numbers of reviewed-bearing and plant-only clusters,
then a member and a contiguous window from each cluster. The model reconstructs
15% of canonical residues in spans of length 3-7. Selected positions are 80%
masked, 10% random replacements and 10% unchanged. The submitted MLM model
does not use the optional coherence objective in the shared architecture.

Training uses AdamW with learning rate 0.001, weight decay 0.01, batch size 128,
gradient clipping at 1, FP32 stored weights and FP16 autocast. Validation chooses
among 1, 25, 50 and 100 million input-token milestones. Three initializations use
seeds 1801, 2801 and 3801. Sampling and masking streams remain fixed across these
initializations. The corpus size is distinct from the number of sampled proteins
and repeated token presentations.

## Compression

Activation-aware rank-96 factors replace the two linear maps in each residual
block. Calibration uses 64 source-training proteins from distinct source-balanced
clusters, with sampling seed 20261108. Bias-free output second moments exclude
padding and are collected from the unchanged parent. Their leading eigenspaces
determine the projections. This leaves 2,824,192 residue-encoder parameters.
The donor classifier is refitted after compression.

Across the three runs, compressed accuracy is 0.793382 and dense accuracy is
0.789346. The paired mean difference is 0.004036, with a sequence-group bootstrap
95% interval from -0.014297 to 0.022961. This does not establish superiority or
independently powered noninferiority.

## Inference

The provided frozen donor adapter follows the full 6.36M encoder export and its
all-development RBF SVC readout (C=100). That readout was fitted on all 413
development proteins after grouped inner selection. It has no independent score
of its own. Use the nested held-out scores for performance reporting.

This adapter is specific to the full encoder and matching readout. Compression
and feature extraction support rank-96 checkpoints separately; no compressed
deployment classifier is supplied. The adapter uses CPU FP32 and accepts full
sequences of 319-2,358 residues. Ambiguous residues map to X. The eighth annotation
is the unresolved source category `Eight`. Classifier margins are not calibrated
probabilities.

Only load trusted checkpoint and readout files. PyTorch loading uses its
weights-only reader. Joblib model files use Python serialization and can execute
code when loaded.

## Limits and available inputs

Results are exploratory development evidence. Repeated development and model-seed
variation limit generalization claims. The original full model recovered none
of the eight xylose or three rhamnose donor annotations. On plant GT1, its accuracy
matched the majority-class baseline.

The acceptor ranking study is supervised compatibility prediction on measured
pairs. Unassayed pairs are not treated as inactive. Glycosylation activity,
product distribution, pharmacological activity and clinical suitability are
different endpoints. The code does not predict therapeutic benefit.

Model weights, the pretraining corpus and cached feature banks are not included.
Corpus and checkpoint redistribution rights have not all been established.
The supplied donor fold file contains only the released development identifiers,
group assignments and fold assignments; no reserved final data is present.
