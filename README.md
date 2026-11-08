# Compact Models for Natural-Product Glycosylation

Team: **How to find drugs (fast)**

## Natural-product modification and enzyme selection

Natural-product drug discovery involves finding useful compounds and modifying them to obtain more suitable properties. Glycosylation is particularly relevant to plant natural products: attaching a sugar can alter a compound’s solubility, stability and biological activity. The effect depends on the compound, the sugar and where the sugar is attached.

Glycosyltransferases carry out these reactions. For a chosen natural compound, researchers need to find enzymes that can act on it and identify the sugar donors they can use. Many glycosyltransferase sequences are available, but experimentally measured substrate preferences cover only a fraction of them. Computational predictions can help prioritize combinations for experimental screening.

## Existing methods and our approach

Glycosyltransferase specificity has been studied using sequence-derived features, sequence similarity and protein representations. [GT-Predict](https://doi.org/10.1038/s41589-018-0154-9) and [GASP](https://doi.org/10.1021/acsomega.4c01583) provide experimental data and methods for studying enzyme–substrate compatibility. General protein models such as CARP and ESM2 provide representations pretrained on broad collections of protein sequences.

We investigated whether focused pretraining on glycosyltransferase sequences could produce useful representations in a compact model. We trained an encoder from scratch, evaluated it on sugar-donor prediction, and compressed it. We also studied whether the learned representations could help rank enzymes for natural-product compounds.

These experiments address two parts of enzyme selection:

- **Sugar-donor prediction:** identifying the sugar donor associated with an enzyme.
- **Acceptor-dependent enzyme ranking:** prioritizing enzymes for the compound receiving the sugar.

## Model and pretraining

We trained our encoder on **134,129 unlabelled glycosyltransferase sequences**, with another **14,280 sequences** reserved for pretraining validation. The collection includes plant UGT sequences and reviewed glycosyltransferase sequences.

The encoder uses residual dilated convolutions and has **6.36 million parameters**. During pretraining, some amino acids are hidden and the model learns to predict them from the surrounding sequence. For donor prediction, we use the resulting protein features to train a support vector classifier.

We trained three versions of the encoder and evaluated matched randomly initialized encoders using the same downstream procedure.

### Compression

We compressed the trained encoder using activation-aware low-rank approximation. Activations from unlabelled pretraining sequences were used to identify directions that contribute strongly to the layer outputs. The large linear transformations were then replaced with smaller ones.

The compressed encoder has **2.82 million parameters**, a **55.6% reduction**. We refitted the donor classifier on the compressed features and evaluated it using the same folds as the full encoder.

## Donor prediction and validation

The donor experiments use data from [Hennen et al.’s GT-B study](https://doi.org/10.1039/D4DD00338A). Our development evaluation contains **413 proteins with single donor labels**. The original study addresses multi-label prediction, so we reconstructed its feature-based method for our task and compared the methods on the same proteins and splits.

A random split can place closely related proteins in both training and evaluation data. To test performance on more distinct sequences, we grouped proteins connected by at least **70% sequence identity** over either the full sequence or the Rossmann domain. The resulting **171 groups** were kept together when constructing the folds.

We used five outer folds for evaluation and three inner folds to choose classifier settings. Preprocessing was fitted using the training data in each fold. All methods share the same outer folds. A separate final test set remains reserved.

## Donor-prediction results

The scores below are accuracies on the grouped development evaluation. Our full and compressed model scores are averages across three pretraining runs. CARP and ESM2 use their released pretrained encoders.

| Method | Encoder parameters | Donor accuracy |
|---|---:|---:|
| Reconstructed paper method | — | 73.85% |
| Compressed random encoder | 2.82 million | 70.22% |
| Our full pretrained encoder | 6.36 million | 78.93% |
| Our compressed pretrained encoder | 2.82 million | 79.34% |
| ESM2-8M | Approximately 8 million | 79.90% |
| CARP-600K | Approximately 0.60 million | 80.15% |

The compressed model retained similar average donor performance with less than half the encoder parameters. Its mean accuracy was **79.34%**, compared with **78.93%** for the full encoder. The paired difference was **+0.40 percentage points**, with a sequence-group bootstrap interval of **−1.43 to +2.30 percentage points**.

The individual runs show the variation behind the averages:

| Training run | Full encoder | Compressed encoder |
|---|---:|---:|
| 1 | 81.84% | 81.60% |
| 2 | 79.42% | 79.90% |
| 3 | 75.54% | 76.51% |

We also tested end-to-end fine-tuning using Rossmann-domain inputs. The pretrained model scored **78.69%**, compared with **71.19%** for the same architecture trained from scratch under the matched evaluation.

## Ranking enzymes for natural compounds

We evaluated enzyme ranking using **3,990 measured enzyme–compound pairs**, covering **73 enzymes and 77 compounds**, from the GASP/GT-Predict data. The predictor combines protein and compound representations. Evaluation uses held-out enzyme groups, and candidate enzymes are ranked separately for each compound.

| Protein representation | Mean average precision |
|---|---:|
| Matched random encoder | 0.5718 |
| ESM2-8M | 0.5900 |
| Our pretrained encoder | 0.6048 |
| Ordered BLOSUM representation | 0.6290 |

Our pretrained encoder scored higher than its random counterpart and ESM2-8M in this comparison. Ordered BLOSUM gave the strongest ranking.

We also examined a polyphenol screen covering **30 enzymes and 32 compounds**, with measurements of conversion and product distribution. For each compound, we selected three enzymes and measured the best dominant-product fraction among them. Compound-informed selection using our representations scored **0.7689**, compared with **0.7371** for a fixed supervised panel. An enzyme-identity model also scored **0.7689**, and a chemical-neighbour method scored **0.7735**.

## Further work

- Investigate variation across training runs and improve consistency.
- Compare model sizes and training-data mixtures.
- Improve compound-conditioned enzyme ranking and evaluate additional glycosyltransferase tasks.
- Measure inference time and memory use for the full encoder, compressed encoder and reference models.

## References

- [GT-B donor-prediction study](https://doi.org/10.1039/D4DD00338A), [source code](https://github.com/vbharadwcomosci/GTB_substrate_prediction) and [released data](https://doi.org/10.5281/zenodo.15786055)
- [GASP](https://doi.org/10.1021/acsomega.4c01583) and [GT-Predict](https://doi.org/10.1038/s41589-018-0154-9)
- [Polyphenol glycosylation screen](https://doi.org/10.1021/acsomega.3c08255)
- [ESM2](https://doi.org/10.1126/science.ade2574) and [CARP](https://pubmed.ncbi.nlm.nih.gov/38428432/)
- [Data leakage in machine-learning-based science](https://arxiv.org/abs/2207.07048) and [variation in model evaluation](https://arxiv.org/abs/2304.01910)

## Current limitations

Results are from development datasets, vary across training runs and do not establish improved performance on rare donor classes or new natural-product reactions; independent experimental validation is still needed.

## Repository

See [installation and usage](docs/usage.md), [evaluation inputs](docs/evaluation.md) and [model details](docs/model.md).

Code is covered by the [GPL-3.0 license](LICENSE). Sources and licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Running the code

Install with `python -m pip install -e ".[neural]"`. The six scripts in `scripts/` provide pretraining, compression, feature extraction, donor prediction and the two grouped evaluations. Each has `--help`; [usage instructions](docs/usage.md) give the commands and input formats.

- The repository includes the 413 development-protein fold assignments. `prepare_training` retrieves the three released donor-training files from the pinned upstream source.
- Pretraining requires the qualified sequence corpus. Compression requires that corpus and a trained encoder. Neither is bundled.
- Frozen donor prediction requires the full 6.36M encoder and its matching all-development RBF SVC. These trained files are not bundled. This adapter does not deploy the compressed encoder; a matching all-development compressed classifier has not been verified.
- Repeating the reported evaluations requires the original frozen feature banks and, for acceptors, the measured-pair table, chemical descriptors and fold assignments. These inputs are not bundled.

The trained files currently remain in private research storage. Redistribution terms for the training corpus and derived checkpoints must be confirmed before a public weight release. The next release step is to confirm those terms and publish the encoder with its matching classifier. Until then, the repository supplies runnable implementations and input formats, not a ready-to-run trained predictor.
