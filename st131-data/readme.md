# ST131 inputs

The inference and summary scripts use these three files in `input/`:

| File | Purpose |
| --- | --- |
| `named.tree_lsd.date.noref.pruned_unannotated.nwk` | Dated ST131 tree, pruned of the reference, with named internal nodes. |
| `genetic_features_marginal.csv` | Genetic feature states indexed by the `node` column. Sampled tips have binary states; reconstructed internal-node states can be fractional marginal probabilities. Some columns represent correlated or grouped genetic features. |
| `meta_features_marginal.csv` | Reconstructed sampling covariates indexed by node. The default sampling predictor is `blood_META`, with urine as the baseline specimen category. |

The files are copied unchanged from the research workflow and cover 882 sampled tips and 881 internal nodes. Tree node names must match the feature and sampling table indices. Do not rename nodes in only one input file or round fractional ancestral states.

The candidate runner constructs interval-specific tree segments itself. The pre-split tree, sampling-mask table, historical fitting-parameter dumps, and simulated datasets are not needed for this workflow and are excluded from this export.
