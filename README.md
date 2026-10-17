# BayesMFBD: ST131 inference and summaries

This directory contains the ST131 candidate-model workflow: inference, genetic-effect priors, WAIC comparison, posterior summaries, and plotting. It retains the required supporting modules and the three ST131 input files. Generated fits and figures, simulated datasets, IDE files, caches, and historical parameter dumps are excluded.

## Setup

Use Python 3.12. The requirements list the versions used for validation.

```bash
conda env create -f environment.yml
conda activate bayesmfbd
```

Alternatively, install `requirements.txt` into an existing Python 3.12 environment:

```bash
python -m pip install -r requirements.txt
```

The statistical figures use R. Install the plotting packages once:

```r
install.packages(c("ggplot2", "patchwork", "jsonlite", "svglite", "ragg", "MASS"))
```

Run the commands below from this directory. Input paths default to `st131-data/input/`.

## Inference

`HMCST131Candidates.py` fits five response families: linear, unconstrained B-spline, monotone B-spline, unimodal B-spline, and monotone GP. Each supports a reduced set of main effects or a full set including pairwise interactions. The bundled default predictors give 13 main effects and 78 interactions.

Check the full GP model's initial density and gradients before sampling:

```bash
python HMCST131Candidates.py \
  --model gp_monotone --feature-set full \
  --prior SpikeSlab --spike-slab-inclusion-prob 0.1 \
  --check-only
```

Run the GP with fixed inclusion probability 0.1, normal spike SD 0.1, and normal slab SD 1:

```bash
python HMCST131Candidates.py \
  --model gp_monotone --feature-set full \
  --prior SpikeSlab \
  --spike-slab-inclusion-prob 0.1 \
  --spike-slab-spike-scale 0.1 --spike-slab-scale 1 \
  --num-warmup 2000 --num-samples 5000 --num-chains 4 \
  --target-accept 0.95 \
  --output-dir ST131_HMC_candidates_soft_spike_slab
```

Spike-and-slab applies equally to raw main-effect and interaction coefficients, without predictor-SD scaling or coefficient normalization. Binary component indicators are marginalized. To estimate a shared inclusion probability instead, replace the fixed-probability flag with `--spike-slab-learn-inclusion-prob`; do not pass both flags. `--spike-slab-p-alpha` and `--spike-slab-p-beta` specify its Beta prior.

The GP uses `Phi(z / 4)` as its interpolation coordinate. It predicts log fitness and exponentiates after interpolation. Nonlinear latent coefficients and their exponentials are not background-specific marginal fitness effects.

To compare all ten candidates under the same prior and dataset, use `--model all --feature-set all`. Other supported priors are `Laplace`, `Normal`, `Uniform`, and `R2D2`. Branch random effects are included by default; `--no-branch-effects` disables them. Full option and model definitions are in [HMCST131Candidates.md](HMCST131Candidates.md).

Check `diagnostics.json`, R-hat, ESS, and divergent transitions before interpreting posterior results. The runner saves chain-preserving posterior samples, sampler diagnostics, coefficient summaries, and pointwise log likelihoods inside each candidate directory.

## WAIC and minimum ESS

```bash
python CompareST131WAIC.py ST131_HMC_candidates_soft_spike_slab
```

The comparison reports minimum ESS and the corresponding parameter names from the saved summaries. Compare candidates fit to the same data, tree, and likelihood factorization. This WAIC uses the stored tree-event factorization; its rows are not independent held-out tips or trees.

## MFE, fitness curve, and contributions

Generate summaries from an existing full GP fit:

```bash
RESULTS=ST131_HMC_candidates_soft_spike_slab
FIT="$RESULTS/gp_monotone_full"
SUMMARY="$RESULTS/gp_monotone_full_summaries"

python ST131GPFitnessCurve.py --fit-dir "$FIT" \
  --output-dir "$SUMMARY/fitness_curve" --grid-size 1501 --no-save-draws
python ST131MFE.py --results-dir "$RESULTS" --candidate gp_monotone_full \
  --output-dir "$SUMMARY/MFE" --point-estimator mean --no-save-draws
python ST131InteractionMFE.py --fit-dir "$FIT" \
  --output-dir "$SUMMARY/MFE_interactions"
python ST131GPContributions.py --fit-dir "$FIT" \
  --output-dir "$SUMMARY/contributions_all" --interaction-filter all
python ST131GPContributions.py --fit-dir "$FIT" \
  --output-dir "$SUMMARY/contributions" \
  --interaction-filter significant --interaction-credible-mass 0.90

Rscript PlotST131GPFitnessCurve.R "$SUMMARY/fitness_curve" linear
Rscript PlotST131GPFitnessCurve.R "$SUMMARY/fitness_curve" linear -4 10
Rscript PlotST131GPAllMFE.R "$SUMMARY/MFE"
Rscript PlotST131GPInteractionMFE.R "$SUMMARY/MFE_interactions"
Rscript PlotST131GPContributions.R "$SUMMARY/contributions_all"
Rscript PlotST131GPContributions.R "$SUMMARY/contributions"
```

All retained posterior draws enter the summaries. `--no-save-draws` omits large derived-draw archives after calculating the summaries; it does not thin the source posterior. The full fitness curve is generated over the fitted latent range, with the optional −4 to 10 view as a separate display. Individual figures have PDF/SVG/PNG exports; the MFE and fitness figures also provide TIFFs.

- **Mutation MFE:** paired genetic fitness with a mutation divided by fitness after removing that mutation, with affected interaction predictors recomputed.
- **Interaction MFE:** full genetic fitness divided by fitness after setting only the focal interaction coefficient to zero. Main effects, other interactions, and the fitted GP are retained. The interaction plotter produces an overview and pages covering all observed double-carrier backgrounds.
- **Contributions:** unique decreases in OLS R-squared for main effects, interactions, and global epistasis (Nonlinear). Raw contributions need not sum to one; ternary plots show draw-normalized relative shares. Under the 90% coefficient-interval filter, excluded interactions remain unassigned and are not moved into the nonlinear component. Fitted predictions are unchanged by attribution filtering.

Genetic fitness and contribution plots exclude branch random effects; common branch multipliers cancel in paired MFE ratios. See the generated metadata, source CSVs, and figure notes for the precise averaging and interval definitions.

`ST131FitnessCurve.py` and `ST131Contributions.py` provide model-general entry points for other candidate families. `PlotST131MFE.R` supports the general MFE summaries; the specialized B-spline plotters and optional GP mean/combined plotters are also included. `PlotFeatureTree.py` is the optional Python tree/feature visualization and uses `balticmod.py` and `mutation_states.py`.

## Slurm

`HMCST131Candidates.sbatch` runs one candidate per array task. IDs 0/1 are linear reduced/full, 2/3 unconstrained B-spline, 4/5 monotone B-spline, 6/7 unimodal B-spline, and 8/9 GP.

```bash
sbatch --array=9 \
  --export=ALL,PRIOR=SpikeSlab,CONDA_ENV=/path/to/conda/envs/bayesmfbd \
  HMCST131Candidates.sbatch --spike-slab-inclusion-prob 0.1
```

The supplied batch script retains the research cluster's defaults. Override `CONDA_ENV` or `PYTHON_BIN` for your installation and set `CONDA_MODULE=''` if your cluster does not use a Conda module. Job outputs go into a separate directory per array submission by default.

## Retained inputs and code

The three files documented in [st131-data/readme.md](st131-data/readme.md) are sufficient for the candidate workflow. `HMCST131.py` supplies the shared likelihood; `TreeUtils.py` and `VectorizedTree.py` handle tree preprocessing. `R2D2Prior.py` and `SpikeSlabPrior.py` supply the newer genetic-effect priors.

This is a source/input export. It does not include posterior samples, generated figures, simulation fixtures, or workspace-specific development tests.
