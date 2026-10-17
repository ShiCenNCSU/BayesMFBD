# ST131 candidate inference

`HMCST131Candidates.py` ports the five fitness mappings from
`PhyloJAX_hazel/MTBD_site-effect_model_jax.py`. Each is fit with either main
effects (`reduced`) or main effects and every pairwise interaction (`full`),
giving ten candidates. With `--feature-set full`, `--interactions` can instead
restrict the fit to explicitly named pairs while retaining all main effects:

| Family | Fitness mapping | Shape |
| --- | --- | --- |
| `linear` | `exp(X @ log_beta + base_fit)` | Multiplicative site effects |
| `bspline_unconstrained` | `exp(B(probit(2 * eta)) @ coefficients)` | Free cubic spline |
| `bspline_monotone` | Same spline mapping, with increasing coefficients | Increasing |
| `bspline_unimodal` | Same spline mapping, with derivative coefficients changing sign once | One maximum; flat endpoint slopes |
| `gp_monotone` | `exp(interp(probit(eta / 4), grid, log_fitness_knots))` | Increasing |

For nonlinear models, `eta = X @ log_beta + base_fit`. Legacy priors normalize
`log_beta` to unit length; R2D2 and SpikeSlab retain its magnitude. The probit transform
is the standard Normal CDF. The current spline fitting path uses `Phi(2*eta)`;
GP uses `Phi(eta / 4)` for every prior and both feature sets. `B` is a cubic
B-spline basis on `[0, 1]`; there are ten basis functions by default.
The GP defaults to ten knots and an RBF kernel, with Matérn-1/2 and
Matérn-3/2 also available. The unimodal model preserves the source's
single-turn derivative construction; it does not require global concavity.

The monotone GP predicts **log fitness**, like the B-spline candidates.
Let `A(u)` be the trapezoid-integrated positive derivative `exp(lambda(u))`
on the GP grid. Its log-fitness knots are `c + A(u) - A(0.5)`, so `c` is
log fitness at latent score zero. The model interpolates these log knots
at `u = Phi(eta / 4)` and exponentiates the interpolated value. Independent
branch effects then multiply the resulting birth rates as before.

The GP priors are `c ~ Normal(0, 1)`,
`eta ~ LogNormal(log(0.3), 0.5)`, `sigma2 ~ HalfNormal(0.5)`, and
`tau2 ~ HalfNormal(0.1)`. Here the sampled GP hyperparameter `eta` is
distinct from the latent additive score also called `eta` above.
`sigma2` and `tau2` enter the log-derivative covariance as variance
parameters: `sigma2 * K + tau2 * I`. The smaller variance prior scales
regularize the derivative before the final exponentiation. The signed
Normal prior on `c` supports fitness below and above one; its prior
median fitness at latent score zero is one.

New GP fits save `gp_output_scale: "log"`, `gp_reference_coordinate: 0.5`,
`latent_cdf_scale: 0.25`, and the GP priors in `run_config.json`.
`GP_LATENT_CDF_SCALE` in the runner controls both the fitted transformation
and this metadata. Dividing the latent input by four changes the model's
effective coefficient scale, so refit to use this transformation. Use a
new output directory to keep the preceding fit available for comparison.
SpikeSlab retains its default slab SD of `1.0`. Each fit exports `log_fitness_knots`
and their exponentiated `fitness_knots`; prediction must interpolate the
log knots before exponentiating. Existing saved fits without
`gp_output_scale` retain their original fitness-scale interpretation in
the summary tools. Readers use the saved `latent_cdf_scale`, with the
historical default `1.0` when that field is absent; older fits keep their
original input transformation.
For new fits, the mean-only R preview requires the draw-based
`fitness_curve/curve_summary.csv`; posterior knot means alone cannot
reconstruct the mean of an exponentiated interpolated log response.

## Likelihood and priors

The extinction-probability method is inherited directly from
`HMCST131.MTBD._calc_PEs`. The new script uses its edge density, birth-event,
sampling-on-removal, and contemporaneous-sampling likelihood terms. It
also retains the replacement of exactly zero edge densities with `1e-6`.
It uses the same non-iterative extinction calculation as the active ST131
script, rather than enabling the unfinished iterative calculation.

Each candidate has the ST131 baseline prior, uniform on
`[log(0.5), log(1.5)]`. Optional branch effects multiply the birth rates on
the log scale and use independent Normal priors with an inferred variance,
as in the active `HMCST131.py`. The Brownian contribution is disabled in
that script and is not added here. Use `--no-branch-effects` to disable
branch effects.

The default site-effect and shape priors follow the Hazel candidates. Linear,
unconstrained spline, monotone spline, and GP models sample multiplicative
site effects in `(0.0001, 5.0)`. Laplace and Normal priors are centered at one,
with variance drawn from `InverseGamma(1, 1)`; Uniform is also available.
The unimodal spline samples bounded log effects with Laplace/Normal priors
centered at zero or a Uniform prior on the log bounds. Nonlinear directions
are normalized with a finite-gradient safeguard at zero.

Unconstrained and unimodal splines need a direction convention. The source
uses known simulation mutation effects. For empirical ST131 data, this
runner anchors the first selected main feature to a positive log coefficient
by default. Set `--anchor-feature NAME --anchor-sign negative` to change it.
This specifies latent orientation, not a biological conclusion about the
mutation's marginal fitness effect.

### R2D2 genetic coefficient prior

Select `--prior R2D2` for any of the five families and either feature set.
`Laplace` remains the default. R2D2 replaces the genetic coefficient prior;
the baseline, independent branch effects, likelihood, and spline/GP shape
priors retain their current definitions.

The implementation in `R2D2Prior.py` uses R2D2 with the original Laplace kernel
of [Zhang et al. (2022), equations (8) and (13)](https://doi.org/10.1080/01621459.2020.1825449):

```text
a = mean_r2 * precision; b = (1 - mean_r2) * precision
R2 ~ Beta(a, b)
omega = R2 / (1 - R2)
phi ~ Dirichlet(a/K, ..., a/K)
psi_j ~ Exponential(rate=1/2)
z_j ~ Normal(0, 1)
beta_j = reference_scale / predictor_sd_j * sqrt(psi_j * phi_j * omega / 2) * z_j
```

`K` is the number of nonconstant fitted predictors, including selected
interaction columns. Integrating out `psi_j` gives a zero-centered Laplace
coefficient prior with scale
`reference_scale * sqrt(phi_j * omega / 2) / predictor_sd_j`.
The concentration is linked to the Beta shape as in the paper: `a = K*a_pi`.
The free/unimodal spline anchor conditions the selected coefficient on its
chosen sign through a HalfNormal standard variate, giving the corresponding
half-Laplace marginal. Monotone and linear models use unrestricted signs.

Predictor standard deviations use equally weighted inference edge and birth
rows with `ddof=0`. Coefficients are converted back to the original feature
units before entering the likelihood, so saved predictions use the original
feature matrix. Columns with standard deviation at most `1e-12` are excluded
from the variance allocation and assigned a zero coefficient. An anchored
feature must vary; a model with no varying predictors is rejected.

This tree likelihood has no Gaussian residual variance. Consequently,
`reference_scale` supplies a fixed reference standard deviation, and the
sampled `r2d2_R2` controls a **latent genetic variance budget**. It is not a
calibrated prior on observed-data R², the component contribution R², or the
fraction explained by genetic versus branch effects. Correlated predictors
and the nonlinear fitness mapping also affect the actual fitted variance.
`phi` contains prior variance allocations, not posterior inclusion
probabilities or fitted component contributions.

R2D2 coefficients retain their magnitude for nonlinear families. Dividing
them by their norm would cancel the global `omega` shrinkage. The existing
Laplace/Normal/Uniform normalization is preserved. R2D2 therefore also
changes the prior over the latent input range; previous bounds derived
from a unit-length coefficient vector do not apply. It has no hard bounds
on signed coefficients or their exponentials. The existing CDF input
mapping is retained.

Options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--r2d2-mean-r2` | `0.5` | Beta mean, strictly between 0 and 1 |
| `--r2d2-precision` | `2.0` | Positive Beta precision `a+b` |
| `--r2d2-scale` | `1.0` | Positive reference scale for the latent index |

All three must be finite. A smaller mean favors a smaller variance budget
at fixed precision; precision controls concentration around that mean.
The defaults give `Beta(1,1)` and `Dirichlet(1/K, ..., 1/K)`.
Main and interaction predictors share this allocation prior; it does not
enforce interaction heredity or reserve a larger budget for main effects.

For example, from `PhyloJAX/`:

```bash
python3 HMCST131Candidates.py \
  --model gp_monotone --feature-set full --prior R2D2 \
  --r2d2-mean-r2 0.2 --r2d2-precision 5 --r2d2-scale 1 \
  --num-warmup 1000 --num-samples 5000 --num-chains 2 \
  --target-accept 0.95 --max-tree-depth 12 \
  --output-dir ST131_HMC_candidates_r2d2
```

Append `--check-only` to validate the initial density and gradients. The
R2D2 CLI default output root is `ST131_HMC_candidates_r2d2`; other priors
retain `ST131_HMC_candidates`. An explicit `--output-dir` overrides this.
As with other runs, reusing the same candidate folder replaces its fit.

For Hazel, transfer both `R2D2Prior.py` and `SpikeSlabPrior.py` alongside
the updated runner and use:

```bash
sbatch --array=9 --export=ALL,PRIOR=R2D2 HMCST131Candidates.sbatch \
  --r2d2-mean-r2 0.2 --r2d2-precision 5 --r2d2-scale 1
```

The existing job-specific cluster output directory applies. No production
inference or cluster job is launched by installing this option.

R2D2 runs export `HMC_samples_r2d2_*` and `HMC_summary_r2d2_*` files,
including `R2`, `omega`, `phi`, local scales, and standard variates. They
export the fitted signed coefficients as `log_fit_effects`, and `exp(beta)`
as `fit_effects`; neither is a background-specific MFE. There is no
`beta_coef_raw` or shared legacy `var` site for this prior.
`run_config.json` records `r2d2_prior` (hyperparameters, predictor scales,
active/constant columns, and interpretation), `coefficient_normalization`,
and `latent_cdf_scale`. Updated MFE, contribution, and fitness-curve readers
preserve these fitted conventions without normalizing again. Current
spline runs record `Phi(2*z)` and GP runs record `Phi(z/4)`; archives without
the CDF-scale field retain the historical reader default `Phi(z)`.

### Soft spike-and-slab variable selection

`--prior SpikeSlab` now uses a **Normal spike and Normal slab** for all
supplied varying main effects and fitted interaction columns. It replaces
the preceding point-mass implementation. Baseline and branch parameters
remain outside this prior. The mixture now applies directly to raw latent
coefficients, giving every varying mutation and interaction the same prior
for a 0-to-1 change in its predictor:

```text
beta_j | p ~ (1-p) * Normal(0, spike_sd^2) + p * Normal(0, slab_sd^2)
spike_sd = 0.1                          # variance 0.01
slab_sd = 1.0                           # variance 1
```

The Normal notation above uses variances; the CLI options specify SDs.
`--spike-slab-spike-scale` controls the spike SD and `--spike-slab-scale`
controls the slab SD. Both must be finite, with `0 < spike_sd < slab_sd`.
Coefficients are not divided by predictor SDs. Rare interactions and common
mutations therefore both have spike variance `0.01` and slab variance `1`
under the defaults. Predictor population SDs from equally weighted inference
edge/birth rows are retained only to detect constant columns and report
standardized diagnostics. No unit-length coefficient normalization is applied.
The spike shrinks effects toward zero; eligible coefficients are no longer
set exactly to zero, and their exponentials need not equal one.

Component indicators are marginalized in the two-Normal mixture density.
NUTS samples continuous coefficients and, when enabled, shared `p`. There
is no Gibbs sweep and no sampled binary inclusion site. The implementation
uses NumPyro's `MixtureSameFamily`, which evaluates the exact mixture
without enumerating all joint assignments. A narrow spike can still create
challenging posterior geometry; sampling checks establish execution, not
convergence of a production fit.

By default, `p` is fixed at `0.1`. To estimate one shared slab probability,
use `--spike-slab-learn-inclusion-prob`, with `p ~ Beta(1,1)` by default.
`--spike-slab-p-alpha` and `--spike-slab-p-beta` set positive Beta shapes.
Their prior mean is `alpha/(alpha+beta)`. The learned flag requires
`--prior SpikeSlab` and cannot be combined with `--spike-slab-inclusion-prob`.
The component scales remain fixed. This shared `p` applies to main effects
and interactions alike; there is no interaction heredity requirement.

The sign anchor for unconstrained/unimodal splines conditions **both**
Normal components on the chosen sign, giving two signed HalfNormals.
The anchor can belong to either the spike or slab. Columns with SD at
most `1e-12` remain fixed zero and ineligible; the anchor must vary and at
least one predictor must vary.

For every posterior draw, the conditional slab probability is:

```text
r_j = p * NormalPDF(beta_j; 0, slab_sd)
      / ((1-p) * NormalPDF(beta_j; 0, spike_sd)
         + p * NormalPDF(beta_j; 0, slab_sd))
PIP_j = posterior mean(r_j)
```

Here `NormalPDF` takes SD as its final argument. The calculation uses log
densities for numerical stability. The mean is a Rao-Blackwellized estimate
of the posterior component probability. It is not a probability of an
exactly nonzero coefficient: continuous mixture draws are almost surely
nonzero. Predictions always use the actual coefficient draw, without
multiplying it by `r_j` or thresholding it.

Run from `PhyloJAX/` with learned `p`:

```bash
python3 HMCST131Candidates.py \
  --model gp_monotone --feature-set full --prior SpikeSlab \
  --spike-slab-learn-inclusion-prob \
  --spike-slab-spike-scale 0.1 --spike-slab-scale 1 \
  --num-warmup 2000 --num-samples 5000 --num-chains 4 \
  --target-accept 0.95 --max-tree-depth 10 \
  --output-dir ST131_HMC_candidates_soft_spike_slab_raw_learn_p
```

GP keeps the `Phi(z/4)` mapping and saves `latent_cdf_scale: 0.25`.
For fixed `p`, replace the learned flag with
`--spike-slab-inclusion-prob 0.1`. These settings apply to every family and
both feature sets. Append `--check-only` to check initial density and
continuous gradients without MCMC.

The new default roots are `ST131_HMC_candidates_soft_spike_slab_raw` and
`ST131_HMC_candidates_soft_spike_slab_raw_learn_p`. Existing result folders
retain their original fits unless explicitly reused as `--output-dir`.
The `SpikeSlab` prior name and learned-p flags are retained. When explicitly
reusing a candidate folder, exports owned by its preceding posterior archive
are replaced or removed; unrelated files are preserved. Existing fits from
the predictor-standardized prior retain their saved coefficients and
interpretation; using the raw-coefficient prior requires a new inference run.

Each fitted candidate writes:

- `HMC_samples_inclusion_probabilities.csv` and archive field
  `inclusion_probabilities`: per-draw conditional slab probabilities for
  every feature, with zero for constant predictors.
- `posterior_inclusion_probabilities.csv`: per-feature PIPs, prior slab
  probabilities, `responsibility_ess`, `responsibility_rhat`, and the range
  of chain-specific means. With learned `p`, the prior probability column
  contains the Beta prior mean. `selected_at_0_5` is a reporting rule and
  does not alter predictions. There are no binary indicator switch counts.
- `HMC_samples_inclusion_prob.csv` and `HMC_summary_inclusion_prob.csv`
  when learning shared `p`; these are distinct from feature-specific PIPs.
- `spike_slab_coef`: raw continuous mixture coefficients for unanchored,
  varying features, with the raw anchor in `spike_slab_anchor` when present.
- `standardized_coefficients`: diagnostic `beta_j * predictor_sd_j` values;
  the mixture prior is applied to raw `beta_j`, not these diagnostics.
- `log_fit_effects`: actual coefficients in original feature units;
  `fit_effects` contains their exponentials. These are not background-specific MFEs.
- `run_config.json`: `component_scales_units: "raw latent coefficients"`,
  `coefficient_scaling: "none"` within `spike_slab_prior`, both component
  scales/variances, diagnostic predictor SDs,
  fixed or learned `p`, marginalization, anchor, and sampler conventions.

MFE, contribution, and fitness-curve readers use the actual coefficient
draws and saved mapping scale. For GP contributions without an additional
credible-interval filter, use `--interaction-filter all`; its existing
default retains the pooled 90% interval filter. Near-constant predictions
can still make variance contribution fractions undefined; their summaries
report these counts and use mathematically defined draws.

For Hazel, copy the updated runner and both prior modules, then submit:

```bash
sbatch --array=9 --export=ALL,PRIOR=SpikeSlab HMCST131Candidates.sbatch \
  --spike-slab-learn-inclusion-prob \
  --spike-slab-spike-scale 0.1 --spike-slab-scale 1
```

## Running inference

Use a Python environment containing JAX, NumPyro, NumPy, pandas, and ETE3.
The project's existing `environment.yml` does not list NumPyro.
For this workspace, the `phydy` Conda environment contains these dependencies.

From the project directory, run all ten:

```bash
conda run --no-capture-output -n phydy python -u HMCST131Candidates.py --model all --feature-set all
```

Fit one candidate with multiple chains:

```bash
conda run --no-capture-output -n phydy python -u HMCST131Candidates.py \
  --model bspline_unimodal --feature-set full \
  --num-warmup 1000 --num-samples 2000 --num-chains 4 \
  --anchor-feature gyrAS83L_AMR --anchor-sign positive
```

Refit the monotone GP with the 13 default ST131 main effects and only
`gyrAS83L_AMR:glpTE448K_AMR`, retaining branch effects:

```bash
conda run --no-capture-output -n phydy python -u HMCST131Candidates.py \
  --model gp_monotone --feature-set full \
  --interactions 'gyrAS83L_AMR:glpTE448K_AMR' \
  --num-warmup 2000 --num-samples 5000 --num-chains 2 \
  --target-accept 0.95 --max-tree-depth 12 \
  --output-dir ST131_HMC_candidates_gp_one_interaction
```

This fits 14 coefficients and writes to
`ST131_HMC_candidates_gp_one_interaction/gp_monotone_full/`. Multiple pairs
can be listed after `--interactions`; both feature names must be selected
main effects. Omit the option to fit all pairs. The saved configuration
records `main_feature_names`, `interaction_mode`, and `interaction_pairs`,
which the MFE, contribution, and fitness-curve tools use to reconstruct
the fitted design matrix. Existing all-pair configurations remain supported.
Append `--check-only` to validate this command without sampling.

Validate data, initial densities, and gradients without running MCMC:

```bash
conda run --no-capture-output -n phydy python -u HMCST131Candidates.py --check-only
```

The runner prints the current candidate (for example, `[3/10] Running
bspline_unconstrained/reduced`), its feature and branch counts, the start
and end of sampling, and the saved output path with elapsed time and
divergences. NumPyro displays warmup and sampling progress bars by default.
`--no-progress` hides those bars but keeps the status messages.
`--no-capture-output` makes Conda stream the output as it is printed.

Default inputs and rate settings match the active ST131 script:

- The original named tree and marginal genetic/sampling features in
  `st131-data/input/`.
- Its 13 selected AMR features: 13 coefficients for reduced candidates
  and 91 for full candidates. Fractional ancestral feature values are
  preserved when constructing interactions.
- Calendar root time `1862.125219`; interval endpoints `1985.9`, `2003`,
  `2013`, and `2023`; birth rate `1.000001`; death rate `1`; sampling
  fractions `0`, `0.0001`, `0.0001`, `0.0001`; present sampling `0.0001`.
- Sampling covariates are loaded for likelihood compatibility, but their
  coefficients and background sampling fractions are fixed during CLI
  inference, as in the active ST131 HMC model. The imported sampling mask
  from the historical data package is not applied by `HMCST131.py` or this
  runner.
- Three chains, 1,000 warmup steps, and 5,000 posterior draws (at least four
  draws per chain are required for summaries). Chains and
  candidates run sequentially. These are configurable defaults, not a
  convergence guarantee.

`--features amr` selects every column ending in `_AMR`; `--features all`
selects every genetic feature. An explicit list is also accepted.
Use `--sampling-features none` for a dataset without sampling covariates.
Input paths and rate/time settings are configurable; see `--help`.
Nonlinear candidates use deterministic interior initial values rather
than requiring simulated truth or a preceding MLE fit.

## Running on Slurm

Copy the project and `st131-data/input/` to the cluster. From the cluster's
PhyloJAX directory, submit:

```bash
sbatch HMCST131Candidates.sbatch
```

The array runs all ten candidates, with at most five jobs running at once.
Each task requests four CPUs, 32 GB RAM, and two days. Each candidate fits
four sequential chains, with 2,000 warmup steps and 5,000 retained draws
per chain (20,000 retained draws per candidate). Target acceptance is
0.95 and maximum tree depth is 12. The CPUs are available for numerical operations; chains run
sequentially as in the inference runner. Jobs use CPU JAX.
These are starting resource settings; adjust them for your cluster and
the runtime of these models. Account and partition can be supplied as
`sbatch --account=ACCOUNT --partition=PARTITION ...`.

The script defaults to the existing NCSU environment at
`/usr/local/usrapps/bjreich/scen2/phydy`, loading the `conda` module.
Override the interpreter and module when needed:

```bash
CONDA_MODULE='' PYTHON_BIN=/path/to/environment/bin/python \
  NUM_WARMUP=3000 NUM_SAMPLES=8000 NUM_CHAINS=4 \
  TARGET_ACCEPT=0.95 MAX_TREE_DEPTH=12 \
  sbatch --time=3-00:00:00 HMCST131Candidates.sbatch
```

The default project directory is `SLURM_SUBMIT_DIR`. If submitting from
elsewhere, set `PROJECT_DIR=/absolute/path/to/PhyloJAX`. `CONDA_ENV` accepts
an environment directory; `PYTHON_BIN` takes precedence. The Python
environment needs the inference dependencies listed above, including
NumPyro.

Jobs write to `ST131_HMC_candidates_cluster/ARRAY_JOB_ID/FAMILY_FEATURE_SET/`,
with stdout/stderr in `st131_candidates_ARRAY_JOB_ID_TASK_ID.out/.err`
in the submission directory. Model and sampling status messages are
streamed; progress bars are disabled for readable batch logs. Set
`OUTPUT_DIR` to choose another results directory. `PRIOR`, `SEED`,
`TARGET_ACCEPT`, and `MAX_TREE_DEPTH` also override the script defaults.
Extra inference arguments can follow the sbatch filename, for example
`sbatch HMCST131Candidates.sbatch --no-branch-effects`. Keep the array's
`--model`, `--feature-set`, and `--output-dir` selections when passing
extra arguments.

| Array task | Candidate |
| --- | --- |
| 0 / 1 | `linear` reduced / full |
| 2 / 3 | `bspline_unconstrained` reduced / full |
| 4 / 5 | `bspline_monotone` reduced / full |
| 6 / 7 | `bspline_unimodal` reduced / full |
| 8 / 9 | `gp_monotone` reduced / full |

For a quick cluster validation, submit
`sbatch HMCST131Candidates.sbatch --check-only`; this validates the ten
candidates without sampling or writing posterior files.

To rerun only the full unconstrained spline, submit task 3:

```bash
sbatch --array=3 HMCST131Candidates.sbatch
```

To limit concurrent fits, override the array throttle, for example
`sbatch --array=0-9%2 HMCST131Candidates.sbatch`.

After every task finishes, compare the saved pointwise likelihoods:

```bash
python CompareST131WAIC.py ST131_HMC_candidates_cluster/ARRAY_JOB_ID
```

Replace `ARRAY_JOB_ID` with the number printed by `sbatch`. Each archive
retains `(chain, draw, likelihood_term)` dimensions; no per-chain files
need merging. Check each candidate's parameter summaries and sampler
diagnostics before interpreting the WAIC ranking. The longer runs and
larger tree-depth limit do not guarantee convergence or resolve posterior
geometry problems automatically.

## Outputs

Each candidate writes to its own directory under `ST131_HMC_candidates/`,
such as `linear_reduced/` or `gp_monotone_full/`. Change the parent with
`--output-dir`. Reusing a candidate directory replaces that run's files
and removes obsolete parameter CSVs identified by its previous posterior
archive. Unrelated files are preserved.

- `posterior_samples.npz`: all native NumPyro posterior arrays, including
  pointwise log likelihoods, with dimensions `(chain, draw, ...)`.
- `HMC_samples_<parameter>.csv`: draws with `chain` and `draw` columns.
  Baseline and branch effects are exponentiated to match existing ST131
  CSV conventions; the archive retains their native log scale.
- `HMC_summary_<parameter>.csv`: summaries computed from the exported
  scale, including standard deviation, intervals, effective sample size,
  and R-hat. Pointwise likelihood arrays are kept only in the archive.
- `sampler_diagnostics.npz` and `diagnostics.json`: divergences,
  acceptance probabilities, and tree-step counts.
- `run_config.json`: input paths, selected features, model, priors,
  direction anchor, rate settings, sampling settings, and library versions.

For nonlinear candidates with the legacy priors, `fit_effects` is the exponentiated normalized
latent direction. It is not the fitness ratio caused by changing a feature
on a particular genetic background. The `beta_coef_raw` scale is
multiplicative except for the unimodal model, whose raw coefficients are
on the log scale, preserving the source model's prior parameterization.

For GP with the legacy priors, let `r = beta_coef_raw` in each posterior draw. The signed latent
coefficient is `beta = log(r) / sqrt(sum(log(r)**2))`; `log_fit_effects`
stores `beta`, while `fit_effects` stores `exp(beta)`. Raw values below 1
give negative latent coefficients. Summary statistics are calculated after
this transformation in each draw; transforming a raw posterior mean does
not give the corresponding transformed posterior mean.

The HMC summary CSVs use NumPyro's default **90% highest-posterior-density
interval**, stored under the labels `5.0%` and `95.0%`. Those labels do not
mean a 95% interval, and the bounds are not equal-tailed quantiles. The
contribution filter below calculates its own 90% equal-tailed interval
directly from all retained coefficient draws.

## WAIC from existing or cluster results

The current runner already saves the required pointwise log likelihoods.
Parameter CSVs and posterior summaries alone are insufficient; use
`posterior_samples.npz`, whose `log_lik` array has shape
`(chain, retained_draw, likelihood_term)`. Warmup is excluded. These terms
contain only the likelihood, excluding parameter and branch-effect priors.
No new inference run or export format is needed for existing candidate
archives that contain `log_lik`.

Compare completed local candidates:

```bash
python CompareST131WAIC.py ST131_HMC_candidates
```

After the cluster array finishes, replace `ARRAY_JOB_ID` with its ID:

```bash
python CompareST131WAIC.py ST131_HMC_candidates_cluster/ARRAY_JOB_ID
```

The command needs NumPy and SciPy, without requiring JAX or ArviZ. It
writes `waic_comparison.csv` in the selected results directory, ordered
from lowest to highest WAIC, with `delta_waic` relative to the lowest
score. `--output /path/to/comparison.csv` changes the destination.
Only candidates with a saved posterior archive are included; wait for
all ten jobs to finish for the complete comparison.

Chains are pooled over posterior draws. For each term, the calculation is
`lppd_i = logmeanexp(log_lik_i)` and
`p_waic_i = var(log_lik_i, ddof=1)`, then
`WAIC = -2 * sum(lppd_i - p_waic_i)`.
The unbiased sample variance matches Hazel's existing comparison code.
The CSV also records `elpd_waic`, `lppd`, `p_waic`, chain/draw counts,
term counts, and the number of terms with variance greater than `0.4`.

The saved order is edge segments, removal-sampling events,
contemporaneous-sampling events, then birth events. This gives WAIC for
the existing tree-event factorization. Edge segmentation depends on the
time intervals, and events within a phylogeny are related. Compare
candidates using the same tree, data, and time grid, and interpret the
result as a score for this factorization. Prediction for held-out tips
or independent trees requires a corresponding likelihood grouping and
validation design. Check divergences, effective sample sizes, and R-hat
before using a score to select a model; high pointwise variance can also
make WAIC unreliable.

## Mutational fitness effects (MFE)

Compute effects from the saved candidate posteriors, then plot in R:

```bash
python ST131MFE.py --results-dir ST131_HMC_candidates --output-dir ST131_MFE
Rscript PlotST131MFE.R ST131_MFE
```

For cluster results, pass `--results-dir ST131_HMC_candidates_cluster/ARRAY_JOB_ID`.
`--candidate FAMILY_FEATURE_SET` processes one candidate. `--tree` and
`--features-file` override saved input paths after transferring results;
otherwise the script uses the saved paths, falling back to matching
filenames under `st131-data/input/`. The computation needs NumPy, SciPy,
pandas, and ETE3. Plotting needs ggplot2, patchwork, jsonlite, svglite, and ragg.

The pair definition follows Hazel's MFE analysis: for each distinct
observed carrier genotype, switch the feature from one to zero, hold
other main features fixed, and recompute the pairwise interaction columns.
Within each paired posterior draw, calculate
`MFE = exp(log_fitness(mutant) - log_fitness(background))`. Keep an unobserved
counterfactual background and flag it. Use the actual ratio draws to
compute the posterior median and equal-tailed 95% credible interval,
following Hazel's current median-based workflow. Diagnostic posterior
means are also exported.

ST131 evaluation uses the 882 sampled tips with binary feature states,
rather than treating fractional ancestral probabilities as observed
genotypes. The 13 fitted AMR features include some combined mutation/gene
indicators: the estimate applies to the complete encoded feature.
The predictions retain the fitted baseline inside each nonlinear mapping,
the saved normalized coefficients, and the fitted spline/GP mapping.
Multiplicative branch effects cancel in a paired comparison with the
same branch effect.

Each feature also has an additional summary: average its genotype-specific
ratios within each draw using the observed mutant-tip counts as weights,
then summarize those weighted draws. This estimates the average MFE among
the sampled carriers of that feature. It is distinct from an average over
all possible backgrounds or all sampled tips.

Outputs under `ST131_MFE/` include:

- `pair_summaries.csv`: background-specific estimates, intervals,
  posterior means, posterior median background fitness, carrier counts,
  and background-observation flags.
- `mutation_summaries.csv`: weighted feature estimates and intervals,
  with the aggregation definition and number of carriers/backgrounds.
- `draws/CANDIDATE.npz`: every paired ratio and weighted feature draw,
  retaining `(chain, draw, pair_or_feature)` dimensions and genotype keys.
- `sampler_flags.csv`: saved divergences, chain counts, minimum site-effect
  ESS, and maximum site-effect split R-hat. Fits are retained irrespective
  of these diagnostics; no inference is rerun.
- `analysis_metadata.json`: definitions, data selection, input hashes,
  posterior hashes, feature order, and sample counts.
- `figures/by_mutation/`: one figure per fitted mutation/AMR feature in
  PDF, SVG, and PNG, plus the combined 13-page `all_mutations.pdf`. Each
  mutation figure contains panels for all ten candidates. The x coordinate
  is posterior median background fitness, and the y coordinate is the
  posterior median draw-paired MFE. Vertical bars show the complete 95%
  credible interval for MFE; uncertainty in the x coordinate is not shown.
  Filled/open points identify observed/unobserved counterfactual backgrounds.
  The y-axis is logarithmic, and both axis ranges vary by model to display
  all points and intervals clearly. Points represent distinct carrier
  genotypes without weighting, jitter, smoothing, or line connections.
  `figure_index.csv` lists the figure files and counts; `panel_audit.csv`
  records the plotted definitions and checks for each mutation/model panel.

These background-fitness plots replace the previous plotting script's
weighted-summary forest plots. Previously generated forest-plot files may
still be present directly under `figures/`; the current plots are under
`figures/by_mutation/`.

These are conditional effects under the fitted models. MFE above one
indicates greater modeled fitness after adding the feature, and MFE below
one indicates lower modeled fitness. Posterior medians do not resolve
sampler mixing problems or uncertainty in unobserved counterfactual
genotypes. The existing single-chain outputs need further convergence
assessment before drawing biological conclusions.

## GP component contributions and credible intervals

`ST131GPContributions.py` applies the fitted-variation decomposition from
`PhyloJAX_hazel/model_comparison_one_tree.py` to an existing ST131 GP fit.
It works with `gp_monotone_full` or `gp_monotone_reduced` and does not run
MCMC or refit reduced models.

GP contribution commands now default to `--interaction-filter significant`:
only interactions whose **signed `log_fit_effects`** have a pooled
equal-tailed 90% posterior credible interval excluding zero enter `I_selected`.
The equivalent neutral value on the exponentiated `fit_effects` scale is 1.
The selection uses all retained draws from all chains once, rather than the
interval columns in the existing HMC summary CSVs. Change the selection
interval with `--interaction-credible-mass`; `--credible-mass` separately
controls the contribution summary intervals. There is no multiplicity
adjustment. Use `--interaction-filter all` for the earlier decomposition.

From the project directory:

```bash
python ST131GPContributions.py \
  --fit-dir ST131_HMC_candidates/gp_monotone_full \
  --output-dir ST131_GP_contributions
```

For cluster output, point `--fit-dir` at the completed candidate directory,
for example `ST131_HMC_candidates_cluster/JOB_ID/gp_monotone_full`.
The directory must contain `posterior_samples.npz` and `run_config.json`.
Saved MFE ratio draws and posterior summary tables do not contain enough
information for this calculation. After transferring results, use
`--tree PATH` and `--features-file PATH` if the saved input paths no longer
exist. To run this analysis in the minimal cluster package, also copy
`ST131GPContributions.py` and `ST131MFE.py` there; it already contains the
required `TreeUtils.py` and `VectorizedTree.py`.

Within each posterior draw, the script computes the main-effect index
`M = X @ beta_main`, the interaction index `I = Z @ beta_interaction`,
and genetic log fitness
`h = interp(ndtr(cdf_scale * (M + I + base_fit)), grid, log_fitness_knots)`
for log-scale GP fits. Here `cdf_scale` is the saved `latent_cdf_scale`,
`0.25` for new fits and historically `1.0`. For legacy GP fits it uses
`h = log(interp(ndtr(cdf_scale * (M + I + base_fit)), grid, fitness_knots))`.
The saved nonlinear coefficients retain their fitted normalization.
Branch random effects
are excluded, matching the reference's decomposition of genetic fitness.
The baseline remains inside the GP's input transformation.

The fitted `h` always uses all original coefficients, with their saved
normalization. The projection always uses the complete interaction index:
after centering across feature rows, project `h` onto `t = M + I_all` to
obtain the original slope `b` and nonlinear component `G`. Report the
selected interaction component separately and leave excluded interactions
unassigned; do not add them to `G`, change fitted predictions or re-normalize
coefficients. The exact decomposition is:

```text
h_centered = b*M_centered + b*I_selected_centered + G + b*I_excluded_centered
G = h_centered - b*(M_centered + I_all_centered)
```

Figures label the original global epistasis component `G` **Nonlinear**.
The reported components are `b*M_centered`, `b*I_selected_centered` and `G`.
Excluded interactions are absent from the attribution regressions. If no
interactions pass the criterion, `C_I` is exactly zero for every defined draw.

For each of the three components, least squares predicts centered `h`
using all three reported components and then the other two. Its unique
contribution is `C_component = R2_retained - R2_without_component`.
This equals the reference's `1 - R2_without_component` when all interactions
are included, because the complete decomposition fits centered `h` exactly.
With filtered interactions, `1 - R2_without_component` would include
unassigned variation, so the full-minus-reduced definition is necessary.
`coverage_summary.csv` reports the retained regression's R2 and its
unassigned fraction `1 - R2_retained`. This measures fitted log-fitness
variation uniquely attributable to that component after the other two
components can adjust their regression coefficients. It is distinct from
comparing WAIC across separately fitted candidates or removing a mutation
to calculate MFE.

By default, `--rows events` uses the inference feature matrix: edge rows
followed by birth-event rows, including repeated feature vectors. Every
row has equal weight; weights do not depend on branch length. This matches
the Hazel comparison's row selection. `--rows nodes` instead uses one row
per indexed tree node, and `--rows tips` uses one row per sampled tip. The
analysis population affects contribution values, so report the chosen
row set. Fractional ancestral features are retained for event/node analyses.

The posterior mean is the primary estimate, matching the reference; the
median is also exported. The default interval is the equal-tailed 95%
credible interval across all retained draws from all chains. Change it
with `--credible-mass 0.90`, for example. `--chunk-size 100` controls the
number of draws processed together without thinning or discarding draws.
The script prints the candidate, processing progress, and final percentages.

Outputs include:

- `component_summary.csv`: one row each for `main`, `interaction`, and
  `nonlinear`, with posterior means, medians, credible bounds, their
  percentage equivalents, and counts of defined/undefined draws.
- `variance_ratio_summary.csv`: separate summaries of raw component
  variance divided by total fitted log-fitness variance. These are
  magnitudes, distinct from unique drop-one contributions.
- `component_draws.npz`: every `C_main`, `C_interaction`, `C_nonlinear`,
  variance-ratio draw, projection slope `b`, retained-model R2 and
  unassigned fraction, with `(chain, draw)` axes.
- `coverage_summary.csv`: pooled intervals for `R2_retained` and
  `unassigned_fraction`; excluded variation is not forced into a reported component.
- `component_draws.csv`: the same scalar draws, with explicit `chain` and
  `draw` columns for use by the R plotting script.
- `analysis_rows.csv`: the ordered input features actually used, with
  observation identifiers such as `edge:NODE` and `birth:NODE`.
- `interaction_selection.csv`: signed coefficient means, medians, pooled
  credible bounds, significance flags, and inclusion flags for each interaction.
- `analysis_metadata.json`: formulas, row weighting, input hashes, fit
  configuration, chain/draw counts, and saved sampler diagnostics when available.

Unique contributions need not sum to 100%, since correlated components
can explain shared variation. Raw variance ratios can exceed one and
also need not sum to one. Neither measure is rescaled to force a sum of
100%. If fitted fitness is constant within a draw, its contributions are
undefined; such draws are counted explicitly, and summaries use only the
mathematically defined draws. These intervals describe uncertainty under
the fitted posterior and do not establish MCMC convergence or causality.

To plot the full GP contributions and uncertainty in ternary coordinates:

```bash
Rscript PlotST131GPContributions.R ST131_GP_contributions
```

This exports PDF, SVG, and 300-dpi PNG under `ST131_GP_contributions/figures/`.
Because unique contributions need not sum to one, each draw is normalized
by its own sum before plotting. The ternary chart therefore describes
relative shares of the three unique contributions. Its means and credible
intervals are recalculated from these normalized draws; they are distinct
from the original contribution estimates in `component_summary.csv`.
All draws appear as points, with a diamond marking the posterior mean.
For two-dimensional posterior support, the shaded outline is an approximate
95% density region: a two-dimensional
KDE is evaluated on a 350-by-350 grid, and its threshold is the fifth
percentile of the interpolated density at the actual posterior draws.
The density grid is masked outside the simplex; no posterior draws are
discarded. When support lies on a line, such as when `C_I` is zero, a thick
segment shows the equal-tailed 95% interval along that line instead of a
two-dimensional density region. Vertex labels show means and marginal equal-tailed 95% credible
intervals, which are distinct from simultaneous joint intervals.
`normalized_draws.csv`, `normalized_summary.csv`, and figure metadata
record the transformation, posterior coverage, and complete draw counts.

## Verification

```bash
python -m pytest tests -q
```

The tests compare likelihood values and gradients with `HMCST131.py`,
check feature/event preservation and spline boundary derivatives, evaluate
all prior/kernel choices, and run short NUTS/export checks for all ten
candidates. Short sampling tests verify execution, not posterior mixing.
The comparison tests check WAIC against hand-calculated values and run a
short real inference through the array script with simulated Slurm variables.
MFE tests verify genotype pairing, carrier-count weighting, full-model
interaction updates, ratio-scale medians, GP prediction scale, sampled-tip
selection, and agreement with the fitted JAX spline mapping.
Contribution tests check known variance fractions, correlated components,
reduced GP fits, cross-chain draw pairing, credible intervals, undefined
draw accounting, and agreement with the actual inference feature matrix.
Soft spike-and-slab tests check marginalized Normal densities, component
probabilities, sign-conditioned anchors, fixed/learned shared probabilities,
analytic Gaussian-convolution posteriors, and export migration.
