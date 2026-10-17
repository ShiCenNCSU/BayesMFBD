"""Fit Hazel's ten candidate fitness models with the HMCST131 likelihood.

Five fitness mappings are crossed with reduced (main effects) and full
(main effects plus pairwise interactions) feature sets. The extinction
calculation is inherited from HMCST131.MTBD; edge densities, sampling,
birth events, baseline fitness, and optional branch effects follow that
script. Fitness mappings and site-effect priors follow
PhyloJAX_hazel/MTBD_site-effect_model_jax.py.

Run all ten on the bundled ST131 data::

    python HMCST131Candidates.py --model all --feature-set all

Run one candidate::

    python HMCST131Candidates.py --model bspline_monotone --feature-set full

Nonlinear mappings use probit-transformed log site effects. Legacy priors
use unit directions; R2D2 and SpikeSlab retain the coefficient scale.
SpikeSlab applies the same mixture directly to raw mutation and interaction
coefficients, without rescaling them by predictor standard deviations.
For the unconstrained and unimodal splines, --anchor-feature and
--anchor-sign choose the orientation (default: first feature, positive).
Unlike simulations, empirical data do not supply known mutation effects.
Exponentiated nonlinear coefficients parameterize the latent index, rather
than marginal fitness effects. See HMCST131Candidates.md for details.
The monotone GP predicts centered log fitness and exponentiates after
interpolation, with a signed Normal prior on the log-fitness intercept.
"""

import argparse
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.scipy.special import ndtr
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_value
import pandas as pd
from ete3 import Tree

from HMCST131 import MTBD as ST131Likelihood
import TreeUtils
from VectorizedTree import VectorizedTree
from R2D2Prior import R2D2Prior, validate_r2d2
from SpikeSlabPrior import SpikeSlabPrior, inclusion_summary, validate_spike_slab


FAMILIES = (
    "linear", "bspline_unconstrained", "bspline_monotone",
    "bspline_unimodal", "gp_monotone",
)
GP_LATENT_CDF_SCALE = 0.25  # Phi(z / 4), also recorded for posterior readers.
DEFAULT_FEATURES = (
    "gyrAS83L_AMR", "parCE84V_AMR", "gyrAD87NparCS80I_AMR", "parEI529L_AMR",
    "blaCTXM27_AMR", "tet_AMR", "mdtM_AMR", "glpTE448K_AMR",
    "blaCTXM15_AMR", "aac3_AMR", "aac6_AMR", "sul_AMR", "aac6IbcrblaOXAOH_AMR",
)


def make_spline_knots(num_splines, degree=3):
    if num_splines < degree + 2:
        raise ValueError("At least five basis functions are required for cubic splines")
    internal = jnp.linspace(0., 1., num_splines - degree + 1)[1:-1]
    return jnp.concatenate((jnp.zeros(degree + 1), internal, jnp.ones(degree + 1)))


def bspline_basis_jax(x, knots, degree=3):
    """Cox-de Boor basis, including the left-hand derivative at x=1.

    At the right boundary, activate the last interval with nonzero width
    before recursion. Activating the last repeated-knot interval instead
    would produce an all-zero basis after recursion.
    """
    x, knots = jnp.asarray(x), jnp.asarray(knots)
    X = x[:, None]
    left, right = knots[:-1], knots[1:]
    at_right = (X == knots[-1]) & (left < knots[-1]) & (right == knots[-1])
    basis = (((X >= left) & (X < right)) | at_right).astype(x.dtype)
    for k in range(1, degree + 1):
        count = len(knots) - k - 1
        left_width = knots[k:-1] - knots[:count]
        right_width = knots[k + 1:] - knots[1:count + 1]
        left_weight = jnp.where(
            left_width > 0, (X - knots[:count]) / jnp.where(left_width > 0, left_width, 1.), 0.,
        )
        right_weight = jnp.where(
            right_width > 0, (knots[k + 1:] - X) / jnp.where(right_width > 0, right_width, 1.), 0.,
        )
        basis = left_weight * basis[:, :count] + right_weight * basis[:, 1:count + 1]
    return basis


def normalize_log_effects(effects):
    return effects / jnp.sqrt(jnp.maximum(jnp.sum(effects ** 2), 1e-12))


class MTBDCandidates(ST131Likelihood):
    """A pure evaluation of HMCST131's likelihood on locally built event arrays."""

    def __init__(self, feature_names, feature_set="reduced", num_splines=10,
                 interaction_terms=None, **params):
        if feature_set not in ("reduced", "full"):
            raise ValueError("feature_set must be reduced or full")
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("Feature names must be nonempty and unique")
        self.main_feature_names = list(feature_names)
        self.feature_set = feature_set
        self.pairs = list(combinations(range(len(feature_names)), 2)) if feature_set == "full" else []
        self.interaction_mode = "all" if feature_set == "full" else "none"
        if interaction_terms is not None:
            if feature_set != "full" or not interaction_terms:
                raise ValueError("Explicit interactions require feature_set=full and at least one pair")
            self.interaction_mode = "selected"
            self.pairs = []
            for term in interaction_terms:
                parts = term.split(":")
                if len(parts) != 2 or any(name not in self.main_feature_names for name in parts):
                    raise ValueError(f"Unknown or invalid interaction: {term}; use MAIN_FEATURE:MAIN_FEATURE")
                pair = tuple(sorted(self.main_feature_names.index(name) for name in parts))
                if pair[0] == pair[1] or pair in self.pairs:
                    raise ValueError(f"Self-interaction or duplicate interaction: {term}")
                self.pairs.append(pair)
        self.feature_names = list(feature_names) + [f"{feature_names[i]}:{feature_names[j]}" for i, j in self.pairs]
        params["fit_effects"] = np.ones(len(self.feature_names))
        super().__init__(**params)
        intervals = np.asarray(self.time_intervals, dtype=float)
        if intervals.ndim != 1 or not len(intervals) or not np.all(np.isfinite(intervals)) or np.any(np.diff(intervals) <= 0):
            raise ValueError("time_intervals must be finite and strictly increasing, in distance from root")
        if np.any(self.beta <= 0) or np.any(self.d <= 0) or np.any(self.gamma < 0):
            raise ValueError("Birth and death rates must be positive; transition rates must be nonnegative")
        if np.any((self.s < 0) | (self.s > 1)) or np.any((self.rho <= 0) | (self.rho > 1)):
            raise ValueError("Sampling fractions must lie in [0, 1], and rho in (0, 1]")
        self.num_splines = num_splines
        self.full_knots = make_spline_knots(num_splines)

    def build(self, tree, features_dic, sampling_dic=None):
        tree = TreeUtils.index_branches(tree)
        tree, times = TreeUtils.add_tree_times(tree)
        if max(times) > self.time_intervals[-1] + 0.001:
            raise ValueError("The last time interval ends before the most recent tree tip")
        names = [node.name for node in tree.traverse("postorder")]
        missing = sorted(set(names) - set(features_dic))
        if missing:
            raise ValueError(f"Missing genetic features for tree nodes: {missing[:10]}")
        features_dic = {name: np.asarray(features_dic[name], dtype=float) for name in names}
        if any(row.shape != (len(self.main_feature_names),) or not np.all(np.isfinite(row)) for row in features_dic.values()):
            raise ValueError("Each node must have one finite value per selected genetic feature")
        if sampling_dic is not None:
            missing = sorted(set(names) - set(sampling_dic))
            if missing:
                raise ValueError(f"Missing sampling features for tree nodes: {missing[:10]}")
            sampling_dic = {name: np.asarray(sampling_dic[name], dtype=float) for name in names}
            widths = {row.shape for row in sampling_dic.values()}
            if len(widths) != 1 or any(row.ndim != 1 or not np.all(np.isfinite(row)) for row in sampling_dic.values()):
                raise ValueError("Sampling feature rows must be finite and have a consistent width")
        # VectorizedTree zips sampling rows with event rows. A placeholder
        # keeps all events when no sampling covariates have been requested.
        vector_sampling = sampling_dic if sampling_dic is not None else {name: [0.] for name in names}
        vt = VectorizedTree(tree, self.time_intervals, features_dic, sampling_dict=vector_sampling)
        arrays = {}
        for field, values in (("b", self.beta), ("gamma", self.gamma), ("d", self.d), ("s", self.s)):
            arrays[f"edge_{field}"] = values[vt.edge_arr["param_interval"]]
        for key in ("time_step", "back_time", "param_interval"):
            arrays[f"edge_{key}"] = vt.edge_arr[key]
        arrays["birth_b"] = self.beta[vt.birth_arr["param_interval"]]
        arrays["birth_param_interval"] = vt.birth_arr["param_interval"]
        arrays["sample_s"] = self.s[vt.sample_arr["param_interval"]]
        arrays["sample_d"] = self.d[vt.sample_arr["param_interval"]]
        arrays["sample_param_interval"] = vt.sample_arr["param_interval"]
        arrays["csa_rho"] = self.rho[vt.rho_arr["param_interval"]]
        for event in ("edge", "birth"):
            rows = getattr(vt, f"{event}_arr")["features"]
            main = np.stack(rows.tolist()) if len(rows) else np.empty((0, len(self.main_feature_names)))
            interactions = np.column_stack([main[:, i] * main[:, j] for i, j in self.pairs]) if self.pairs else np.empty((len(main), 0))
            arrays[f"{event}_features"] = np.column_stack((main, interactions))
            arrays[f"{event}_branch_idx"] = getattr(vt, f"{event}_arr")["idx"]
        self.n_sampling_features = len(next(iter(sampling_dic.values()))) if sampling_dic is not None else 0
        if sampling_dic is not None:
            for event in ("edge", "sample"):
                rows = getattr(vt, f"{event}_arr")["sampling_features"]
                arrays[f"{event}_sampling_features"] = np.stack(rows.tolist()) if len(rows) else np.empty((0, self.n_sampling_features))
        arrays["parent_br_indexes"] = np.asarray(TreeUtils.get_parent_branch_indexes(tree), dtype=int)
        arrays["parent_dists"] = np.asarray(TreeUtils.get_parent_distances(tree))
        self.branch_names = [node.name for node in tree.traverse("postorder") if len(node.children) != 1]
        self.tree = {key: jnp.asarray(value) for key, value in arrays.items()}
        self.n_edges = len(vt.edge_arr)
        self.n_branches = len(self.branch_names)
        self.features = jnp.concatenate((self.tree["edge_features"], self.tree["birth_features"]))
        return self.tree

    def pointwise_log_like(self, est_params, fitness=None):
        """Edge, removal-sampling, contemporaneous-sampling, then birth terms.

        This retains HMCST131's non-iterative pE calculation and its exact
        zero-density replacement (1e-6). The Brownian term is disabled in
        that script; branch regularization belongs to the NumPyro prior.
        """
        t = dict(self.tree)
        beta = est_params.get("beta", self.beta)
        t["edge_b"] = jnp.asarray(beta)[t["edge_param_interval"]]
        t["birth_b"] = jnp.asarray(beta)[t["birth_param_interval"]]
        if "s" in est_params:
            s = est_params["s"].at[0].set(0.)
            t["edge_s"] = s[t["edge_param_interval"]]
            t["sample_s"] = s[t["sample_param_interval"]]
        if "sampling_effects" in est_params:
            log_sampling = jnp.log(est_params["sampling_effects"])
            t["edge_s"] = t["edge_s"] * jnp.exp(t["edge_sampling_features"] @ log_sampling)
            t["sample_s"] = t["sample_s"] * jnp.exp(t["sample_sampling_features"] @ log_sampling)
        if fitness is None:
            latent = self.features @ est_params["fit_effects"] + est_params.get("base_fit", 0.)
            if "bs_coef" in est_params:
                fitness = jnp.exp(bspline_basis_jax(ndtr(2 * latent), self.full_knots) @ est_params["bs_coef"])
            else:
                fitness = jnp.exp(latent)
        t["edge_b"] = t["edge_b"] * fitness[:self.n_edges]
        t["birth_b"] = t["birth_b"] * fitness[self.n_edges:]
        if "branch_fit_effects" in est_params:
            branch_fitness = jnp.exp(est_params["branch_fit_effects"])
            t["edge_b"] = t["edge_b"] * branch_fitness[t["edge_branch_idx"]]
            t["birth_b"] = t["birth_b"] * branch_fitness[t["birth_branch_idx"]]
        x, y, c, pE = self._calc_PEs(t)
        decay = jnp.exp(-c * t["edge_time_step"])
        denominator = (y + t["edge_b"] * pE) * decay - (x + t["edge_b"] * pE)
        pD = decay * jnp.square((y - x) / denominator)
        pD = jnp.where(pD == 0., 1e-6, pD)
        return jnp.concatenate((
            jnp.log(pD), jnp.log(t["sample_s"] * t["sample_d"]),
            jnp.log(t["csa_rho"]), jnp.log(t["birth_b"]),
        ))

    def log_like(self, est_params, fitness=None):
        return jnp.sum(self.pointwise_log_like(est_params, fitness))


def sample_site_effects(count, prior, *, log_scale=False, anchor_index=None, anchor_sign=1):
    """Preserve Hazel's family-specific bounded site-effect priors."""
    lower, upper = jnp.full(count, 0.0001), jnp.full(count, 5.0)
    if anchor_index is not None:
        lower = lower.at[anchor_index].set(1.0 if anchor_sign > 0 else 0.0001)
        upper = upper.at[anchor_index].set(5.0 if anchor_sign > 0 else 1.)
    if log_scale:
        lower, upper = jnp.log(lower), jnp.log(upper)
    if prior == "Uniform":
        distribution = dist.Uniform(lower, upper)
    else:
        variance = numpyro.sample("var", dist.InverseGamma(1., 1.))
        scale = jnp.full(count, jnp.sqrt(variance))
        center = 0. if log_scale else 1.
        base = dist.Laplace(center, scale) if prior == "Laplace" else dist.Normal(center, scale)
        distribution = dist.TruncatedDistribution(base, low=lower, high=upper)
    raw = numpyro.sample("beta_coef_raw", distribution.to_event(1))
    return raw if log_scale else jnp.log(raw)


def gp_covariance(grid, lengthscale, kernel):
    difference = grid[:, None] - grid[None, :]
    if kernel == "RBF":
        return jnp.exp(-difference ** 2 / (2. * lengthscale))
    distance = jnp.abs(difference) / lengthscale
    if kernel == "matern12":
        return jnp.exp(-distance)
    scaled = jnp.sqrt(3.) * distance
    return (1. + scaled) * jnp.exp(-scaled)


def make_candidate_model(model, family, prior="Laplace", branch_effects=True,
                         anchor_index=0, anchor_sign=1, kernel="RBF", gp_knots=10,
                         r2d2_mean_r2=.5, r2d2_precision=2., r2d2_scale=1.,
                         spike_slab_inclusion_prob=.1, spike_slab_scale=1., spike_slab_spike_scale=.1,
                         spike_slab_learn_inclusion_prob=False, spike_slab_p_alpha=1., spike_slab_p_beta=1.):
    if family not in FAMILIES or prior not in ("Laplace", "Normal", "Uniform", "R2D2", "SpikeSlab"):
        raise ValueError("Unknown model family or site-effect prior")
    if kernel not in ("RBF", "matern12", "matern32") or gp_knots < 2:
        raise ValueError("Unknown GP kernel or fewer than two GP knots")
    if not 0 <= anchor_index < len(model.main_feature_names) or anchor_sign not in (-1, 1):
        raise ValueError("The direction anchor must be a main feature with sign -1 or +1")
    anchored = family in ("bspline_unconstrained", "bspline_unimodal")
    r2d2 = R2D2Prior(model.features, model.feature_names, mean_r2=r2d2_mean_r2,
                     precision=r2d2_precision, scale=r2d2_scale,
                     anchor_index=anchor_index if anchored else None, anchor_sign=anchor_sign) if prior == "R2D2" else None
    spike_slab = SpikeSlabPrior(model.features, model.feature_names,
                               inclusion_prob=spike_slab_inclusion_prob, slab_scale=spike_slab_scale,
                               spike_scale=spike_slab_spike_scale,
                               learn_inclusion_prob=spike_slab_learn_inclusion_prob,
                               p_alpha=spike_slab_p_alpha, p_beta=spike_slab_p_beta,
                               anchor_index=anchor_index if anchored else None,
                               anchor_sign=anchor_sign) if prior == "SpikeSlab" else None

    def probabilistic_model():
        genetic_prior = spike_slab if spike_slab is not None else r2d2
        log_effects = genetic_prior.sample() if genetic_prior is not None else sample_site_effects(
            model.n_features, prior, log_scale=family == "bspline_unimodal",
            anchor_index=anchor_index if anchored else None, anchor_sign=anchor_sign,
        )
        # Keep shrinkage/slab magnitude; unit normalization would remove it.
        if family != "linear" and prior not in ("R2D2", "SpikeSlab"):
            log_effects = normalize_log_effects(log_effects)
        numpyro.deterministic("log_fit_effects", log_effects)
        numpyro.deterministic("fit_effects", jnp.exp(log_effects))
        params = {
            "fit_effects": log_effects,
            "base_fit": numpyro.sample("base_fit", dist.Uniform(jnp.log(0.5), jnp.log(1.5))),
        }
        if branch_effects:
            variance = numpyro.sample(
                "branch_fit_effects_var", dist.InverseGamma(1., 1.)
            )
            z = numpyro.sample(
                "branch_fit_effects_z",
                dist.Normal(0., 1.).expand((model.n_branches,)).to_event(1),
            )
            params["branch_fit_effects"] = numpyro.deterministic(
                "branch_fit_effects", jnp.sqrt(variance) * z
            )
        fitness = None
        if family == "bspline_unconstrained":
            params["bs_coef"] = numpyro.sample("bs_coef", dist.Normal(1., 10.).expand((model.num_splines,)).to_event(1))
        elif family == "bspline_monotone":
            locations = jnp.where(jnp.arange(model.num_splines) == 0, 0., -3.)
            alpha = numpyro.sample("alpha", dist.Normal(locations, 3.).to_event(1))
            increments = jnn.softplus(alpha)
            params["bs_coef"] = numpyro.deterministic("bs_coef", jnp.cumsum(increments) + alpha[0] - increments[0])
        elif family == "bspline_unimodal":
            knots, b, degree = model.full_knots, model.num_splines, 3
            locations = jnp.stack([jnp.mean(knots[i + 2:i + degree + 1]) for i in range(b - 1)])[1:-1]
            c0 = numpyro.sample("c0", dist.Normal(0., 1.))
            fraction = numpyro.sample("turn_fraction", dist.Beta(2., 2.))
            turn = locations[0] + fraction * (locations[-1] - locations[0])
            magnitude = numpyro.sample("log_slope_magnitude", dist.Normal(jnp.log(8.), 1.).expand((b - 3,)).to_event(1))
            derivative = jnp.pad((turn - locations) * jnp.exp(magnitude), (1, 1))
            widths = (knots[degree + 1:degree + b] - knots[1:b]) / degree
            numpyro.deterministic("derivative_coef", derivative)
            params["bs_coef"] = numpyro.deterministic("bs_coef", c0 + jnp.pad(jnp.cumsum(widths * derivative), (1, 0)))
        elif family == "gp_monotone":
            grid = jnp.linspace(0., 1., gp_knots)
            eta = numpyro.sample("eta", dist.LogNormal(jnp.log(0.3), 0.5))
            sigma2 = numpyro.sample("sigma2", dist.HalfNormal(0.5))
            tau2 = numpyro.sample("tau2", dist.HalfNormal(0.1))
            covariance = sigma2 * gp_covariance(grid, eta, kernel) + tau2 * jnp.eye(gp_knots)
            z = numpyro.sample("lambda_knots_z", dist.Normal(0., 1.).expand((gp_knots,)).to_event(1))
            log_derivative = numpyro.deterministic("lambda_knots", jnp.linalg.cholesky(covariance) @ z)
            positive_derivative = jnp.exp(log_derivative)
            increments = 0.5 * (positive_derivative[:-1] + positive_derivative[1:]) * jnp.diff(grid)
            intercept = numpyro.sample("c", dist.Normal(0., 1.))
            integrated = jnp.pad(jnp.cumsum(increments), (1, 0))
            # c is log fitness at latent score zero, whose probit coordinate is 0.5.
            log_fitness_knots = numpyro.deterministic(
                "log_fitness_knots", intercept + integrated - jnp.interp(0.5, grid, integrated),
            )
            numpyro.deterministic("fitness_knots", jnp.exp(log_fitness_knots))
            latent = model.features @ log_effects + params["base_fit"]
            fitness = jnp.exp(jnp.interp(ndtr(GP_LATENT_CDF_SCALE * latent), grid, log_fitness_knots))
        pointwise = model.pointwise_log_like(params, fitness)
        numpyro.deterministic("log_lik", pointwise)
        numpyro.factor("custom_likelihood_factor", jnp.sum(pointwise))

    return probabilistic_model


def initial_values(model, family, prior="Laplace", branch_effects=True, anchor_index=0, anchor_sign=1,
                   kernel="RBF", gp_knots=10, r2d2_mean_r2=.5, r2d2_precision=2., r2d2_scale=1.,
                   spike_slab_inclusion_prob=.1, spike_slab_scale=1., spike_slab_spike_scale=.1,
                   spike_slab_learn_inclusion_prob=False, spike_slab_p_alpha=1., spike_slab_p_beta=1.):
    log_beta = np.full(model.n_features, 0.1 / np.sqrt(model.n_features))
    if family == "bspline_monotone":
        rng = np.random.default_rng(42)
        log_beta = rng.normal(size=model.n_features)
        log_beta /= np.linalg.norm(log_beta)
    if family in ("bspline_unconstrained", "bspline_unimodal"):
        log_beta[anchor_index] = anchor_sign * 0.15
    if family == "linear":
        log_beta[:] = 0.
    values = {"beta_coef_raw": jnp.asarray(log_beta if family == "bspline_unimodal" else np.exp(log_beta)),
              "base_fit": jnp.array(0.)}
    if prior == "R2D2":
        values.pop("beta_coef_raw")
        r2d2 = R2D2Prior(model.features, model.feature_names, mean_r2=r2d2_mean_r2,
                         precision=r2d2_precision, scale=r2d2_scale,
                         anchor_index=anchor_index if family in ("bspline_unconstrained", "bspline_unimodal") else None,
                         anchor_sign=anchor_sign)
        values.update(r2d2.initial_values(log_beta))
    elif prior == "SpikeSlab":
        values.pop("beta_coef_raw")
        spike_slab = SpikeSlabPrior(model.features, model.feature_names,
                                   inclusion_prob=spike_slab_inclusion_prob, slab_scale=spike_slab_scale,
                                   spike_scale=spike_slab_spike_scale,
                                   learn_inclusion_prob=spike_slab_learn_inclusion_prob,
                                   p_alpha=spike_slab_p_alpha, p_beta=spike_slab_p_beta,
                                   anchor_index=anchor_index if family in ("bspline_unconstrained", "bspline_unimodal") else None,
                                   anchor_sign=anchor_sign)
        values.update(spike_slab.initial_values(log_beta))
    elif prior != "Uniform":
        values["var"] = jnp.array(0.5)
    if branch_effects:
        values.update(
            branch_fit_effects_var=jnp.array(0.1),
            branch_fit_effects_z=jnp.zeros(model.n_branches),
        )
    if family == "bspline_unconstrained":
        values["bs_coef"] = jnp.linspace(-0.5, 0.5, model.num_splines)
    elif family == "bspline_monotone":
        values["alpha"] = jnp.full(model.num_splines, jnp.log(jnp.expm1(1. / (model.num_splines - 1)))).at[0].set(-0.5)
    elif family == "bspline_unimodal":
        values.update(c0=jnp.array(-0.25), turn_fraction=jnp.array(0.5),
                      log_slope_magnitude=jnp.full(model.num_splines - 3, jnp.log(8.)))
    elif family == "gp_monotone":
        values.update(eta=jnp.array(0.3), sigma2=jnp.array(0.25), tau2=jnp.array(0.01),
                      lambda_knots_z=jnp.zeros(gp_knots), c=jnp.array(0.))
    return values


def save_results(mcmc, model, output_dir, configuration):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "posterior_samples.npz"
    previous_exports = set()
    if archive_path.exists():
        with np.load(archive_path) as previous:
            previous_exports = {
                f"HMC_{kind}_{name}.csv" for name in previous.files
                if name != "log_lik" and name.isidentifier()
                for kind in ("samples", "summary")
            }
    samples = {name: np.asarray(value) for name, value in mcmc.get_samples(group_by_chain=True).items()}
    np.savez_compressed(archive_path, **samples)
    csv_samples = {
        name: np.exp(value) if name in ("base_fit", "branch_fit_effects") else value
        for name, value in samples.items() if name != "log_lik"
    }
    # Compute summaries from transformed draws, including their standard
    # deviations, instead of exponentiating summaries of log-scale draws.
    statistics = summary(csv_samples, group_by_chain=True)
    n_chains, n_draws = samples["fit_effects"].shape[:2]
    names = {"fit_effects": model.feature_names, "log_fit_effects": model.feature_names,
             "beta_coef_raw": model.feature_names, "branch_fit_effects": model.branch_names}
    if configuration.get("prior") == "R2D2":
        information = configuration["r2d2_prior"]
        names.update(r2d2_phi=information["active_features"],
                     r2d2_local_scale=information["active_features"], r2d2_z=information["normal_features"])
    if configuration.get("prior") == "SpikeSlab":
        information = configuration["spike_slab_prior"]
        names.update(inclusion_probabilities=model.feature_names, standardized_coefficients=model.feature_names,
                     spike_slab_coef=information["normal_features"])
    current_exports = set()
    for name, values in csv_samples.items():
        flattened = values.reshape(n_chains * n_draws, -1)
        columns = names.get(name, [name] if flattened.shape[1] == 1 else [f"{name}_{i}" for i in range(flattened.shape[1])])
        table = pd.DataFrame(flattened, columns=columns)
        table.insert(0, "draw", np.tile(np.arange(n_draws), n_chains))
        table.insert(0, "chain", np.repeat(np.arange(n_chains), n_draws))
        table.to_csv(output_dir / f"HMC_samples_{name}.csv", index=False)
        pd.DataFrame({key: np.asarray(value).reshape(-1) for key, value in statistics[name].items()},
                     index=columns).to_csv(output_dir / f"HMC_summary_{name}.csv")
        current_exports.update((f"HMC_samples_{name}.csv", f"HMC_summary_{name}.csv"))
    if configuration.get("prior") == "SpikeSlab":
        pd.DataFrame(inclusion_summary(samples["inclusion_probabilities"], model.feature_names,
                                        configuration["spike_slab_prior"], statistics["inclusion_probabilities"])).to_csv(
            output_dir / "posterior_inclusion_probabilities.csv", index=False)
    elif {"HMC_samples_inclusion.csv", "HMC_samples_inclusion_probabilities.csv"} & previous_exports:
        (output_dir / "posterior_inclusion_probabilities.csv").unlink(missing_ok=True)
    # Only remove files identified by the previous runner-owned archive.
    # Unrelated files in a reused output directory belong to the user.
    for filename in previous_exports - current_exports:
        (output_dir / filename).unlink(missing_ok=True)
    # Preserve compatibility with diagnostic names from historical Gibbs runs.
    extra = {name.removeprefix("hmc_state."): np.asarray(value)
             for name, value in mcmc.get_extra_fields(group_by_chain=True).items()}
    np.savez_compressed(output_dir / "sampler_diagnostics.npz", **extra)
    diagnostics = {
        "chains": n_chains, "draws_per_chain": n_draws,
        "divergences": int(np.sum(extra["diverging"])),
        "mean_acceptance_probability": float(np.mean(extra["accept_prob"])),
        "mean_tree_steps": float(np.mean(extra["num_steps"])),
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (output_dir / "run_config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    return diagnostics


def run_inference(model, family, output_dir, *, prior="Laplace", branch_effects=True,
                  anchor_index=0, anchor_sign=1, kernel="RBF", gp_knots=10,
                  num_warmup=1000, num_samples=5000, num_chains=1, seed=0,
                  target_accept_prob=0.9, max_tree_depth=10, progress_bar=True, configuration=None,
                  r2d2_mean_r2=.5, r2d2_precision=2., r2d2_scale=1.,
                  spike_slab_inclusion_prob=.1, spike_slab_scale=1., spike_slab_spike_scale=.1,
                  spike_slab_learn_inclusion_prob=False, spike_slab_p_alpha=1., spike_slab_p_beta=1.):
    if num_samples < 4:
        raise ValueError("num_samples must be at least 4 for posterior summaries")
    started = perf_counter()
    sampler_name = "NUTS"
    print(f"  Starting {sampler_name} (including JAX compilation): {num_chains} chain(s), "
          f"{num_warmup} warmup steps and {num_samples} samples per chain", flush=True)
    settings = dict(prior=prior, branch_effects=branch_effects, anchor_index=anchor_index,
                    anchor_sign=anchor_sign, kernel=kernel, gp_knots=gp_knots)
    if prior == "R2D2":
        settings.update(r2d2_mean_r2=r2d2_mean_r2, r2d2_precision=r2d2_precision, r2d2_scale=r2d2_scale)
    elif prior == "SpikeSlab":
        settings.update(spike_slab_inclusion_prob=spike_slab_inclusion_prob, spike_slab_scale=spike_slab_scale,
                        spike_slab_spike_scale=spike_slab_spike_scale,
                        spike_slab_learn_inclusion_prob=spike_slab_learn_inclusion_prob,
                        spike_slab_p_alpha=spike_slab_p_alpha, spike_slab_p_beta=spike_slab_p_beta)
    probabilistic_model = make_candidate_model(model, family, **settings)
    values = initial_values(model, family, **settings)
    nuts = NUTS(probabilistic_model, init_strategy=init_to_value(values=values),
                target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="sequential", progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "accept_prob", "num_steps"))
    jax.block_until_ready(mcmc.get_samples(group_by_chain=True))
    print(f"  Sampling complete in {perf_counter() - started:.1f}s; saving results...", flush=True)
    config = dict(configuration or {})
    config.update(model=family, feature_set=model.feature_set, feature_names=model.feature_names,
                  main_feature_names=model.main_feature_names, interaction_mode=model.interaction_mode,
                  interaction_pairs=[[model.main_feature_names[i], model.main_feature_names[j]] for i, j in model.pairs],
                  n_features=model.n_features, n_branches=model.n_branches,
                  num_splines=model.num_splines, num_warmup=num_warmup, num_samples=num_samples,
                  num_chains=num_chains, seed=seed, target_accept_prob=target_accept_prob,
                  max_tree_depth=max_tree_depth, **settings)
    config.update(coefficient_normalization="unit_l2" if family != "linear" and prior not in ("R2D2", "SpikeSlab") else "none",
                  latent_cdf_scale=(GP_LATENT_CDF_SCALE if family == "gp_monotone" else
                                    2. if family.startswith("bspline") else 1.), sampler=sampler_name)
    if prior == "R2D2":
        config["r2d2_prior"] = R2D2Prior(
            model.features, model.feature_names, mean_r2=r2d2_mean_r2, precision=r2d2_precision,
            scale=r2d2_scale, anchor_index=anchor_index if family in ("bspline_unconstrained", "bspline_unimodal") else None,
            anchor_sign=anchor_sign).metadata()
        config["nonlinear_coefficients"] = "Exponentiated unnormalized R2D2 latent coefficients; not marginal fitness effects"
    elif prior == "SpikeSlab":
        config["spike_slab_prior"] = SpikeSlabPrior(
            model.features, model.feature_names, inclusion_prob=spike_slab_inclusion_prob, slab_scale=spike_slab_scale,
            spike_scale=spike_slab_spike_scale,
            learn_inclusion_prob=spike_slab_learn_inclusion_prob, p_alpha=spike_slab_p_alpha, p_beta=spike_slab_p_beta,
            anchor_index=anchor_index if family in ("bspline_unconstrained", "bspline_unimodal") else None,
            anchor_sign=anchor_sign).metadata()
        config["nonlinear_coefficients"] = "Exponentiated soft spike-and-slab coefficients; not marginal fitness effects"
    if family == "gp_monotone":
        config.update(gp_output_scale="log", gp_reference_coordinate=0.5,
                      gp_prior=dict(c=dict(distribution="Normal", loc=0., scale=1.),
                                    eta=dict(distribution="LogNormal", loc=float(np.log(0.3)), scale=0.5),
                                    sigma2=dict(distribution="HalfNormal", scale=0.5),
                                    tau2=dict(distribution="HalfNormal", scale=0.1),
                                    log_derivative_mean=0.))
    diagnostics = save_results(mcmc, model, output_dir, config)
    print(f"  Saved {family}/{model.feature_set} to {output_dir}; "
          f"divergences={diagnostics['divergences']}; elapsed={perf_counter() - started:.1f}s", flush=True)
    return mcmc


def read_node_features(path, selected):
    frame = pd.read_csv(path, index_col=0)
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate node identifiers in {path}")
    if selected == ["all"]:
        selected = list(frame.columns)
    elif selected == ["amr"]:
        selected = [name for name in frame.columns if name.endswith("_AMR")]
    missing = [name for name in selected if name not in frame.columns]
    if missing:
        raise ValueError(f"Unknown features in {path}: {missing}")
    values = frame[selected].to_numpy(dtype=float)
    return list(selected), dict(zip(frame.index.astype(str), values))


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("all",) + FAMILIES, default="all")
    parser.add_argument("--feature-set", choices=("all", "reduced", "full"), default="all")
    parser.add_argument("--input-dir", type=Path, default=root / "st131-data" / "input")
    parser.add_argument("--tree", default="named.tree_lsd.date.noref.pruned_unannotated.nwk")
    parser.add_argument("--features-file", default="genetic_features_marginal.csv")
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES), help="Selected columns, or all/amr")
    parser.add_argument("--interactions", nargs="+",
                        help="With --feature-set full, include only these MAIN_FEATURE:MAIN_FEATURE pairs; default: all pairs")
    parser.add_argument("--sampling-file", default="meta_features_marginal.csv")
    parser.add_argument("--sampling-features", nargs="+", default=["blood_META"], help="Selected sampling columns, or none")
    parser.add_argument("--root-time", type=float, default=1862.125219)
    parser.add_argument("--time-intervals", nargs="+", type=float, default=[1985.9, 2003., 2013., 2023.], help="Interval endpoints in calendar time; root-time is subtracted")
    parser.add_argument("--beta", nargs="+", type=float, default=[1.000001])
    parser.add_argument("--death-rate", type=float, default=1.)
    parser.add_argument("--sampling-fractions", nargs="+", type=float, default=[0., 0.0001, 0.0001, 0.0001])
    parser.add_argument("--rho", type=float, default=0.0001)
    parser.add_argument("--prior", choices=("Laplace", "Normal", "Uniform", "R2D2", "SpikeSlab"), default="Laplace")
    parser.add_argument("--r2d2-mean-r2", type=float, default=.5,
                        help="R2D2 latent variance-budget Beta mean in (0, 1); not observed-data R2")
    parser.add_argument("--r2d2-precision", type=float, default=2., help="Positive R2D2 Beta precision (a+b)")
    parser.add_argument("--r2d2-scale", type=float, default=1., help="Positive R2D2 reference scale for the genetic latent index")
    inclusion_mode = parser.add_mutually_exclusive_group()
    inclusion_mode.add_argument("--spike-slab-inclusion-prob", type=float, default=.1,
                                help="Fixed prior inclusion probability for every varying genetic predictor; default: 0.1")
    inclusion_mode.add_argument("--spike-slab-learn-inclusion-prob", action="store_true",
                                help="Infer one shared inclusion probability p with a Beta prior (default: Beta(1,1))")
    parser.add_argument("--spike-slab-p-alpha", type=float, default=1., help="Positive alpha shape for the learned p Beta prior")
    parser.add_argument("--spike-slab-p-beta", type=float, default=1., help="Positive beta shape for the learned p Beta prior")
    parser.add_argument("--spike-slab-scale", type=float, default=1.,
                        help="Positive Normal slab SD on every raw latent coefficient; default 1")
    parser.add_argument("--spike-slab-spike-scale", type=float, default=.1,
                        help="Normal spike SD on every raw latent coefficient; default 0.1; must be smaller than slab SD")
    parser.add_argument("--splines", type=positive_int, default=10)
    parser.add_argument("--kernel", choices=("RBF", "matern12", "matern32"), default="RBF")
    parser.add_argument("--gp-knots", type=positive_int, default=10)
    parser.add_argument("--anchor-feature", help="Main feature used to orient free/unimodal spline directions")
    parser.add_argument("--anchor-sign", choices=("positive", "negative"), default="positive")
    parser.add_argument("--no-branch-effects", action="store_true", help="Disable the independent log-normal branch effects used in HMCST131")
    parser.add_argument("--num-warmup", type=positive_int, default=500)
    parser.add_argument("--num-samples", type=positive_int, default=2000)
    parser.add_argument("--num-chains", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--max-tree-depth", type=positive_int, default=10)
    parser.add_argument("--no-progress", action="store_true", help="Disable NUTS progress bars; keep model status messages")
    parser.add_argument("--output-dir", type=Path,
                        help="Prior-specific output root; learned-p SpikeSlab defaults to ST131_HMC_candidates_soft_spike_slab_raw_learn_p")
    parser.add_argument("--check-only", action="store_true", help="Validate every selected candidate's initial density and gradients without sampling")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        suffix = {"R2D2": "_r2d2", "SpikeSlab": "_soft_spike_slab_raw"}.get(args.prior, "")
        if args.prior == "SpikeSlab" and args.spike_slab_learn_inclusion_prob:
            suffix += "_learn_p"
        args.output_dir = root / ("ST131_HMC_candidates" + suffix)
    if args.spike_slab_learn_inclusion_prob and args.prior != "SpikeSlab":
        parser.error("--spike-slab-learn-inclusion-prob requires --prior SpikeSlab")
    if args.interactions is not None and args.feature_set != "full":
        parser.error("--interactions requires --feature-set full")
    if args.num_samples < 4:
        parser.error("--num-samples must be at least 4 for posterior summaries")
    if args.splines < 5 or args.gp_knots < 2:
        parser.error("--splines must be at least 5 and --gp-knots at least 2")
    if not 0 < args.target_accept < 1 or args.seed < 0:
        parser.error("--target-accept must be in (0, 1) and --seed must be nonnegative")
    try:
        if args.prior == "R2D2":
            validate_r2d2(args.r2d2_mean_r2, args.r2d2_precision, args.r2d2_scale)
        elif args.prior == "SpikeSlab":
            validate_spike_slab(args.spike_slab_inclusion_prob, args.spike_slab_scale,
                                spike_scale=args.spike_slab_spike_scale,
                                learn_inclusion_prob=args.spike_slab_learn_inclusion_prob,
                                p_alpha=args.spike_slab_p_alpha, p_beta=args.spike_slab_p_beta)
        names, features = read_node_features(args.input_dir / args.features_file, args.features)
        sampling_names, sampling = ([], None) if args.sampling_features == ["none"] else read_node_features(args.input_dir / args.sampling_file, args.sampling_features)
        anchor = args.anchor_feature or names[0]
        if anchor not in names:
            raise ValueError("--anchor-feature must name a selected main feature")
        params = dict(time_intervals=np.asarray(args.time_intervals) - args.root_time,
                      beta=np.asarray(args.beta), d=args.death_rate, s=np.asarray(args.sampling_fractions), rho=args.rho)
        count = len(params["time_intervals"])
        if len(args.beta) not in (1, count) or len(args.sampling_fractions) not in (1, count):
            raise ValueError("--beta and --sampling-fractions require one value or one per time interval")
        tree_path = args.input_dir / args.tree
        input_tree = Tree(str(tree_path), format=1)
        families = FAMILIES if args.model == "all" else (args.model,)
        feature_sets = ("reduced", "full") if args.feature_set == "all" else (args.feature_set,)
        settings = dict(prior=args.prior, branch_effects=not args.no_branch_effects,
                        anchor_index=names.index(anchor), anchor_sign=1 if args.anchor_sign == "positive" else -1,
                        kernel=args.kernel, gp_knots=args.gp_knots)
        if args.prior == "R2D2":
            settings.update(r2d2_mean_r2=args.r2d2_mean_r2, r2d2_precision=args.r2d2_precision, r2d2_scale=args.r2d2_scale)
        elif args.prior == "SpikeSlab":
            settings.update(spike_slab_inclusion_prob=args.spike_slab_inclusion_prob, spike_slab_scale=args.spike_slab_scale,
                            spike_slab_spike_scale=args.spike_slab_spike_scale,
                            spike_slab_learn_inclusion_prob=args.spike_slab_learn_inclusion_prob,
                            spike_slab_p_alpha=args.spike_slab_p_alpha, spike_slab_p_beta=args.spike_slab_p_beta)
        configuration = dict(tree_file=str(tree_path.resolve()),
                             features_file=str((args.input_dir / args.features_file).resolve()),
                             sampling_file=str((args.input_dir / args.sampling_file).resolve()) if sampling is not None else None,
                             sampling_features=sampling_names, root_time=args.root_time,
                             time_intervals=args.time_intervals, beta=args.beta,
                             death_rate=args.death_rate, sampling_fractions=args.sampling_fractions, rho=args.rho,
                             anchor_feature=anchor, jax_version=jax.__version__, numpyro_version=numpyro.__version__,
                             nonlinear_coefficients="Exponentiated normalized latent direction; not marginal fitness effects")
        total = len(families) * len(feature_sets)
        started = perf_counter()
        current = 0
        print(f"{'Checking' if args.check_only else 'Starting inference for'} {total} candidate(s)", flush=True)
        for family in families:
            for feature_set in feature_sets:
                current += 1
                print(f"[{current}/{total}] {'Checking' if args.check_only else 'Running'} "
                      f"{family}/{feature_set}", flush=True)
                model = MTBDCandidates(names, feature_set=feature_set, num_splines=args.splines,
                                       interaction_terms=args.interactions, **params)
                model.build(input_tree.copy(), features, sampling)
                print(f"  {model.n_features} coefficients, {model.n_branches} branches", flush=True)
                if args.check_only:
                    from numpyro.infer.util import log_density
                    fn = make_candidate_model(model, family, **settings)
                    values = initial_values(model, family, **settings)
                    trace = numpyro.handlers.trace(numpyro.handlers.seed(numpyro.handlers.substitute(fn, data=values), rng_seed=args.seed)).get_trace()
                    continuous, discrete = {}, {}
                    for name, site in trace.items():
                        if site["type"] == "sample" and not site["is_observed"]:
                            (discrete if site["fn"].support.is_discrete else continuous)[name] = site["value"]
                    states = (0, 1) if discrete else (None,)
                    for included in states:
                        indicators = {name: jnp.full_like(value, included) for name, value in discrete.items()}
                        value, gradient = jax.jit(jax.value_and_grad(
                            lambda p: log_density(fn, (), {}, {**indicators, **p})[0]))(continuous)
                        if not np.isfinite(value) or not all(np.all(np.isfinite(g)) for g in jax.tree.leaves(gradient)):
                            raise ValueError(f"Nonfinite initial density or gradient: {family}/{feature_set}")
                        label = "" if included is None else " (all-included)" if included else " (all-excluded)"
                        print(f"  Initial log density{label}: {float(value):.6f}; gradients finite", flush=True)
                else:
                    run_inference(model, family, args.output_dir / f"{family}_{feature_set}", **settings,
                                  num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                                  seed=args.seed, target_accept_prob=args.target_accept, max_tree_depth=args.max_tree_depth,
                                  progress_bar=not args.no_progress, configuration=configuration)
        print(f"Completed {current}/{total} candidate {'checks' if args.check_only else 'fits'} "
              f"in {perf_counter() - started:.1f}s", flush=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
