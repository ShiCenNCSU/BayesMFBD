"""Marginalized Normal spike and Normal slab on genetic coefficients.

Both component scales apply directly to raw latent coefficients, giving
each varying mutation and interaction the same prior regardless of frequency.
NUTS samples the continuous mixture; slab probabilities are Rao-Blackwellized.
"""

import jax.numpy as jnp
from jax.nn import sigmoid
import numpy as np
import numpyro
import numpyro.distributions as dist


def validate_spike_slab(inclusion_prob, slab_scale, *, spike_scale=.1,
                        learn_inclusion_prob=False, p_alpha=1., p_beta=1.):
    if (not np.isfinite([inclusion_prob, spike_scale, slab_scale]).all()
            or not 0 < inclusion_prob < 1 or not 0 < spike_scale < slab_scale):
        raise ValueError("SpikeSlab requires 0 < inclusion probability < 1 and finite scales 0 < spike SD < slab SD")
    if learn_inclusion_prob and (not np.isfinite([p_alpha, p_beta]).all() or min(p_alpha, p_beta) <= 0
                                or not np.isfinite(float(p_alpha) + float(p_beta))):
        raise ValueError("SpikeSlab Beta prior shapes must be finite and positive, with a finite sum")


class SpikeSlabPrior:
    def __init__(self, features, feature_names, *, inclusion_prob=.1, slab_scale=1., spike_scale=.1,
                 anchor_index=None, anchor_sign=1, learn_inclusion_prob=False, p_alpha=1., p_beta=1.):
        validate_spike_slab(inclusion_prob, slab_scale, spike_scale=spike_scale,
                            learn_inclusion_prob=learn_inclusion_prob, p_alpha=p_alpha, p_beta=p_beta)
        self.learn_probability, self.p_alpha, self.p_beta = learn_inclusion_prob, p_alpha, p_beta
        self.prior_mean = p_alpha / (p_alpha + p_beta) if learn_inclusion_prob else inclusion_prob
        self.probability = None if learn_inclusion_prob else inclusion_prob
        self.scale, self.spike_scale = slab_scale, spike_scale
        self.names = list(feature_names)
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or not len(x) or x.shape[1] != len(self.names) or not np.isfinite(x).all():
            raise ValueError("SpikeSlab requires finite inference feature rows with matching names")
        self.sd = np.std(x, axis=0, ddof=0)
        self.active = np.flatnonzero(self.sd > 1e-12)
        if not len(self.active):
            raise ValueError("SpikeSlab requires at least one nonconstant predictor")
        if anchor_sign not in (-1, 1):
            raise ValueError("SpikeSlab anchor sign must be -1 or +1")
        if anchor_index is not None and anchor_index not in self.active:
            raise ValueError("SpikeSlab anchor feature must vary across inference rows")
        self.anchor_index, self.anchor_sign = anchor_index, anchor_sign
        self.anchor_position = None if anchor_index is None else list(self.active).index(anchor_index)
        self.free = np.array([i for i, column in enumerate(self.active) if column != anchor_index], dtype=int)

    def sample(self):
        probability = (numpyro.sample("inclusion_prob", dist.Beta(self.p_alpha, self.p_beta))
                       if self.learn_probability else self.probability)
        weights = dist.Categorical(probs=jnp.array([1. - probability, probability]))
        scales = jnp.array([self.spike_scale, self.scale])
        beta = jnp.zeros(len(self.active))
        if len(self.free):
            mixture = dist.MixtureSameFamily(weights, dist.Normal(jnp.zeros(2), scales))
            free = numpyro.sample("spike_slab_coef", mixture.expand((len(self.free),)).to_event(1))
            beta = beta.at[self.free].set(free)
        if self.anchor_position is not None:
            # Conditioning either centered Normal on its sign gives a HalfNormal;
            # both components get the same factor of two, preserving their weights.
            mixture = dist.MixtureSameFamily(weights, dist.HalfNormal(scales))
            anchor = numpyro.sample("spike_slab_anchor", mixture)
            beta = beta.at[self.anchor_position].set(self.anchor_sign * anchor)
        # Retain standardized coefficients as a diagnostic; the prior and
        # likelihood both use beta directly, without predictor-SD rescaling.
        numpyro.deterministic("standardized_coefficients", jnp.zeros(len(self.names)).at[self.active].set(
            beta * jnp.asarray(self.sd[self.active])))
        log_spike = jnp.log1p(-probability) + dist.Normal(0., self.spike_scale).log_prob(beta)
        log_slab = jnp.log(probability) + dist.Normal(0., self.scale).log_prob(beta)
        responsibility = sigmoid(log_slab - log_spike)
        numpyro.deterministic("inclusion_probabilities",
                              jnp.zeros(len(self.names)).at[self.active].set(responsibility))
        return jnp.zeros(len(self.names)).at[self.active].set(beta)

    def initial_values(self, coefficients):
        beta = np.asarray(coefficients)[self.active]
        values = {}
        if self.learn_probability:
            values["inclusion_prob"] = jnp.array(self.prior_mean)
        if len(self.free):
            values["spike_slab_coef"] = jnp.asarray(beta[self.free])
        if self.anchor_position is not None:
            values["spike_slab_anchor"] = jnp.array(max(abs(beta[self.anchor_position]), self.spike_scale / 2.))
        return values

    def metadata(self):
        return dict(
            selection_scope="all_genetic_predictors", parameterization="marginalized_normal_mixture",
            spike="Normal", slab="Normal", spike_scale=self.spike_scale, spike_variance=self.spike_scale**2,
            slab_scale=self.scale, slab_variance=self.scale**2, indicators_marginalized=True,
            component_scales_units="raw latent coefficients", coefficient_scaling="none",
            inclusion_probability=self.probability,
            inclusion_probability_is_fixed=not self.learn_probability,
            inclusion_probability_prior_mean=self.prior_mean,
            inclusion_probability_prior=(dict(distribution="Beta", alpha=self.p_alpha, beta=self.p_beta,
                                               mean=self.prior_mean) if self.learn_probability else None),
            inclusion_probability_site="inclusion_prob" if self.learn_probability else None,
            predictor_sd=self.sd.tolist(),
            predictor_sd_rows="equally weighted inference edges and births; ddof=0; diagnostic only",
            constant_sd_threshold=1e-12,
            active_features=[self.names[i] for i in self.active],
            constant_features=[name for i, name in enumerate(self.names) if i not in self.active],
            normal_features=[self.names[self.active[i]] for i in self.free],
            coefficient_site="spike_slab_coef", responsibility_site="inclusion_probabilities",
            responsibility_estimator="conditional slab probability averaged over posterior draws",
            anchor_feature=None if self.anchor_index is None else self.names[self.anchor_index],
            anchor_sign=self.anchor_sign if self.anchor_index is not None else None,
            anchor_components="signed HalfNormal" if self.anchor_index is not None else None,
            interaction_heredity="none",
        )


def inclusion_summary(probabilities, feature_names, metadata, statistics):
    """Estimate P(component=slab | data) from conditional mixture probabilities."""
    values = np.asarray(probabilities)
    if (values.ndim != 3 or values.shape[-1] != len(feature_names) or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))):
        raise ValueError("Slab probabilities must be finite (chain, draw, feature) values in [0, 1]")
    chains, draws, count = values.shape
    chain_means = values.mean(axis=1)
    ess, rhat = (np.asarray(statistics[key]).reshape(count) for key in ("n_eff", "r_hat"))
    active = set(metadata["active_features"])
    rows = []
    for i, name in enumerate(feature_names):
        eligible = name in active
        pip = float(values[:, :, i].mean())
        rows.append(dict(
            feature=name, eligible_for_selection=eligible,
            prior_inclusion_probability=metadata["inclusion_probability_prior_mean"] if eligible else 0.,
            posterior_inclusion_probability=pip, selected_at_0_5=bool(eligible and pip >= .5),
            n_total_draws=chains * draws, responsibility_ess=float(ess[i]), responsibility_rhat=float(rhat[i]),
            min_chain_inclusion_probability=float(chain_means[:, i].min()),
            max_chain_inclusion_probability=float(chain_means[:, i].max()),
            estimator="mean_conditional_slab_probability",
            mixing_status="continuous_mixture" if eligible else "constant_predictor",
        ))
    return rows
