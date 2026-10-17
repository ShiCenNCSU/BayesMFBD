"""R2D2 with a Laplace kernel for the genetic index of a tree likelihood.

Implements Zhang et al. (2022), equations (8) and (13):
https://doi.org/10.1080/01621459.2020.1825449
The reference scale replaces Gaussian residual sigma; R2 is a latent variance
budget here, not a calibrated coefficient of determination for the tree model.
"""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist


def validate_r2d2(mean_r2, precision, scale):
    values = np.asarray([mean_r2, precision, scale], dtype=float)
    if not np.isfinite(values).all() or not 0 < mean_r2 < 1 or precision <= 0 or scale <= 0:
        raise ValueError("R2D2 requires 0 < mean_r2 < 1 and finite positive precision and scale")
    a, b = mean_r2 * precision, (1 - mean_r2) * precision
    if not np.isfinite([a, b]).all() or min(a, b) <= 0:
        raise ValueError("R2D2 Beta shape parameters must be finite and positive")
    return a, b


class R2D2Prior:
    """Scale on equally weighted inference rows; export effects in input units."""

    def __init__(self, features, feature_names, *, mean_r2=.5, precision=2., scale=1.,
                 anchor_index=None, anchor_sign=1):
        self.a, self.b = validate_r2d2(mean_r2, precision, scale)
        self.mean_r2, self.precision, self.scale = mean_r2, precision, scale
        self.names = list(feature_names)
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or not len(x) or x.shape[1] != len(self.names) or not np.isfinite(x).all():
            raise ValueError("R2D2 requires finite inference feature rows with matching names")
        self.sd = np.std(x, axis=0, ddof=0)
        self.active = np.flatnonzero(self.sd > 1e-12)
        if not len(self.active):
            raise ValueError("R2D2 requires at least one nonconstant predictor")
        self.anchor_index, self.anchor_sign = anchor_index, anchor_sign
        if anchor_sign not in (-1, 1):
            raise ValueError("R2D2 anchor sign must be -1 or +1")
        if anchor_index is not None and anchor_index not in self.active:
            raise ValueError("R2D2 anchor feature must vary across inference rows")
        self.free = np.array([i for i, column in enumerate(self.active) if column != anchor_index], dtype=int)
        self.anchor_position = None if anchor_index is None else list(self.active).index(anchor_index)

    def sample(self):
        count = len(self.active)
        r2 = numpyro.sample("r2d2_R2", dist.Beta(self.a, self.b))
        omega = numpyro.deterministic("r2d2_omega", r2 / (1 - r2))
        # a = count * a_pi, the original paper's linked concentration.
        phi = (numpyro.sample("r2d2_phi", dist.Dirichlet(jnp.full(count, self.a / count)))
               if count > 1 else numpyro.deterministic("r2d2_phi", jnp.ones(1)))
        local = numpyro.sample("r2d2_local_scale", dist.Exponential(.5).expand((count,)).to_event(1))
        z = jnp.zeros(count)
        if len(self.free):
            free_z = numpyro.sample("r2d2_z", dist.Normal(0., 1.).expand((len(self.free),)).to_event(1))
            z = z.at[self.free].set(free_z)
        if self.anchor_position is not None:
            anchor = numpyro.sample("r2d2_anchor_z", dist.HalfNormal(1.))
            z = z.at[self.anchor_position].set(self.anchor_sign * anchor)
        effects = self.scale * jnp.sqrt(local * phi * omega / 2) * z / jnp.asarray(self.sd[self.active])
        return jnp.zeros(len(self.names)).at[self.active].set(effects)

    def initial_values(self, coefficients):
        count = len(self.active)
        omega = self.mean_r2 / (1 - self.mean_r2)
        # E[local] = 2 and uniform phi make the conditional scale sqrt(omega/count).
        z = np.asarray(coefficients)[self.active] * self.sd[self.active] / (self.scale * np.sqrt(omega / count))
        values = dict(r2d2_R2=jnp.array(self.mean_r2), r2d2_local_scale=jnp.full(count, 2.))
        if count > 1:
            values["r2d2_phi"] = jnp.full(count, 1 / count)
        if len(self.free):
            values["r2d2_z"] = jnp.asarray(z[self.free])
        if self.anchor_position is not None:
            values["r2d2_anchor_z"] = jnp.array(max(abs(z[self.anchor_position]), .01))
        return values

    def metadata(self):
        return dict(
            hierarchy="R2D2 Laplace kernel; noncentered Normal-Exponential mixture",
            mean_r2=self.mean_r2, precision=self.precision, a_r2=self.a, b_r2=self.b,
            dirichlet_concentration=self.a / len(self.active), reference_scale=self.scale,
            interpretation="latent_index_variance_budget; not observed-data R2",
            predictor_sd=self.sd.tolist(), scaling_rows="equally weighted inference edges and births; ddof=0",
            constant_sd_threshold=1e-12,
            active_features=[self.names[i] for i in self.active],
            constant_features=[name for i, name in enumerate(self.names) if i not in self.active],
            normal_features=[self.names[self.active[i]] for i in self.free],
            anchor_feature=None if self.anchor_index is None else self.names[self.anchor_index],
            anchor_sign=self.anchor_sign if self.anchor_index is not None else None,
        )
