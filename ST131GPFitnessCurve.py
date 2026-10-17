"""Summarize a fitted ST131 fitness mapping on its latent additive scale.

Uses all retained posterior draws and the exact fitted response: new GP log
fitness knots are interpolated at Phi(latent) and then exponentiated. Legacy
GP fits interpolate fitness-scale knots. Spline coefficients map Phi(latent)
to log fitness, which is then exponentiated. The full-model latent
score includes fitted main and pairwise effects plus base_fit. R2D2/SpikeSlab
retain coefficient magnitude; legacy nonlinear fits use unit directions. Bands are
pointwise credible intervals for genetic fitness at a fixed latent score,
excluding independent branch random effects. Linear fits use exp(latent).

Run: python ST131GPFitnessCurve.py
Other fitted families: python ST131FitnessCurve.py --fit-dir FIT_DIR
Plot: Rscript PlotST131GPFitnessCurve.R ST131_GP_fitness_curve
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ete3 import Tree
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.special import ndtr

from ST131MFE import FAMILIES, coefficients_are_normalized, file_digest, fitted_interaction_pairs, gp_prediction_knots, input_path, latent_cdf_scale, main_feature_names, predict_log_fitness


def latent_additive_scores(samples, config, genotypes):
    """Return (chain, draw, genotype) scores on the fitted latent scale."""
    if config.get("model") not in FAMILIES:
        raise ValueError("Unknown fitted model family")
    names = main_feature_names(config)
    x = np.asarray(genotypes, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(names) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Genotypes must contain finite fitted main-feature columns")
    if "log_fit_effects" in samples:
        beta = np.asarray(samples["log_fit_effects"], dtype=float)
    else:
        effects = np.asarray(samples["fit_effects"], dtype=float)
        if np.any(effects <= 0):
            raise ValueError("Multiplicative coefficients must be positive")
        beta = np.log(effects)
    if beta.ndim != 3 or not all(beta.shape) or beta.shape[-1] != len(config["feature_names"]) or not np.isfinite(beta).all():
        raise ValueError("Coefficients must have finite (chain, draw, fitted feature) dimensions")
    if coefficients_are_normalized(config) and not np.allclose(np.linalg.norm(beta, axis=-1), 1., atol=1e-6, rtol=0):
        raise ValueError("Nonlinear coefficients must already be normalized as in inference")
    base = np.asarray(samples["base_fit"], dtype=float)
    if base.shape != beta.shape[:2] or not np.isfinite(base).all():
        raise ValueError("Intercepts must match the coefficient chain/draw dimensions")
    if config["feature_set"] == "full":
        x = np.column_stack([x] + [x[:, i] * x[:, j] for i, j in fitted_interaction_pairs(config, names)])
    return np.einsum("cdf,nf->cdn", beta, x) + base[..., None]


def fitness_at_latent(samples, config, latent):
    """Evaluate the fitted response on the fitness scale with draw pairing.

    A one-dimensional latent grid is shared by all draws. A three-dimensional
    array supplies a different latent score per chain, draw, and genotype.
    """
    family = config.get("model")
    if family not in FAMILIES:
        raise ValueError("Unknown fitted model family")
    if family == "gp_monotone":
        knots, gp_log_scale = gp_prediction_knots(samples, config)
        count = config["gp_knots"]
        shape = knots.shape[:2]
    elif family.startswith("bspline"):
        coef = np.asarray(samples["bs_coef"], dtype=float)
        count = config["num_splines"]
        if coef.ndim != 3 or not all(coef.shape) or coef.shape[-1] != count or count < 5 or not np.isfinite(coef).all():
            raise ValueError("Spline coefficients must match the fitted basis and chain/draw dimensions")
        shape = coef.shape[:2]
        knots = np.r_[np.zeros(4), np.linspace(0., 1., count - 2)[1:-1], np.ones(4)]
        basis = BSpline(knots, np.eye(count), 3, extrapolate=False)
    else:
        base = np.asarray(samples["base_fit"], dtype=float)
        if base.ndim != 2 or not all(base.shape) or not np.isfinite(base).all():
            raise ValueError("Baseline draws must have finite chain/draw dimensions")
        shape = base.shape
    latent = np.asarray(latent, dtype=float)
    if not np.isfinite(latent).all() or not latent.size:
        raise ValueError("Latent scores must be finite and nonempty")
    shared_grid = latent.ndim == 1
    if not shared_grid and (latent.ndim != 3 or latent.shape[:2] != shape):
        raise ValueError("Latent scores must be a grid or match the posterior chain/draw dimensions")
    if family.startswith("bspline"):
        weights = basis(ndtr(latent_cdf_scale(config) * latent))
        log_fitness = (np.einsum("gb,cdb->cdg", weights, coef) if shared_grid else
                       np.einsum("cdgb,cdb->cdg", weights, coef))
        with np.errstate(over="ignore", under="ignore"):
            fitness = np.exp(log_fitness)
    else:
        if shared_grid:
            latent = np.broadcast_to(latent, (*shape, len(latent)))
        if family == "linear":
            with np.errstate(over="ignore", under="ignore"):
                fitness = np.exp(latent)
        else:
            coordinates = ndtr(latent_cdf_scale(config) * latent).reshape(-1, latent.shape[-1])
            grid = np.linspace(0., 1., count)
            fitness = np.stack([np.interp(u, grid, values) for u, values in
                                zip(coordinates, knots.reshape(-1, count))]).reshape(latent.shape)
            if gp_log_scale:
                with np.errstate(over="ignore", under="ignore"):
                    fitness = np.exp(fitness)
    if not np.isfinite(fitness).all() or np.any(fitness <= 0):
        raise ValueError("Nonfinite or underflowed fitness; no draws were clipped or removed")
    return fitness


def summarize_curve(draws, latent_grid, credible_mass=0.95, cdf_scale=1.):
    """Pool retained chains for mean estimates and equal-tailed intervals."""
    draws = np.asarray(draws, dtype=float)
    latent_grid = np.asarray(latent_grid, dtype=float)
    if not 0 < credible_mass < 1 or draws.ndim != 3 or draws.shape[-1] != len(latent_grid):
        raise ValueError("Invalid credible mass or curve dimensions")
    if not np.isfinite(draws).all() or np.any(draws <= 0):
        raise ValueError("Fitness draws must be finite and positive")
    pooled = draws.reshape(-1, draws.shape[-1])
    tail = (1. - credible_mass) / 2.
    lower, median, upper = np.quantile(pooled, [tail, 0.5, 1. - tail], axis=0, method="linear")
    return pd.DataFrame(dict(latent_score=latent_grid, cdf_coordinate=ndtr(cdf_scale * latent_grid),
                             fitness_median=median, fitness_mean=np.mean(pooled, axis=0),
                             fitness_lower=lower, fitness_upper=upper))


def main(argv=None, gp_only=True):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_candidate = "gp_monotone_full"
    parser.add_argument("--fit-dir", type=Path, default=root / "ST131_HMC_candidates" / default_candidate)
    parser.add_argument("--output-dir", type=Path, default=root / ("ST131_GP_fitness_curve" if gp_only else "ST131_fitness_curve"))
    parser.add_argument("--features-file", type=Path, help="Override the saved feature path")
    parser.add_argument("--tree", type=Path, help="Override the saved tree path")
    parser.add_argument("--grid-size", type=int, default=501)
    parser.add_argument("--credible-mass", type=float, default=0.95)
    parser.add_argument("--no-save-draws", action="store_true",
                        help="Use every draw but export only small CSV/JSON plotting summaries")
    args = parser.parse_args(argv)
    try:
        if args.grid_size < 3 or not 0 < args.credible_mass < 1:
            raise ValueError("--grid-size must be at least 3; --credible-mass must be between 0 and 1")
        archive = args.fit_dir / "posterior_samples.npz"
        config_path = args.fit_dir / "run_config.json"
        config = json.loads(config_path.read_text())
        if gp_only and config.get("model") != "gp_monotone":
            raise ValueError("Use ST131FitnessCurve.py for non-GP fits")
        print(f"Loading fitted {config['model']}: {args.fit_dir.name}", flush=True)
        with np.load(archive) as saved:
            samples = {k: saved[k] for k in ("log_fit_effects", "fit_effects", "base_fit", "fitness_knots", "log_fitness_knots", "bs_coef") if k in saved}
        feature_path = input_path(config, "features_file", args.features_file, root)
        tree_path = input_path(config, "tree_file", args.tree, root)
        names = main_feature_names(config)
        table = pd.read_csv(feature_path, index_col=0)
        table.index = table.index.astype(str)
        if table.index.has_duplicates:
            raise ValueError("Duplicate node identifiers in the feature table")
        tip_names = Tree(str(tree_path), format=1).get_leaf_names()
        if len(set(tip_names)) != len(tip_names):
            raise ValueError("Duplicate sampled-tip identifiers")
        observed = table.loc[tip_names, names].to_numpy(dtype=float)
        if not np.isin(observed, [0, 1]).all():
            raise ValueError("Sampled-tip feature values must be binary")
        unique, inverse, counts = np.unique(observed, axis=0, return_inverse=True, return_counts=True)
        latent = latent_additive_scores(samples, config, unique)
        genotype_fitness = fitness_at_latent(samples, config, latent)
        reference = predict_log_fitness(samples, config, unique).reshape(latent.shape)
        if not np.allclose(np.log(genotype_fitness), reference, atol=1e-12, rtol=1e-12):
            raise ValueError("Response mapping differs from the fitted ST131 prediction")
        n_chains, draws_per_chain = latent.shape[:2]
        print(f"Summarizing all {n_chains * draws_per_chain} draws; {len(tip_names)} tips, {len(unique)} distinct genotypes", flush=True)
        extent = np.array([latent.min(), latent.max()])
        padding = max(np.diff(extent).item() * 0.05, 0.05)
        grid = np.linspace(extent[0] - padding, extent[1] + padding, args.grid_size)
        fitness = fitness_at_latent(samples, config, grid)
        summary = summarize_curve(fitness, grid, args.credible_mass, cdf_scale=latent_cdf_scale(config))
        if config["model"] in ("gp_monotone", "bspline_monotone", "linear") and np.any(np.diff(fitness, axis=-1) < -1e-12):
            raise ValueError("A reconstructed monotone response is not monotone")
        tail = (1. - args.credible_mass) / 2.
        latent_q = np.quantile(latent.reshape(-1, len(unique)), [tail, 0.5, 1. - tail], axis=0)
        fitness_q = np.quantile(genotype_fitness.reshape(-1, len(unique)), [tail, 0.5, 1. - tail], axis=0)
        bits = np.array(["".join(str(int(v)) for v in row) for row in unique])
        genotypes = pd.DataFrame(dict(genotype=bits, sampled_tips=counts,
                                     latent_lower=latent_q[0], latent_median=latent_q[1], latent_upper=latent_q[2],
                                     latent_mean=np.mean(latent, axis=(0, 1)),
                                     fitness_lower=fitness_q[0], fitness_median=fitness_q[1], fitness_upper=fitness_q[2],
                                     fitness_mean=np.mean(genotype_fitness, axis=(0, 1))))
        tips = genotypes.iloc[inverse].reset_index(drop=True)
        tips.insert(0, "tip", tip_names)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_dir / "curve_summary.csv", index=False)
        genotypes.to_csv(args.output_dir / "genotype_summary.csv", index=False)
        tips.to_csv(args.output_dir / "tip_summary.csv", index=False)
        if not args.no_save_draws:
            np.savez_compressed(args.output_dir / "curve_draws.npz", latent_grid=grid, fitness_curve=fitness,
                                genotype_latent=latent, genotype_fitness=genotype_fitness, genotypes=bits,
                                tip_genotype_index=inverse, tip_names=np.asarray(tip_names))
        diagnostic_path = args.fit_dir / "diagnostics.json"
        diagnostics = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {}
        mappings = {"gp_monotone": "interp(Phi(latent_score), linspace(0, 1, gp_knots), fitness_knots)",
                    "linear": "exp(latent_score)"}
        if config.get("gp_output_scale") == "log":
            mappings["gp_monotone"] = "exp(interp(Phi(latent_score), linspace(0, 1, gp_knots), log_fitness_knots))"
        mapping = mappings.get(config["model"], "exp(cubic_BSpline(Phi(latent_score)) @ bs_coef)")
        if latent_cdf_scale(config) != 1.:
            mapping = mapping.replace("Phi(latent_score)", f"Phi({latent_cdf_scale(config):g} * latent_score)")
        normalization = "normalized " if coefficients_are_normalized(config) else ""
        metadata = dict(candidate=args.fit_dir.name, model=config["model"], feature_set=config["feature_set"],
                        main_feature_names=names, n_fitted_features=len(config["feature_names"]),
                        created_utc=datetime.now(timezone.utc).isoformat(), n_sampled_tips=len(tips),
                        n_unique_genotypes=len(genotypes), n_chains=n_chains, draws_per_chain=draws_per_chain,
                        n_total_draws=n_chains * draws_per_chain, n_excluded_draws=0, n_excluded_tips=0,
                        latent_definition=f"base_fit + {normalization}main effects" +
                                          (f" + {normalization}pairwise interaction effects" if config["feature_set"] == "full" else ""),
                        latent_range=grid[[0, -1]].tolist(), range_rule="all sampled-genotype latent draws plus 5 percent padding",
                        mapping=mapping,
                        gp_output_scale=config.get("gp_output_scale", "fitness") if config["model"] == "gp_monotone" else None,
                        point_estimator="posterior_mean_on_fitness_scale", credible_mass=args.credible_mass,
                        interval="pointwise equal-tailed credible interval at a fixed latent score",
                        branch_random_effects_included=False, rug="posterior mean latent score for every sampled tip",
                        n_grid_points=args.grid_size, diagnostics=diagnostics,
                        paired_draw_archives_saved=not args.no_save_draws,
                        input_hashes=dict(posterior_sha256=file_digest(archive), config_sha256=file_digest(config_path),
                                          features_sha256=file_digest(feature_path), tree_sha256=file_digest(tree_path)))
        (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved curve tables{' and paired draws' if not args.no_save_draws else ''} to {args.output_dir}", flush=True)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
