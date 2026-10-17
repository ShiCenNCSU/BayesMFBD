"""Calculate Hazel-style mutational fitness ratios from ST131 posterior draws.

For each distinct sampled-tip genotype carrying feature j, compare its fitness
to the same genotype with j set to zero. Keep other main features fixed and
recompute interactions. The two predictions use the same posterior draw.
Report ratio-scale posterior summaries and equal-tailed 95% credible intervals.
Use --point-estimator mean for posterior means (the default remains median).
Also summarize the carrier-frequency-weighted mean ratio within each draw.

Run: python ST131MFE.py --results-dir ST131_HMC_candidates
Plot: Rscript PlotST131MFE.R ST131_MFE
"""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

from ete3 import Tree
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.special import ndtr


FAMILIES = ("linear", "bspline_unconstrained", "bspline_monotone", "bspline_unimodal", "gp_monotone")


def main_feature_names(config):
    names = list(config["feature_names"])
    if "main_feature_names" in config:
        main = list(config["main_feature_names"])
        if not main or len(set(main)) != len(main):
            raise ValueError("Saved main-feature names must be nonempty and unique")
        pairs = fitted_interaction_pairs(config, main)
        expected = main + [f"{main[i]}:{main[j]}" for i, j in pairs]
        if names != expected:
            raise ValueError("Feature columns do not match the saved main features and interaction pairs")
        return main
    if config["feature_set"] == "reduced":
        return list(names)
    if config["feature_set"] != "full":
        raise ValueError("Unknown feature set")
    count = int((np.sqrt(1 + 8 * len(names)) - 1) / 2)
    main = names[:count]
    expected = main + [f"{main[i]}:{main[j]}" for i, j in combinations(range(count), 2)]
    if list(names) != expected:
        raise ValueError("Full-model feature columns do not match the inference interaction order")
    return main


def fitted_interaction_pairs(config, names=None):
    """Read explicit fitted pairs, preserving legacy all-pair configurations."""
    names = main_feature_names(config) if names is None else names
    feature_set = config["feature_set"]
    if feature_set == "reduced":
        if config.get("interaction_pairs"):
            raise ValueError("Reduced fits cannot contain interaction pairs")
        return []
    if feature_set != "full":
        raise ValueError("Unknown feature set")
    saved = config.get("interaction_pairs")
    if saved is None:
        return list(combinations(range(len(names)), 2))
    result = []
    for pair in saved:
        if len(pair) != 2 or any(name not in names for name in pair):
            raise ValueError(f"Invalid saved interaction pair: {pair}")
        indices = tuple(names.index(name) for name in pair)
        if indices[0] >= indices[1] or indices in result:
            raise ValueError(f"Duplicate, reversed or self-interaction pair: {pair}")
        result.append(indices)
    return result


def mutation_pairs(observed, names):
    observed = np.asarray(observed)
    if observed.ndim != 2 or observed.shape[1] != len(names) or not len(observed):
        raise ValueError("Observed genotypes must have one row per tip and one column per feature")
    if not np.isin(observed, [0, 1]).all():
        raise ValueError("Sampled-tip genotypes must be binary; do not round fractional ancestral states")
    unique, counts = np.unique(observed.astype(int), axis=0, return_counts=True)
    bits = lambda row: "".join(map(str, row))
    count_by_genotype = {bits(row): int(count) for row, count in zip(unique, counts)}
    rows, mutants, backgrounds = [], [], []
    for index, name in enumerate(names):
        for mutant in unique[unique[:, index] == 1]:
            background = mutant.copy()
            background[index] = 0
            background_count = count_by_genotype.get(bits(background), 0)
            rows.append(dict(mutation=name, mutation_index=index, mutant_genotype=bits(mutant),
                             background_genotype=bits(background), mutant_count=count_by_genotype[bits(mutant)],
                             background_count=background_count, background_observed=background_count > 0))
            mutants.append(mutant)
            backgrounds.append(background)
    columns = ("mutation", "mutation_index", "mutant_genotype", "background_genotype", "mutant_count",
               "background_count", "background_observed")
    return (pd.DataFrame(rows, columns=columns), np.asarray(mutants).reshape(-1, len(names)),
            np.asarray(backgrounds).reshape(-1, len(names)))


def gp_prediction_knots(samples, config):
    """Read GP knots on their recorded scale; older fits use fitness-scale knots."""
    scale = config.get("gp_output_scale", "fitness")
    if scale not in ("log", "fitness"):
        raise ValueError(f"Unknown GP output scale: {scale}")
    key = "log_fitness_knots" if scale == "log" else "fitness_knots"
    if key not in samples:
        raise ValueError(f"Missing GP posterior array: {key}")
    knots = np.asarray(samples[key], dtype=float)
    count = config["gp_knots"]
    if knots.ndim != 3 or not all(knots.shape) or knots.shape[-1] != count or count < 2:
        raise ValueError("GP knots must match the fitted grid and chain/draw dimensions")
    if not np.isfinite(knots).all() or (scale == "fitness" and np.any(knots <= 0)):
        raise ValueError("GP knots must be finite; fitness-scale knots must be positive")
    if np.any(np.diff(knots, axis=-1) < 0):
        raise ValueError("GP knots must be monotone")
    return knots, scale == "log"


def coefficients_are_normalized(config):
    """Legacy nonlinear fits use unit directions; selection/shrinkage retains scale."""
    unnormalized = config.get("prior") in ("R2D2", "SpikeSlab")
    default = "none" if config["model"] == "linear" or unnormalized else "unit_l2"
    rule = config.get("coefficient_normalization", default)
    if rule not in ("none", "unit_l2"):
        raise ValueError(f"Unknown fitted coefficient normalization: {rule}")
    if unnormalized and rule != "none":
        raise ValueError(f"{config['prior']} coefficients must retain their fitted scale")
    return rule == "unit_l2"


def latent_cdf_scale(config):
    """Missing metadata retains the historical Phi(z) reader convention."""
    scale = float(config.get("latent_cdf_scale", 1.))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Fitted latent CDF scale must be finite and positive")
    return scale


def predict_log_fitness(samples, config, genotypes, chunk_size=128):
    """Reconstruct the fitted mapping, preserving normalization and draw pairing."""
    names = main_feature_names(config)
    x = np.asarray(genotypes, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(names) or not np.all(np.isfinite(x)):
        raise ValueError("Prediction genotypes have invalid feature dimensions or values")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if "log_fit_effects" in samples:
        beta = np.asarray(samples["log_fit_effects"], dtype=float)
    else:
        effects = np.asarray(samples["fit_effects"], dtype=float)
        if np.any(effects <= 0):
            raise ValueError("Multiplicative fitness coefficients must be positive")
        beta = np.log(effects)
    if beta.ndim != 3 or beta.shape[2] != len(config["feature_names"]) or not np.all(np.isfinite(beta)):
        raise ValueError("Fitness coefficients must have (chain, draw, feature) dimensions and finite values")
    shape = beta.shape[:2]
    intercept = np.asarray(samples["base_fit"], dtype=float)
    if intercept.shape != shape or not np.all(np.isfinite(intercept)):
        raise ValueError("Baseline draws must match the coefficient chain and draw dimensions")
    beta, intercept = beta.reshape(-1, beta.shape[-1]), intercept.ravel()
    family = config["model"]
    if family not in FAMILIES:
        raise ValueError(f"Unknown model family: {family}")
    if coefficients_are_normalized(config) and not np.allclose(np.linalg.norm(beta, axis=1), 1., atol=1e-6, rtol=0):
        raise ValueError("Nonlinear posterior coefficients must already be normalized as in inference")
    if family.startswith("bspline"):
        coef = np.asarray(samples["bs_coef"], dtype=float)
        count = config["num_splines"]
        if coef.shape != (*shape, count) or count < 5 or not np.all(np.isfinite(coef)):
            raise ValueError("Spline coefficients do not match the fitted basis or posterior draws")
        knots = np.r_[np.zeros(4), np.linspace(0., 1., count - 2)[1:-1], np.ones(4)]
        basis = BSpline(knots, np.eye(count), 3, extrapolate=False)
        coef = coef.reshape(-1, count)
    elif family == "gp_monotone":
        fitness, gp_log_scale = gp_prediction_knots(samples, config)
        count = config["gp_knots"]
        if fitness.shape != (*shape, count):
            raise ValueError("GP knots must match the coefficient chain and draw dimensions")
        grid, fitness = np.linspace(0., 1., count), fitness.reshape(-1, count)
    result = np.empty((len(beta), len(x)))
    pairs = fitted_interaction_pairs(config, names)
    for start in range(0, len(x), chunk_size):
        features = x[start:start + chunk_size]
        if config["feature_set"] == "full":
            interactions = [features[:, i] * features[:, j] for i, j in pairs]
            if interactions:
                features = np.column_stack([features, *interactions])
        latent = beta @ features.T + intercept[:, None]
        if family == "linear":
            predicted = latent
        elif family.startswith("bspline"):
            predicted = np.einsum("sgb,sb->sg", basis(ndtr(latent_cdf_scale(config) * latent)), coef)
        else:
            coordinate = ndtr(latent_cdf_scale(config) * latent)
            predicted = np.stack([np.interp(u, grid, fit) for u, fit in zip(coordinate, fitness)])
            if not gp_log_scale:
                predicted = np.log(predicted)
        result[:, start:start + chunk_size] = predicted
    if not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite genotype fitness predictions")
    return result


def ratio_summary(draws, point_estimator="median"):
    if point_estimator not in ("mean", "median"):
        raise ValueError("point_estimator must be mean or median")
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
    mean = np.mean(draws, axis=0)
    estimate = mean if point_estimator == "mean" else np.median(draws, axis=0)
    return dict(estimated_mfe=estimate, posterior_mean=mean,
                mfe_lower=lower, mfe_upper=upper, probability_above_one=np.mean(draws > 1., axis=0))


def calculate_mfe(samples, config, observed, chunk_size=128, point_estimator="median"):
    names = main_feature_names(config)
    pairs, mutants, backgrounds = mutation_pairs(observed, names)
    shape = np.asarray(samples.get("log_fit_effects", samples.get("fit_effects"))).shape[:2]
    if not len(pairs):
        raise ValueError("No observed genotype carries any selected mutation")
    unique, inverse = np.unique(np.vstack([mutants, backgrounds]), axis=0, return_inverse=True)
    predicted = predict_log_fitness(samples, config, unique, chunk_size)
    mutant_fitness = predicted[:, inverse[:len(pairs)]]
    background_fitness = predicted[:, inverse[len(pairs):]]
    with np.errstate(over="ignore", under="ignore"):
        ratios = np.exp(mutant_fitness - background_fitness)
        bg_fitness = np.exp(background_fitness)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0) or not np.all(np.isfinite(bg_fitness)) or np.any(bg_fitness <= 0):
        raise ValueError("Nonfinite or underflowed fitness ratio/background fitness; no draws were clipped or removed")
    for key, value in ratio_summary(ratios, point_estimator).items():
        pairs[key] = value
    pairs["background_fitness_median"] = np.median(bg_fitness, axis=0)
    pairs["background_fitness_mean"] = np.mean(bg_fitness, axis=0)
    pairs["point_estimator"] = f"posterior_{point_estimator}"
    pairs["posterior_draw_count"] = len(ratios)
    mutation_draws = np.full((len(ratios), len(names)), np.nan)
    records = []
    for index, name in enumerate(names):
        selected = pairs.mutation_index.to_numpy() == index
        count = int(pairs.loc[selected, "mutant_count"].sum())
        record = dict(mutation=name, mutation_index=index, n_carriers=count, n_backgrounds=int(selected.sum()),
                      n_unobserved_backgrounds=int((~pairs.loc[selected, "background_observed"]).sum()),
                      point_estimator=f"posterior_{point_estimator}", aggregation="carrier_frequency_weighted_mean_within_draw",
                      posterior_draw_count=len(ratios), status="estimated" if count else "no_observed_carriers")
        if count:
            weights = pairs.loc[selected, "mutant_count"].to_numpy(dtype=float) / count
            mutation_draws[:, index] = ratios[:, selected] @ weights
            record.update({key: float(value) for key, value in ratio_summary(mutation_draws[:, index], point_estimator).items()})
        else:
            record.update({key: np.nan for key in ("estimated_mfe", "posterior_mean", "mfe_lower", "mfe_upper", "probability_above_one")})
        records.append(record)
    draws = dict(pair_mfe=ratios.reshape(*shape, len(pairs)), mutation_mfe=mutation_draws.reshape(*shape, len(names)),
                 pair_background_fitness=bg_fitness.reshape(*shape, len(pairs)),
                 mutation_names=np.asarray(names), pair_mutation_names=pairs.mutation.to_numpy(dtype=str),
                 background_genotypes=pairs.background_genotype.to_numpy(dtype=str),
                 mutant_genotypes=pairs.mutant_genotype.to_numpy(dtype=str))
    return pairs, pd.DataFrame(records), draws


def file_digest(filename):
    hasher = hashlib.sha256()
    with Path(filename).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def input_path(config, key, override, root):
    if override is not None:
        return override.resolve()
    saved = Path(config[key])
    return saved if saved.exists() else root / "st131-data" / "input" / saved.name


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=root / "ST131_HMC_candidates")
    parser.add_argument("--output-dir", type=Path, default=root / "ST131_MFE")
    parser.add_argument("--candidate", default="all", help="One candidate directory name, or all completed candidates")
    parser.add_argument("--features-file", type=Path, help="Override saved feature path after transferring results")
    parser.add_argument("--tree", type=Path, help="Override saved tree path after transferring results")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--point-estimator", choices=("mean", "median"), default="median",
                        help="Posterior center of the paired ratio draws; use mean for the GP mutation panels")
    parser.add_argument("--no-save-draws", action="store_true",
                        help="Compute summaries from every draw but omit the large paired-draw NPZ exports")
    args = parser.parse_args(argv)
    started = perf_counter()
    try:
        archives = sorted(args.results_dir.glob("*/posterior_samples.npz"))
        if args.candidate != "all":
            archives = [p for p in archives if p.parent.name == args.candidate]
        if not archives or args.chunk_size <= 0:
            raise ValueError("No matching posterior archives, or nonpositive chunk size")
        config = json.loads((archives[0].parent / "run_config.json").read_text())
        names = main_feature_names(config)
        feature_path = input_path(config, "features_file", args.features_file, root)
        tree_path = input_path(config, "tree_file", args.tree, root)
        table = pd.read_csv(feature_path, index_col=0)
        if table.index.has_duplicates:
            raise ValueError("Duplicate node identifiers in feature data")
        tip_names = Tree(str(tree_path), format=1).get_leaf_names()
        observed = table.loc[tip_names, names].to_numpy(dtype=float)
        mutation_pairs(observed, names)  # Validate binary tip genotypes before writing.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        draws_dir = args.output_dir / "draws"
        if not args.no_save_draws:
            draws_dir.mkdir(exist_ok=True)
        pair_tables, mutation_tables, diagnostic_rows, inputs = [], [], [], []
        for number, archive in enumerate(archives, 1):
            print(f"[{number}/{len(archives)}] Calculating MFE: {archive.parent.name}", flush=True)
            settings = json.loads((archive.parent / "run_config.json").read_text())
            if main_feature_names(settings) != names:
                raise ValueError("Candidates must use the same ordered main features")
            for key, override, expected in (("tree_file", args.tree, tree_path), ("features_file", args.features_file, feature_path)):
                if input_path(settings, key, override, root).resolve() != expected.resolve():
                    raise ValueError("Candidates refer to different input data; compare the same ST131 tree and features")
            with np.load(archive) as posterior:
                wanted = ("log_fit_effects", "fit_effects", "base_fit", "bs_coef", "fitness_knots", "log_fitness_knots")
                samples = {key: posterior[key] for key in wanted if key in posterior}
            pairs, mutations, draws = calculate_mfe(samples, settings, observed, args.chunk_size,
                                                    point_estimator=args.point_estimator)
            for frame in (pairs, mutations):
                frame.insert(0, "candidate", archive.parent.name)
                frame.insert(1, "family", settings["model"])
                frame.insert(2, "feature_set", settings["feature_set"])
            pair_tables.append(pairs)
            mutation_tables.append(mutations)
            if not args.no_save_draws:
                np.savez_compressed(draws_dir / f"{archive.parent.name}.npz", **draws)
            diagnostic = dict(candidate=archive.parent.name, family=settings["model"], feature_set=settings["feature_set"],
                              n_chains=draws["mutation_mfe"].shape[0], draws_per_chain=draws["mutation_mfe"].shape[1],
                              divergences=None, min_site_effect_ess=None, max_site_effect_split_rhat=None)
            diagnostic_path = archive.parent / "diagnostics.json"
            if diagnostic_path.exists():
                diagnostic["divergences"] = json.loads(diagnostic_path.read_text())["divergences"]
            summary_path = archive.parent / "HMC_summary_fit_effects.csv"
            if summary_path.exists():
                stats = pd.read_csv(summary_path, index_col=0)
                diagnostic.update(min_site_effect_ess=float(stats.n_eff.min()), max_site_effect_split_rhat=float(stats.r_hat.max()))
            diagnostic_rows.append(diagnostic)
            inputs.append(dict(candidate=archive.parent.name, posterior_file=str(archive.resolve()), sha256=file_digest(archive),
                               n_pairs=len(pairs), n_chains=diagnostic["n_chains"], draws_per_chain=diagnostic["draws_per_chain"]))
            print(f"  {len(pairs)} mutation/background pairs; all posterior draws retained", flush=True)
        pd.concat(pair_tables, ignore_index=True).to_csv(args.output_dir / "pair_summaries.csv", index=False)
        pd.concat(mutation_tables, ignore_index=True).to_csv(args.output_dir / "mutation_summaries.csv", index=False)
        pd.DataFrame(diagnostic_rows).to_csv(args.output_dir / "sampler_flags.csv", index=False)
        metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                        paired_draw_archives_saved=not args.no_save_draws,
                        mfe_definition="exp(h_draw(mutant) - h_draw(background))",
                        point_estimator=f"posterior_{args.point_estimator}_of_actual_ratio_draws",
                        background_point_estimator=f"posterior_{args.point_estimator}_on_fitness_scale",
                        interval="equal_tailed_95_percent_linear_interpolation",
                        mutation_aggregation="mutant_tip_frequency_weighted_mean_ratio_within_each_draw",
                        genotype_source="sampled_tree_tips_only; fractional ancestral states are excluded",
                        branch_effects="cancel in paired comparisons at the same branch effect",
                        n_sampled_tips=len(observed), n_unique_genotypes=len(np.unique(observed, axis=0)),
                        n_feature_rows=len(table), n_ancestral_rows_excluded=len(table) - len(observed),
                        main_feature_names=names, tree_file=str(tree_path.resolve()), tree_sha256=file_digest(tree_path),
                        features_file=str(feature_path.resolve()), features_sha256=file_digest(feature_path),
                        candidates=inputs, numpy_version=np.__version__, pandas_version=pd.__version__)
        (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        exports = "tables" if args.no_save_draws else "tables and draws"
        print(f"Saved MFE {exports} to {args.output_dir}; elapsed={perf_counter() - started:.1f}s", flush=True)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
