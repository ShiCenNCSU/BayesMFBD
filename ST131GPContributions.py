"""Calculate GP component contributions using Hazel's fitted-variation method.

For each posterior draw, project genetic log fitness on the latent index
M + I, obtaining centered components b*M, b*I, and a nonlinear component G.
The unique contribution of a component is the gain in R2 over least squares
using the other two reported components. No candidate model is refitted. Contributions
need not sum to one. Branch random effects are outside this decomposition.
The GP command defaults to interaction coefficients whose pooled 90%
equal-tailed posterior interval excludes zero. Other interactions remain in
the fitted predictions but are left unassigned. The projection and nonlinear
component are constructed with all fitted interactions. Use
--interaction-filter all to recover the original decomposition.

Run: python ST131GPContributions.py --fit-dir ST131_HMC_candidates/gp_monotone_full
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from ete3 import Tree
import numpy as np
import pandas as pd

from ST131MFE import FAMILIES, coefficients_are_normalized, file_digest, fitted_interaction_pairs, input_path, latent_cdf_scale, main_feature_names, predict_log_fitness
import TreeUtils
from VectorizedTree import VectorizedTree


COMPONENTS = ("main", "interaction", "nonlinear")
DRAW_KEYS = ("b", "R2_retained", "unassigned_fraction") + tuple(f"C_{name}" for name in COMPONENTS) + tuple(
    f"var_ratio_{name}" for name in COMPONENTS)


def component_contributions(M, I, h, tol=1e-12, *, projection_interactions=None):
    """Generalize the reference drop-one measure to a filtered interaction index.

    Preserve centering and the projection on M plus all fitted interactions.
    I can contain only selected terms, while projection_interactions retains
    the complete index used to construct b and G. Measure delta R2 between
    all three reported components and the other two; excluded interactions
    remain unassigned. With I equal to the full index, full R2 is one and
    this recovers the reference measure. Constant predictions give NaN
    contributions. Input validation is added; fractions are not normalized.
    """
    M, h = np.asarray(M, dtype=float), np.asarray(h, dtype=float)
    I = np.zeros_like(M) if I is None else np.asarray(I, dtype=float)
    I_all = I if projection_interactions is None else np.asarray(projection_interactions, dtype=float)
    if M.ndim != 2 or M.shape != I.shape or M.shape != h.shape or M.shape != I_all.shape or not all(M.shape):
        raise ValueError("M, I, and h must have matching nonempty (draw, row) dimensions")
    if not all(np.all(np.isfinite(value)) for value in (M, I, I_all, h)):
        raise ValueError("M, I, and h must contain only finite values")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")

    t = M + I_all
    Mc = M - M.mean(axis=1, keepdims=True)
    Ic = I - I.mean(axis=1, keepdims=True)
    tc = t - t.mean(axis=1, keepdims=True)
    hc = h - h.mean(axis=1, keepdims=True)
    denominator = np.sum(tc ** 2, axis=1)
    b = np.divide(np.sum(tc * hc, axis=1), denominator,
                  out=np.zeros(len(h)), where=denominator > tol)
    main = b[:, None] * Mc
    interaction = b[:, None] * Ic
    nonlinear = hc - b[:, None] * tc
    components = (main, interaction, nonlinear)
    var_h = np.var(hc, axis=1)
    result = dict(b=b, main_component=main, interaction_component=interaction, G=nonlinear)
    result["R2_retained"] = np.full(len(h), np.nan)
    result["unassigned_fraction"] = np.full(len(h), np.nan)
    for name, values in zip(COMPONENTS, components):
        result[f"var_ratio_{name}"] = np.divide(
            np.var(values, axis=1), var_h,
            out=np.full(len(h), np.nan), where=var_h > tol)
        result[f"C_{name}"] = np.empty(len(h))

    for draw, y in enumerate(hc):
        sst = np.sum(y ** 2)
        if sst >= tol:
            full_X = np.column_stack([values[draw] for values in components])
            full_coefficients = np.linalg.lstsq(full_X, y, rcond=None)[0]
            full_sse = np.sum((y - full_X @ full_coefficients) ** 2)
            result["unassigned_fraction"][draw] = full_sse / sst
            result["R2_retained"][draw] = 1. - full_sse / sst
        for removed, name in enumerate(COMPONENTS):
            if sst < tol:
                contribution = np.nan
            else:
                X = np.column_stack([values[draw] for j, values in enumerate(components)
                                     if j != removed])
                coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
                sse = np.sum((y - X @ coefficients) ** 2)
                # Nested least-squares fits cannot decrease explained variation.
                # Correct negative floating-point roundoff in their difference.
                contribution = max(0., (sse - full_sse) / sst)
            result[f"C_{name}"][draw] = contribution
    return result


def interaction_selection(samples, config, credible_mass=0.90):
    """Select interactions once, pooling all chains and retained draws.

    These are fitted latent coefficients, not background-specific MFE.
    Exact zero at either interval endpoint does not pass the sign criterion.
    """
    if not np.isfinite(credible_mass) or not 0 < credible_mass < 1:
        raise ValueError("Interaction credible mass must be between zero and one")
    names = main_feature_names(config)
    key = "log_fit_effects" if "log_fit_effects" in samples else "fit_effects"
    if key not in samples:
        raise ValueError("Missing fitted feature coefficients")
    beta = np.asarray(samples[key], dtype=float)
    if beta.ndim != 3 or not all(beta.shape) or beta.shape[-1] != len(config["feature_names"]):
        raise ValueError("Coefficients must have (chain, draw, fitted feature) dimensions")
    if not np.isfinite(beta).all() or (key == "fit_effects" and np.any(beta <= 0)):
        raise ValueError("Coefficients must be finite; exponentiated coefficients must be positive")
    if key == "fit_effects":
        beta = np.log(beta)
    columns = ["coefficient_index", "feature", "beta_mean", "beta_median",
               "beta_ci_lower", "beta_ci_upper", "credible_mass", "significant"]
    if config["feature_set"] != "full":
        return pd.DataFrame(columns=columns)
    values = beta.reshape(-1, beta.shape[-1])[:, len(names):]
    tail = (1. - credible_mass) / 2.
    lower, median, upper = np.quantile(values, [tail, .5, 1. - tail], axis=0, method="linear")
    return pd.DataFrame(dict(coefficient_index=np.arange(len(names), beta.shape[-1]),
                             feature=config["feature_names"][len(names):],
                             beta_mean=values.mean(axis=0), beta_median=median,
                             beta_ci_lower=lower, beta_ci_upper=upper,
                             credible_mass=credible_mass, significant=(lower > 0.) | (upper < 0.)),
                        columns=columns)


def calculate_gp_contributions(samples, config, features, chunk_size=100, progress=False,
                               *, interaction_filter="all", interaction_credible_mass=0.90):
    """Preserve the GP-only API for existing analyses."""
    if config.get("model") != "gp_monotone":
        raise ValueError("This analysis requires a gp_monotone candidate fit")
    return calculate_contributions(samples, config, features, chunk_size, progress,
                                   interaction_filter=interaction_filter,
                                   interaction_credible_mass=interaction_credible_mass)


def calculate_contributions(samples, config, features, chunk_size=100, progress=False,
                            *, interaction_filter="all", interaction_credible_mass=0.90):
    """Reconstruct fitted genetic log fitness and attribute it draw by draw.

    Return scalar arrays with (chain, draw) dimensions. All posterior draws
    are retained; batching limits memory without subsampling or thinning.
    Coefficients use the fitted normalization and ordered interaction columns.
    With a significance filter, h uses every fitted coefficient, while I
    contains only selected interactions. b and G retain the original full
    interaction projection; excluded terms remain unassigned. The library
    default remains all for compatibility; the GP CLI defaults to significant.
    """
    family = config.get("model")
    if family not in FAMILIES:
        raise ValueError(f"Unknown fitted model: {family}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if interaction_filter not in ("all", "significant"):
        raise ValueError("interaction_filter must be all or significant")
    names = main_feature_names(config)
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(names) or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("At least two finite feature rows with the fitted main-feature columns are required")
    coefficient_key = "log_fit_effects" if "log_fit_effects" in samples else "fit_effects"
    gp_key = "log_fitness_knots" if config.get("gp_output_scale") == "log" else "fitness_knots"
    mapping_keys = (gp_key,) if family == "gp_monotone" else (
        ("bs_coef",) if family.startswith("bspline") else ())
    required = (coefficient_key, "base_fit") + mapping_keys
    missing = [key for key in required if key not in samples]
    if missing:
        raise ValueError(f"Missing {family} posterior arrays: {missing}")
    posterior = {key: np.asarray(samples[key], dtype=float) for key in required}
    beta = posterior[coefficient_key]
    if beta.ndim != 3 or not all(beta.shape) or beta.shape[2] != len(config["feature_names"]):
        raise ValueError("Fitness coefficients must have (chain, draw, fitted feature) dimensions")
    if not np.isfinite(beta).all() or (coefficient_key == "fit_effects" and np.any(beta <= 0)):
        raise ValueError("Fitness coefficients must be finite; multiplicative coefficients must be positive")
    if coefficient_key == "fit_effects":
        beta = np.log(beta)
    shape = beta.shape[:2]
    if posterior["base_fit"].shape != shape:
        raise ValueError("Baseline draws must match the coefficient chain and draw dimensions")
    if family == "gp_monotone" and posterior[gp_key].shape != (*shape, config["gp_knots"]):
        raise ValueError("GP knots must match the coefficient draws and fitted knot count")
    if family.startswith("bspline") and posterior["bs_coef"].shape != (*shape, config["num_splines"]):
        raise ValueError("Spline coefficients must match the coefficient draws and fitted basis size")
    pairs = fitted_interaction_pairs(config, names)
    Z = np.column_stack([x[:, i] * x[:, j] for i, j in pairs]) if pairs else np.empty((len(x), 0))
    selection = interaction_selection(samples, config, interaction_credible_mass)
    mask = (selection.significant.to_numpy(dtype=bool) if interaction_filter == "significant"
            else np.ones(len(pairs), dtype=bool))
    draws = {key: np.empty(shape) for key in DRAW_KEYS}
    for chain in range(shape[0]):
        for start in range(0, shape[1], chunk_size):
            stop = min(start + chunk_size, shape[1])
            block = {key: value[chain:chain + 1, start:stop] for key, value in posterior.items()}
            h = predict_log_fitness(block, config, x)
            effects = beta[chain, start:stop]
            M = effects[:, :len(names)] @ x.T
            I_all = effects[:, len(names):] @ Z.T
            I = effects[:, len(names):][:, mask] @ Z[:, mask].T
            result = component_contributions(M, I, h, projection_interactions=I_all)
            if not mask.any():
                # A structurally absent component has exactly zero contribution;
                # retain NaN for constant predictions rather than hiding them.
                result["C_interaction"][np.isfinite(result["C_interaction"])] = 0.
            for key in DRAW_KEYS:
                draws[key][chain, start:stop] = result[key]
            if progress:
                print(f"  Chain {chain + 1}/{shape[0]}: {stop}/{shape[1]} draws processed", flush=True)
    return draws


def summarize_contributions(draws, credible_mass=0.95, prefix="C_"):
    """Pool chains for means, medians, and equal-tailed credible intervals.

    Undefined constant-prediction draws are counted explicitly. The summary
    is conditional on mathematically defined draws if any are undefined.
    """
    if not np.isfinite(credible_mass) or not 0 < credible_mass < 1:
        raise ValueError("credible_mass must be between zero and one")
    rows, shape = [], None
    tail = (1. - credible_mass) / 2.
    for name in COMPONENTS:
        values = np.asarray(draws[f"{prefix}{name}"], dtype=float)
        if values.ndim != 2 or not all(values.shape) or np.isinf(values).any():
            raise ValueError("Contribution arrays must have nonempty (chain, draw) dimensions and no infinities")
        if shape is not None and shape != values.shape:
            raise ValueError("All components must have the same chain and draw dimensions")
        shape = values.shape
        defined = values[np.isfinite(values)]
        mean, median, lower, upper = np.nan, np.nan, np.nan, np.nan
        if len(defined):
            mean, median = np.mean(defined), np.median(defined)
            lower, upper = np.quantile(defined, [tail, 1. - tail], method="linear")
        rows.append(dict(component=name, posterior_mean=mean, posterior_median=median,
                         ci_lower=lower, ci_upper=upper, posterior_mean_percent=100. * mean,
                         posterior_median_percent=100. * median, ci_lower_percent=100. * lower,
                         ci_upper_percent=100. * upper, credible_mass=credible_mass,
                         n_chains=shape[0], draws_per_chain=shape[1], n_total_draws=values.size,
                         n_defined_draws=len(defined), n_undefined_draws=values.size - len(defined)))
    return pd.DataFrame(rows)


def analysis_rows(config, feature_path, tree_path, rows="events"):
    """Choose the inference edge/birth features, tree nodes, or sampled tips."""
    names = main_feature_names(config)
    table = pd.read_csv(feature_path, index_col=0)
    table.index = table.index.astype(str)
    if table.index.has_duplicates:
        raise ValueError("Duplicate node identifiers in the feature table")
    tree = Tree(str(tree_path), format=1)
    node_names = [node.name for node in tree.traverse("postorder")]
    if len(set(node_names)) != len(node_names):
        raise ValueError("Tree nodes must have distinct names")
    if rows == "events":
        intervals = np.asarray(config["time_intervals"], dtype=float) - config["root_time"]
        if intervals.ndim != 1 or not len(intervals) or not np.isfinite(intervals).all() or np.any(np.diff(intervals) <= 0):
            raise ValueError("The fitted time intervals must be finite and strictly increasing")
        selected = table.loc[node_names, names].astype(float)
        if not np.isfinite(selected.to_numpy()).all():
            raise ValueError("Tree-node feature rows must be finite")
        tree = TreeUtils.index_branches(tree)
        tree, times = TreeUtils.add_tree_times(tree)
        if max(times) > intervals[-1] + 0.001:
            raise ValueError("The last fitted time interval ends before the most recent tip")
        vectorized = VectorizedTree(tree, intervals, dict(zip(node_names, selected.to_numpy())),
                                    sampling_dict={name: [0.] for name in node_names})
        identifiers, matrices = [], []
        for event in ("edge", "birth"):
            array = getattr(vectorized, f"{event}_arr")
            matrices.append(np.stack(array["features"].tolist()) if len(array) else np.empty((0, len(names))))
            event_names = node_names if event == "edge" else [
                node.name for node in tree.traverse("postorder") if len(node.children) > 1]
            identifiers.extend(f"{event}:{name}" for name in event_names)
        result = pd.DataFrame(np.vstack(matrices), index=identifiers, columns=names)
        result.index.name = "observation"
        return result
    elif rows == "tips":
        identifiers = tree.get_leaf_names()
    elif rows == "nodes":
        # Match the inference runner's branch_names, including a bifurcating root.
        identifiers = [node.name for node in tree.traverse("postorder") if len(node.children) != 1]
        if "n_branches" in config and len(identifiers) != config["n_branches"]:
            raise ValueError("Tree-node count differs from the fitted n_branches; check the input tree")
    else:
        raise ValueError("rows must be events, nodes, or tips")
    selected = table.loc[identifiers, names].astype(float)
    selected.index.name = "node"
    return selected


def main(argv=None, *, gp_only=True):
    root = Path(__file__).resolve().parent
    description = __doc__ if gp_only else "Calculate main, interaction, and nonlinear contributions from a fitted ST131 candidate."
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_candidate = "gp_monotone_full"
    parser.add_argument("--fit-dir", type=Path, default=root / "ST131_HMC_candidates" / default_candidate,
                        help="Candidate directory containing posterior_samples.npz and run_config.json")
    parser.add_argument("--output-dir", type=Path, default=root / ("ST131_GP_contributions" if gp_only else "ST131_contributions"))
    parser.add_argument("--rows", choices=("events", "nodes", "tips"), default="events",
                        help="Inference edge/birth features (default), tree nodes, or sampled tips; equally weighted rows")
    parser.add_argument("--features-file", type=Path, help="Override the saved feature path after moving cluster results")
    parser.add_argument("--tree", type=Path, help="Override the saved tree path after moving cluster results")
    parser.add_argument("--credible-mass", type=float, default=0.95)
    parser.add_argument("--interaction-filter", choices=("all", "significant"),
                        help="GP default: significant; other families: all. Filter the attribution score, not fitted predictions")
    parser.add_argument("--interaction-credible-mass", type=float, default=0.90,
                        help="Pooled equal-tailed coefficient interval used for interaction selection (default: 0.90)")
    parser.add_argument("--chunk-size", type=int, default=100, help="Posterior draws processed per memory batch")
    args = parser.parse_args(argv)
    started = perf_counter()
    try:
        archive = args.fit_dir / "posterior_samples.npz"
        config_path = args.fit_dir / "run_config.json"
        if not archive.is_file() or not config_path.is_file():
            raise ValueError(f"{args.fit_dir}: posterior_samples.npz and run_config.json from "
                             "HMCST131Candidates.py are required; saved MFE ratio archives are insufficient")
        config = json.loads(config_path.read_text())
        if gp_only and config.get("model") != "gp_monotone":
            raise ValueError("--fit-dir must contain a gp_monotone candidate fit")
        if args.chunk_size < 1 or not np.isfinite(args.credible_mass) or not 0 < args.credible_mass < 1:
            raise ValueError("--chunk-size must be positive and --credible-mass must be between zero and one")
        if not np.isfinite(args.interaction_credible_mass) or not 0 < args.interaction_credible_mass < 1:
            raise ValueError("--interaction-credible-mass must be between zero and one")
        feature_path = input_path(config, "features_file", args.features_file, root)
        tree_path = input_path(config, "tree_file", args.tree, root)
        selected = analysis_rows(config, feature_path, tree_path, args.rows)
        print(f"Calculating contributions: {args.fit_dir.name}; {len(selected)} {args.rows}", flush=True)
        with np.load(archive, allow_pickle=False) as posterior:
            wanted = ("log_fit_effects", "fit_effects", "base_fit", "fitness_knots", "log_fitness_knots", "bs_coef")
            samples = {key: posterior[key] for key in wanted if key in posterior}
        interaction_filter = args.interaction_filter or ("significant" if config["model"] == "gp_monotone" else "all")
        selection = interaction_selection(samples, config, args.interaction_credible_mass)
        selection["included_in_interaction_component"] = (selection.significant if interaction_filter == "significant"
                                                          else True)
        included = selection.loc[selection.included_in_interaction_component, "feature"].tolist()
        print(f"  Interaction attribution: {len(included)}/{len(selection)} terms; filter={interaction_filter}", flush=True)
        draws = calculate_contributions(samples, config, selected.to_numpy(), args.chunk_size, progress=True,
                                        interaction_filter=interaction_filter,
                                        interaction_credible_mass=args.interaction_credible_mass)
        summary = summarize_contributions(draws, args.credible_mass)
        variance_summary = summarize_contributions(draws, args.credible_mass, prefix="var_ratio_")
        n_chains, n_draws = draws["b"].shape
        diagnostic_path = args.fit_dir / "diagnostics.json"
        mapping = ("M + I + base_fit" if config["model"] == "linear" else
                   "BSpline(ndtr(M + I + base_fit), fitted_knots, bs_coef)" if config["model"].startswith("bspline") else
                   "log(interp(ndtr(M + I + base_fit), GP_grid, fitness_knots))")
        if config["model"] == "gp_monotone" and config.get("gp_output_scale") == "log":
            mapping = "interp(ndtr(M + I + base_fit), GP_grid, log_fitness_knots)"
        if latent_cdf_scale(config) != 1.:
            mapping = mapping.replace("ndtr(M + I + base_fit)", f"ndtr({latent_cdf_scale(config):g} * (M + I + base_fit))")
        metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), candidate=args.fit_dir.name,
                        model=config["model"], feature_set=config["feature_set"],
                        reference_method="PhyloJAX_hazel/model_comparison_one_tree.py:component_contributions",
                        contribution_definition="R2_OLS(h_centered ~ all_three_reported_components) - R2_OLS(h_centered ~ other_two_reported_components); no intercept",
                        decomposition="h_centered = b*M_centered + b*I_centered + G",
                        fitted_log_fitness=mapping,
                        gp_output_scale=config.get("gp_output_scale", "fitness") if config["model"] == "gp_monotone" else None,
                        branch_effects_included=False, rows=args.rows, row_weighting="equal_weight_per_selected_row",
                        root_included=args.rows == "events" or (args.rows == "nodes" and
                                      len(Tree(str(tree_path), format=1).children) != 1),
                        n_edge_rows=int(selected.index.str.startswith("edge:").sum()) if args.rows == "events" else None,
                        n_birth_rows=int(selected.index.str.startswith("birth:").sum()) if args.rows == "events" else None,
                        n_observations=len(selected), n_unique_feature_rows=len(selected.drop_duplicates()),
                        main_feature_names=main_feature_names(config), n_chains=n_chains, draws_per_chain=n_draws,
                        n_total_draws=n_chains * n_draws, credible_mass=args.credible_mass,
                        interval="equal_tailed; quantile interpolation=linear", primary_point_estimator="posterior_mean",
                        undefined_draw_policy="report counts; summarize only mathematically defined draws",
                        single_chain=n_chains == 1, posterior_file=str(archive.resolve()),
                        posterior_sha256=file_digest(archive), run_config_sha256=file_digest(config_path),
                        tree_file=str(tree_path.resolve()), tree_sha256=file_digest(tree_path),
                        features_file=str(feature_path.resolve()), features_sha256=file_digest(feature_path),
                        fitted_configuration=config, sampler_diagnostics=json.loads(diagnostic_path.read_text())
                        if diagnostic_path.exists() else None, numpy_version=np.__version__, pandas_version=pd.__version__)
        metadata["interaction_selection"] = dict(
            mode=interaction_filter, credible_mass=args.interaction_credible_mass,
            interval="equal_tailed; pooled across all chains/draws; linear quantiles",
            criterion="beta_ci_lower > 0 or beta_ci_upper < 0",
            coefficient_scale=("normalized " if coefficients_are_normalized(config) else "") + "latent log_fit_effects; not MFE",
            n_total=len(selection), n_selected=len(included), included_features=included,
            selected_once=True, multiple_comparison_adjustment=False,
            excluded_terms="retained in fitted h; left unassigned; not added to G",
            fitted_predictions_filtered=False, coefficients_renormalized=False)
        if interaction_filter == "significant":
            metadata["decomposition"] = "h_centered = b*M_centered + b*I_selected_centered + G + b*I_excluded_centered"
            metadata["projection"] = "b from h_centered ~ (M_centered + I_all_centered); original full-interaction projection"
            metadata["nonlinear_definition"] = "G = h_centered - b*(M_centered + I_all_centered); original global epistasis component preserved"
            metadata["excluded_interaction_policy"] = "unassigned"
            metadata["interaction_r2_regression"] = "delta R2: h_centered ~ b*M_centered + b*I_selected_centered + G versus h_centered ~ b*M_centered + G"
        coverage_rows = []
        for name in ("R2_retained", "unassigned_fraction"):
            values = draws[name][np.isfinite(draws[name])]
            lower, upper = np.quantile(values, [(1.-args.credible_mass)/2., (1.+args.credible_mass)/2.]) if len(values) else (np.nan, np.nan)
            coverage_rows.append(dict(metric=name, posterior_mean=np.mean(values) if len(values) else np.nan,
                                      posterior_median=np.median(values) if len(values) else np.nan,
                                      ci_lower=lower, ci_upper=upper, credible_mass=args.credible_mass,
                                      n_defined_draws=len(values), n_undefined_draws=n_chains*n_draws-len(values)))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_dir / "component_summary.csv", index=False)
        variance_summary.to_csv(args.output_dir / "variance_ratio_summary.csv", index=False)
        pd.DataFrame(coverage_rows).to_csv(args.output_dir / "coverage_summary.csv", index=False)
        np.savez_compressed(args.output_dir / "component_draws.npz", **draws)
        draw_table = pd.DataFrame({key: value.ravel() for key, value in draws.items()})
        draw_table.insert(0, "draw", np.tile(np.arange(n_draws), n_chains))
        draw_table.insert(0, "chain", np.repeat(np.arange(n_chains), n_draws))
        draw_table.to_csv(args.output_dir / "component_draws.csv", index=False)
        selected.to_csv(args.output_dir / "analysis_rows.csv")
        selection.to_csv(args.output_dir / "interaction_selection.csv", index=False)
        (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for row in summary.itertuples(index=False):
            print(f"  {row.component}: mean={row.posterior_mean_percent:.2f}%; "
                  f"{100. * args.credible_mass:g}% CrI [{row.ci_lower_percent:.2f}%, {row.ci_upper_percent:.2f}%]; "
                  f"undefined draws={row.n_undefined_draws}", flush=True)
        print(f"Saved contributions to {args.output_dir}; {n_chains} chain(s), {n_draws} draws per chain; "
              f"elapsed={perf_counter() - started:.1f}s", flush=True)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
