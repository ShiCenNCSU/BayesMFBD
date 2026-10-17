"""Summarize fitness ratios from ablating one fitted interaction at a time.

For sampled genotypes carrying both mutations, compare the full prediction
with the same prediction after setting only that interaction coefficient to
zero. Main effects, other interactions, the mapping and draw pairing are fixed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ete3 import Tree
import numpy as np
import pandas as pd

from ST131GPFitnessCurve import fitness_at_latent, latent_additive_scores
from ST131MFE import file_digest, fitted_interaction_pairs, input_path, main_feature_names, ratio_summary


def calculate_interaction_mfe(samples, config, observed, *, progress=False):
    names = main_feature_names(config)
    interactions = fitted_interaction_pairs(config, names)
    if not interactions:
        raise ValueError("The fit must contain at least one interaction")
    x = np.asarray(observed)
    if x.ndim != 2 or not len(x) or x.shape[1] != len(names) or not np.isin(x, [0, 1]).all():
        raise ValueError("Sampled-tip genotypes must be nonempty and binary with the fitted main columns")
    unique, counts = np.unique(x.astype(int), axis=0, return_counts=True)
    beta = np.asarray(samples["log_fit_effects"] if "log_fit_effects" in samples else np.log(samples["fit_effects"]))
    shape = beta.shape[:2]
    latent = latent_additive_scores(samples, config, unique)
    full_fitness = fitness_at_latent(samples, config, latent)
    interaction_draws = np.full((*shape, len(interactions)), np.nan)
    pair_tables, records = [], []
    bits = ["".join(map(str, row)) for row in unique]
    for k, (i, j) in enumerate(interactions):
        label = f"{names[i]}:{names[j]}"
        selected = np.flatnonzero((unique[:, i] == 1) & (unique[:, j] == 1))
        count = int(counts[selected].sum())
        record = dict(interaction=label, interaction_index=k, mutation_1=names[i], mutation_2=names[j],
                      n_carriers=count, n_backgrounds=len(selected), posterior_draw_count=int(np.prod(shape)),
                      point_estimator="posterior_mean", aggregation="carrier_frequency_weighted_mean_within_draw",
                      status="estimated" if count else "no_observed_co_carriers")
        if len(selected):
            # The pair indicator is one for every selected genotype. No other
            # coefficient or predictor is changed and there is no renormalization.
            off_latent = latent[..., selected] - beta[..., len(names) + k, None]
            without = fitness_at_latent(samples, config, off_latent).reshape(-1, len(selected))
            with_term = full_fitness[..., selected].reshape(-1, len(selected))
            ratios = with_term / without
            if not np.isfinite(ratios).all() or np.any(ratios <= 0):
                raise ValueError("Nonfinite interaction MFE; no posterior draws were clipped or removed")
            table = pd.DataFrame(dict(interaction=label, interaction_index=k, mutation_1=names[i],
                                      mutation_2=names[j], carrier_genotype=[bits[n] for n in selected],
                                      carrier_count=counts[selected], background_fitness_mean=without.mean(axis=0),
                                      full_fitness_mean=with_term.mean(axis=0), point_estimator="posterior_mean",
                                      posterior_draw_count=len(ratios)))
            for key, value in ratio_summary(ratios, "mean").items():
                table[key] = value
            pair_tables.append(table)
            aggregate = ratios @ (counts[selected] / count)
            interaction_draws[..., k] = aggregate.reshape(shape)
            record.update({key: float(value) for key, value in ratio_summary(aggregate, "mean").items()})
        else:
            record.update({key: np.nan for key in ("estimated_mfe", "posterior_mean", "mfe_lower",
                                                   "mfe_upper", "probability_above_one")})
        records.append(record)
        if progress:
            print(f"  {k + 1}/{len(interactions)} {label}: {len(selected)} backgrounds, {count} carriers", flush=True)
    pairs = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame(columns=[
        "interaction", "interaction_index", "mutation_1", "mutation_2", "carrier_genotype", "carrier_count",
        "background_fitness_mean", "full_fitness_mean", "point_estimator", "posterior_draw_count",
        "estimated_mfe", "posterior_mean", "mfe_lower", "mfe_upper", "probability_above_one"])
    draws = dict(interaction_mfe=interaction_draws,
                 interaction_names=np.asarray([record["interaction"] for record in records]))
    return pairs, pd.DataFrame(records), draws


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--features-file", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        config_path = args.fit_dir / "run_config.json"
        archive = args.fit_dir / "posterior_samples.npz"
        config = json.loads(config_path.read_text())
        features = input_path(config, "features_file", args.features_file, root)
        tree_path = input_path(config, "tree_file", args.tree, root)
        names = main_feature_names(config)
        table = pd.read_csv(features, index_col=0)
        table.index = table.index.astype(str)
        tips = Tree(str(tree_path), format=1).get_leaf_names()
        if table.index.has_duplicates or len(set(tips)) != len(tips):
            raise ValueError("Duplicate feature-row or sampled-tip identifiers")
        observed = table.loc[tips, names].to_numpy(dtype=float)
        with np.load(archive, allow_pickle=False) as saved:
            samples = {key: saved[key] for key in ("log_fit_effects", "fit_effects", "base_fit",
                       "fitness_knots", "log_fitness_knots", "bs_coef") if key in saved}
        print(f"Interaction-only MFE: {args.fit_dir.name}; all chains and draws", flush=True)
        pairs, summary, draws = calculate_interaction_mfe(samples, config, observed, progress=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(args.output_dir / "pair_summaries.csv", index=False)
        summary.to_csv(args.output_dir / "interaction_summaries.csv", index=False)
        np.savez_compressed(args.output_dir / "interaction_draws.npz", **draws)
        shape = draws["interaction_mfe"].shape[:2]
        diagnostics_path = args.fit_dir / "diagnostics.json"
        metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), candidate=args.fit_dir.name,
                        model=config["model"], feature_set=config["feature_set"], fitted_configuration=config,
                        definition="F_full(genotype) / F_same_genotype_with_only_focal_interaction_coefficient_zero",
                        latent_off="latent_full - beta_ij; evaluated only where x_i=x_j=1",
                        main_effects_changed=False, other_interactions_changed=False,
                        coefficients_renormalized=False, branch_random_effects_included=False,
                        refitted=False, interaction_filter="none", point_estimator="posterior_mean",
                        credible_mass=.95, interval="equal_tailed; linear quantiles",
                        summary_aggregation="carrier_frequency_weighted_mean_ratio_within_each_draw_then_posterior_summary",
                        prediction_rows="distinct sampled genotypes carrying both mutations",
                        n_chains=shape[0], draws_per_chain=shape[1], n_total_draws=int(np.prod(shape)),
                        n_excluded_draws=0, n_sampled_tips=len(tips), n_unique_genotypes=len(np.unique(observed, axis=0)),
                        n_interactions=len(summary), n_interaction_background_pairs=len(pairs),
                        n_interactions_without_sampled_co_carriers=int((summary.n_carriers == 0).sum()),
                        main_feature_names=names, interaction_names=summary.interaction.tolist(),
                        sampler_diagnostics=json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {},
                        input_hashes=dict(posterior_sha256=file_digest(archive), config_sha256=file_digest(config_path),
                                          features_sha256=file_digest(features), tree_sha256=file_digest(tree_path)),
                        numpy_version=np.__version__, pandas_version=pd.__version__)
        (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {len(pairs)} interaction/background estimates and {len(summary)} interaction summaries to {args.output_dir}", flush=True)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
