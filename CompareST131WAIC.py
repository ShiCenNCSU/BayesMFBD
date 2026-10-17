"""Compare ST131 candidates using saved pointwise log likelihoods.

Usage: python CompareST131WAIC.py ST131_HMC_candidates

Add ESS columns without reopening posterior archives or recomputing WAIC:
python CompareST131WAIC.py RESULTS_DIR --existing-comparison WAIC_CSV

min_ess is the minimum saved NumPyro n_eff across HMC_summary_*.csv rows,
including transformed/derived coefficients and branch effects. Constant
parameters are excluded. min_ess_parameter identifies the summary variable
and its element; exact ties are separated by semicolons. Missing summaries
leave the ESS columns blank. Undefined ESS for nonconstant rows is reported.

Reads each candidate's posterior_samples.npz['log_lik'], with shape
(chain, draw, likelihood_term). Warmup is already excluded by the runner.
The likelihood terms are edge segments, removal-sampling events,
contemporaneous-sampling events, then birth events, in the saved order.
Use the same tree, time grid, and data settings for every candidate.

The sample-variance penalty (ddof=1) matches the existing Hazel comparison
code. This is WAIC for the stored tree-event factorization; it does not
define a predictive unit such as an independently sampled tree or tip.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.special import logsumexp


def calculate_waic(log_lik):
    """Return deviance-scale WAIC; smaller values indicate better scores."""
    values = np.asarray(log_lik, dtype=np.float64)
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError("log_lik must have nonempty (chain, draw, likelihood_term) dimensions")
    n_chains, n_draws, n_terms = values.shape
    pooled = values.reshape(n_chains * n_draws, n_terms)
    if len(pooled) < 2:
        raise ValueError("WAIC requires at least two posterior draws")
    if not np.all(np.isfinite(pooled)):
        raise ValueError("log_lik contains nonfinite values")
    lppd_terms = logsumexp(pooled, axis=0) - np.log(len(pooled))
    variance_terms = np.var(pooled, axis=0, ddof=1)
    lppd = float(np.sum(lppd_terms))
    p_waic = float(np.sum(variance_terms))
    elpd_waic = lppd - p_waic
    return dict(waic=-2. * elpd_waic, elpd_waic=elpd_waic, lppd=lppd, p_waic=p_waic,
                n_chains=n_chains, draws_per_chain=n_draws, n_terms=n_terms,
                high_variance_terms=int(np.sum(variance_terms > 0.4)))


def minimum_ess(candidate_dir, n_chains=None, draws_per_chain=None):
    """Read the smallest saved n_eff without loading posterior draws."""
    candidate_dir = Path(candidate_dir)
    paths = sorted(candidate_dir.glob("HMC_summary_*.csv"))
    report = dict(min_ess="", min_ess_parameter="", ess_summary_files=[],
                  ess_fixed_parameters_skipped=0, ess_undefined_parameters=[],
                  ess_finite_parameters=0)
    if not paths:
        return report
    diagnostic_path = candidate_dir / "diagnostics.json"
    if diagnostic_path.exists():
        diagnostics = json.loads(diagnostic_path.read_text())
        if ((n_chains is not None and diagnostics["chains"] != n_chains) or
                (draws_per_chain is not None and diagnostics["draws_per_chain"] != draws_per_chain)):
            raise ValueError(f"{candidate_dir}: diagnostics chain/draw counts differ from the WAIC comparison")
    smallest = math.inf
    parameters = []
    for path in paths:
        family = path.stem.removeprefix("HMC_summary_")
        report["ess_summary_files"].append(str(path))
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "n_eff" not in reader.fieldnames:
                raise ValueError(f"{path}: n_eff column is missing")
            index_column = reader.fieldnames[0]
            for row in reader:
                label = row[index_column]
                parameter = label if label == family or label.startswith(family + "_") else f"{family}[{label}]"
                std = float(row.get("std") or "nan")
                if std == 0.:
                    report["ess_fixed_parameters_skipped"] += 1
                    continue
                value = float(row["n_eff"] or "nan")
                if not math.isfinite(value) or value <= 0.:
                    report["ess_undefined_parameters"].append(parameter)
                    continue
                report["ess_finite_parameters"] += 1
                if value < smallest:
                    smallest, parameters = value, [parameter]
                elif value == smallest:
                    parameters.append(parameter)
    if parameters:
        report.update(min_ess=smallest, min_ess_parameter="; ".join(parameters))
    return report


def add_ess(rows, results_dir, require_diagnostics=False):
    reports = {}
    for row in rows:
        candidate = row["candidate"]
        folders = (results_dir / candidate, results_dir / f"{candidate}_summaries")
        folder = next((path for path in folders if any(path.glob("HMC_summary_*.csv"))), folders[0])
        if (require_diagnostics and any(folder.glob("HMC_summary_*.csv")) and
                not (folder / "diagnostics.json").exists()):
            raise ValueError(f"{folder}: diagnostics.json is required to verify the ESS summary chain/draw counts")
        report = minimum_ess(folder, int(row["n_chains"]), int(row["draws_per_chain"]))
        row.update(min_ess=report["min_ess"], min_ess_parameter=report["min_ess_parameter"])
        reports[candidate] = report
    return reports


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, nargs="?",
                        help="Directory containing candidate fits or CANDIDATE_summaries folders")
    parser.add_argument("--output", type=Path, help="Comparison CSV (default: RESULTS_DIR/waic_comparison.csv)")
    parser.add_argument("--existing-comparison", type=Path,
                        help="Add ESS to this saved comparison without loading posterior archives; updates it unless --output is supplied")
    args = parser.parse_args(argv)
    try:
        results_dir = args.results_dir or (args.existing_comparison.parent if args.existing_comparison
                                          else root / "ST131_HMC_candidates")
        if args.existing_comparison:
            with args.existing_comparison.open(newline="") as handle:
                reader = csv.DictReader(handle)
                columns = list(reader.fieldnames or [])
                rows = list(reader)
            required = {"candidate", "waic", "n_chains", "draws_per_chain"}
            if not rows or not required.issubset(columns):
                raise ValueError(f"{args.existing_comparison}: a nonempty comparison with {sorted(required)} is required")
            if len({row["candidate"] for row in rows}) != len(rows):
                raise ValueError("The existing comparison contains duplicate candidates")
        else:
            archives = sorted(results_dir.glob("*/posterior_samples.npz"))
            if not archives:
                raise ValueError(f"No candidate posterior archives found under {results_dir}")
            rows = []
            for archive in archives:
                with np.load(archive) as posterior:
                    if "log_lik" not in posterior:
                        raise ValueError(f"Pointwise log likelihood 'log_lik' missing from {archive}")
                    rows.append(dict(candidate=archive.parent.name, **calculate_waic(posterior["log_lik"])))
            if len({row["n_terms"] for row in rows}) != 1:
                raise ValueError("Candidates have different likelihood-term counts; compare runs on the same data and time grid")
            rows.sort(key=lambda row: row["waic"])
            for row in rows:
                row["delta_waic"] = row["waic"] - rows[0]["waic"]
            columns = ["candidate", "waic", "delta_waic", "elpd_waic", "lppd", "p_waic",
                       "n_chains", "draws_per_chain", "n_terms", "high_variance_terms"]
        reports = add_ess(rows, results_dir, require_diagnostics=bool(args.existing_comparison))
        columns.extend(name for name in ("min_ess", "min_ess_parameter") if name not in columns)
        output = args.output or args.existing_comparison or results_dir / "waic_comparison.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=output.parent, prefix=output.name + ".", newline="", delete=False) as handle:
            temporary_path = Path(handle.name)
            try:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        os.replace(temporary_path, output)
        metadata = dict(ess_definition="minimum saved NumPyro n_eff over all nonconstant HMC_summary rows, including transformed/derived and branch parameters",
                        missing_ess="blank CSV cells; no older or unrelated fit is substituted",
                        results_dir=str(results_dir), candidates=reports)
        output.with_suffix(".ess_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        action = "Annotated" if args.existing_comparison else "Compared"
        print(f"{action} {len(rows)} completed candidate(s); smaller WAIC is better.")
        print(f"{'Candidate':38} {'WAIC':>14} {'Delta WAIC':>14} {'Min ESS':>12}  Parameter")
        for row in rows:
            ess = f"{row['min_ess']:.3f}" if row["min_ess"] != "" else "unavailable"
            delta = float(row.get("delta_waic", 0.))
            print(f"{row['candidate']:38} {float(row['waic']):14.3f} {delta:14.3f} {ess:>12}  {row['min_ess_parameter']}")
        missing = [name for name, report in reports.items() if report["min_ess"] == ""]
        if missing:
            print("ESS summaries unavailable for: " + ", ".join(missing))
        undefined = {name: report["ess_undefined_parameters"] for name, report in reports.items()
                     if report["ess_undefined_parameters"]}
        if undefined:
            print("Undefined ESS for nonconstant parameters is recorded in " + str(output.with_suffix(".ess_metadata.json")))
        flagged = [row["candidate"] for row in rows if int(row.get("high_variance_terms", 0))]
        if flagged:
            print("Log-likelihood variance exceeds 0.4 for some terms in: " + ", ".join(flagged)
                  + ". WAIC may be unreliable; inspect posterior diagnostics.")
        print(f"Saved {output}")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
