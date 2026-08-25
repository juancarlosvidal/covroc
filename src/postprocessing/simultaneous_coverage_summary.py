"""Empirical simultaneous (Bonferroni-corrected) coverage, on top of the pointwise
coverage.csv files src/simulation/coverage_bootstrap_crossfit.py already writes (Reviewer
2, Major Comment 4: "the simulations should report pointwise and simultaneous coverage").

Reuses each subject's already-computed pointwise 95% CI (CI Lower/CI Upper in
coverage.csv) rather than re-running the bootstrap-crossfit-OOB engine: the raw per-subject
bootstrap draws behind that CI are never persisted (bootstrap_crossfit_oob.py's
_aggregate_oob collapses them to mean/CI before returning), and re-deriving them would mean
re-running the multi-hour-per-replicate computation. Instead, approximating each subject's
standard error as SE_i = (CI Upper_i - CI Lower_i) / (2 * 1.96) and widening with a
Bonferroni-corrected critical value gives a valid (if conservative -- Bonferroni holds
regardless of the dependence structure between subjects, unlike a naive independence
assumption) simultaneous band, computed purely from data already on disk.

Checked jointly over a small, reproducible per-replicate subsample of subjects (--n-points,
default 20) rather than every subject in the replicate (which can number in the thousands):
Bonferroni's critical value only grows like sqrt(log N), so this isn't strictly necessary to
avoid absurdly wide bands, but keeps the "simultaneous over N points" claim interpretable and
matches the reviewer's request for a joint check across a curve/profile, not literally every
simulated subject.
"""
import argparse
import os
import re
import zlib

import numpy as np
import pandas as pd
from scipy.stats import norm


def _seed_from_name(name):
    return zlib.crc32(name.encode())


def _scenario_id(scenario):
    match = re.match(r"^(scenario_\d+)_", scenario)
    return match.group(1) if match else scenario


def _simultaneous_covered(df, n_points, rng):
    """One replicate's coverage.csv (already filtered to rows with a valid CI):
    picks n_points subjects deterministically, widens their CI with a Bonferroni critical
    value for n_points simultaneous comparisons, and returns whether all of them
    contained the true AUC at once, plus the actual number of points checked (may be
    less than n_points if the replicate has fewer valid rows) and the widening factor
    relative to the pointwise z=1.96 critical value.
    """
    n_selected = min(n_points, len(df))
    if n_selected == 0:
        return None
    idx = rng.choice(len(df), size=n_selected, replace=False)
    sample = df.iloc[idx]

    se = (sample["CI Upper"] - sample["CI Lower"]) / (2 * 1.96)
    z_bonf = norm.ppf(1 - 0.05 / (2 * n_selected))
    lower = sample["Estimated AUC Mean"] - z_bonf * se
    upper = sample["Estimated AUC Mean"] + z_bonf * se

    all_covered = bool(((sample["True AUC"] >= lower) & (sample["True AUC"] <= upper)).all())
    return all_covered, n_selected, z_bonf / 1.96


def main(root_dir, output_csv, n_points):
    rows = []
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        coverage_file = os.path.join(folder_path, "coverage.csv")
        if not (os.path.isdir(folder_path) and os.path.exists(coverage_file)):
            continue

        df = pd.read_csv(coverage_file)
        df = df.dropna(subset=["Covered"])
        if df.empty:
            continue

        result = _simultaneous_covered(df, n_points, np.random.default_rng(_seed_from_name(folder)))
        if result is None:
            continue
        all_covered, n_selected, widening_factor = result

        rows.append({
            "Scenario ID": _scenario_id(df["Scenario"].iloc[0]),
            "Method": df["Method"].iloc[0],
            "Replicate": folder,
            "N Points Checked": n_selected,
            "Widening Factor vs Pointwise": widening_factor,
            "Simultaneously Covered": all_covered,
        })

    if not rows:
        print(f"No coverage.csv files found under {root_dir}; wrote nothing.")
        return

    detail = pd.DataFrame(rows)
    summary = detail.groupby(["Scenario ID", "Method"]).agg(
        **{
            "Simultaneous Coverage Rate": ("Simultaneously Covered", "mean"),
            "Mean Widening Factor": ("Widening Factor vs Pointwise", "mean"),
            "Mean N Points Checked": ("N Points Checked", "mean"),
            "N Replicates": ("Simultaneously Covered", "count"),
        }
    ).reset_index()

    summary.to_csv(output_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bonferroni-corrected simultaneous coverage from existing coverage.csv files")
    parser.add_argument("--root-dir", default="output/combined", help="Directory containing one subfolder per replicate, each with a coverage.csv")
    parser.add_argument("--output-csv", default="output/combined/simultaneous_coverage_summary.csv", help="Path to write the per-scenario simultaneous coverage summary table")
    parser.add_argument("--n-points", type=int, default=20, help="Number of subjects checked jointly per replicate")
    args = parser.parse_args()
    main(args.root_dir, args.output_csv, args.n_points)
