"""Visualizes the bootstrap-OOB confidence intervals from
coverage_bootstrap_crossfit.py's coverage.csv (Reviewer 2, Major Comment 4), for one
replicate. Subjects are sorted by their true AUC so the true-AUC curve is monotonic;
the estimated AUC and its 95% bootstrap-OOB interval are plotted alongside it, colored
by whether the interval covers the true value.

Sorting by true AUC rather than by a covariate avoids the row-pairing ambiguity in
coverage.csv: each row pairs an independently-drawn healthy-arm covariate profile with
an independently-drawn diseased-arm profile (ground_truth_auc.py's row-paired
convention), so there is no single shared covariate to plot against for scenarios with
more than one covariate.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_coverage(coverage_csv, output_png, max_subjects=150, seed=0):
    df = pd.read_csv(coverage_csv)
    df = df.dropna(subset=["Covered"])
    if df.empty:
        print(f"No rows with a valid CI in {coverage_csv}")
        return

    if len(df) > max_subjects:
        df = df.sample(n=max_subjects, random_state=seed)
    df = df.sort_values("True AUC").reset_index(drop=True)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, df["True AUC"], color="black", linewidth=1.5, label="True AUC")

    covered = df["Covered"].astype(bool)
    for mask, color, label in [
        (covered, "tab:blue", "Estimated AUC, 95% CI (covered)"),
        (~covered, "tab:red", "Estimated AUC, 95% CI (not covered)"),
    ]:
        if not mask.any():
            continue
        ax.errorbar(
            x[mask], df.loc[mask, "Estimated AUC Mean"],
            yerr=[
                df.loc[mask, "Estimated AUC Mean"] - df.loc[mask, "CI Lower"],
                df.loc[mask, "CI Upper"] - df.loc[mask, "Estimated AUC Mean"],
            ],
            fmt="o", markersize=3, color=color, ecolor=color, alpha=0.6, label=label,
        )

    scenario = df["Scenario"].iloc[0]
    ax.set_xlabel("Subjects (sorted by true AUC)")
    ax.set_ylabel("AUC")
    ax.set_title(f"Bootstrap-OOB confidence intervals -- {scenario}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot bootstrap-OOB confidence intervals from one replicate's coverage.csv")
    parser.add_argument("coverage_csv", help="Path to a single replicate's coverage.csv")
    parser.add_argument("--output-png", default=None, help="Output PNG path (default: alongside coverage_csv)")
    parser.add_argument("--max-subjects", type=int, default=150, help="Subsample size for readability")
    args = parser.parse_args()
    output_png = args.output_png or os.path.join(os.path.dirname(args.coverage_csv), "coverage_band_plot.png")
    plot_coverage(args.coverage_csv, output_png, max_subjects=args.max_subjects)
