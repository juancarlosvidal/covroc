"""Reproduces the Supplementary Material's Figures 10-18 (MSE boxplots per scenario,
split by healthy/diseased group), but with every method compared side by side instead
of showing only the FNN (Reviewer 1, Minor Concern 3: "Figures 10-18 only show the
performance of the ROC-NN. Inclusion of the other examined methods' performances would
allow for direct visual comparison with the proposed method.").

Reads every replicate folder's mean_std_mse.csv (written by
mlp_reg_data_simulation_multi.py / rf_reg_data_simulation_multi_2.py, one row per group)
under --root-dir, expects the same method-prefixed-subfolder convention as
statistics_summary.py (e.g. fnn_scenario_1_5000_1/, rf_scenario_1_5000_1/, ...), and
draws one boxplot per (scenario, sample size): Mean-Function MSE and Std-Function MSE
on the x-axis, one row per group (healthy/diseased), colored by method.
"""
import argparse
import os
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

_SCENARIO_N_RE = re.compile(r"^(scenario_\d+)_(\d+)_")


def _scenario_and_n(scenario):
    match = _SCENARIO_N_RE.match(scenario)
    if match:
        return match.group(1), int(match.group(2))
    return scenario, None


def load_mean_std_mse(root_dir):
    rows = []
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        csv_file = os.path.join(folder_path, "mean_std_mse.csv")
        if os.path.isdir(folder_path) and os.path.exists(csv_file):
            rows.append(pd.read_csv(csv_file))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df[["Scenario ID", "Sample Size"]] = df["Scenario"].apply(lambda s: pd.Series(_scenario_and_n(s)))
    return df


def plot_scenario(df, scenario_id, n, output_dir):
    subset = df[(df["Scenario ID"] == scenario_id) & (df["Sample Size"] == n)]
    if subset.empty:
        return
    long = subset.melt(
        id_vars=["Group", "Method"],
        value_vars=["Mean-Function MSE", "Std-Function MSE"],
        var_name="Metric", value_name="Value",
    )
    g = sns.catplot(
        data=long, kind="box",
        x="Metric", y="Value", hue="Method", row="Group",
        height=4, aspect=1.6, legend=True,
    )
    g.figure.suptitle(f"{scenario_id} -- {n} individuals", y=1.02)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{scenario_id}_{n}_mean_std_mse_boxplot.png")
    g.savefig(out_path, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Wrote {out_path}")


def main(root_dir, output_dir):
    df = load_mean_std_mse(root_dir)
    if df.empty:
        print(f"No mean_std_mse.csv files found under {root_dir}")
        return
    for scenario_id, n in sorted(df[["Scenario ID", "Sample Size"]].drop_duplicates().itertuples(index=False)):
        plot_scenario(df, scenario_id, n, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-scenario Mean-/Std-Function MSE boxplots, compared across methods")
    parser.add_argument("--root-dir", default="output", help="Directory containing one method-prefixed subfolder per replicate, each with a mean_std_mse.csv")
    parser.add_argument("--output-dir", default="output/mean_std_mse_boxplots", help="Directory to write the boxplot PNGs")
    args = parser.parse_args()
    main(args.root_dir, args.output_dir)
