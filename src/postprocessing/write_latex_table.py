# Generic LaTeX table writers used to build the simulation-scenario MSE tables
# (statistics_summary.py output) reported in the paper's Supplementary Material.
import argparse
import os
import re

import pandas as pd

class LatexTableWriter:
    """
    Class to create and append rows to a LaTeX table (.tex) with an arbitrary number of columns.
    """
    def __init__(self, filepath, column_headers, caption=None, label=None):
        if not column_headers or not isinstance(column_headers, (list, tuple)):
            raise ValueError("column_headers must be a non-empty list or tuple of header names.")
        self.filepath = filepath
        self.headers = column_headers
        self.caption = caption
        self.label = label
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        # Write header if new file
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', encoding='utf-8') as f:
                f.write("\\begin{table}[ht]\n")
                f.write("  \\centering\n")
                # Column format
                col_fmt = ' | '.join(['l'] * len(self.headers))
                f.write(f"  \\begin{{tabular}}{{| {col_fmt} |}}\n")
                f.write("    \\hline\n")
                f.write("    " + " & ".join(self.headers) + " \\\\ \n")
                f.write("    \\hline\n")

    def append_row(self, row):
        """Append one row matching the number of headers."""
        if len(row) != len(self.headers):
            raise ValueError(f"Row has {len(row)} entries but expected {len(self.headers)}.")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("    " + " & ".join(map(str, row)) + " \\\\ \n")
            f.write("    \\hline\n")

    def finalize(self):
        """Close the tabular and table environments."""
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("  \\end{tabular}\n")
            if self.caption:
                f.write(f"  \\caption{{{self.caption}}}\n")
            if self.label:
                f.write(f"  \\label{{{self.label}}}\n")
            f.write("\\end{table}\n")


class LatexTableDatWriter:
    """
    Class to create and append LaTeX table code into a .dat file with arbitrary columns.
    """
    def __init__(self, filepath, column_headers, caption=None, label=None):
        if not filepath.endswith('.dat'):
            raise ValueError("filepath must have a .dat extension.")
        if not column_headers or not isinstance(column_headers, (list, tuple)):
            raise ValueError("column_headers must be a non-empty list or tuple of header names.")
        self.filepath = filepath
        self.headers = column_headers
        self.caption = caption
        self.label = label
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        # Write header if new
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', encoding='utf-8') as f:
                f.write("\\begin{table}[ht]\n")
                f.write("  \\centering\n")
                col_fmt = ' | '.join(['l'] * len(self.headers))
                f.write(f"  \\begin{{tabular}}{{| {col_fmt} |}}\n")
                f.write("    \\hline\n")
                f.write("    " + " & ".join(self.headers) + " \\\\ \n")
                f.write("    \\hline\n")

    def append_row(self, row):
        if len(row) != len(self.headers):
            raise ValueError(f"Row has {len(row)} entries but expected {len(self.headers)}.")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("    " + " & ".join(map(str, row)) + " \\\\ \n")
            f.write("    \\hline\n")

    def finalize(self):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("  \\end{tabular}\n")
            if self.caption:
                f.write(f"  \\caption{{{self.caption}}}\n")
            if self.label:
                f.write(f"  \\label{{{self.label}}}\n")
            f.write("\\end{table}\n")


# statistics_summary.csv (statistics_summary.py's roc_mse_values.csv aggregation) has one
# row per *replicate folder*, not one row per (scenario, method) -- unlike timing_summary.csv/
# mean_std_mse_summary.csv/coverage_summary.csv, roc_mse_values.csv carries no "Method"
# column of its own, so statistics_summary.py can't group by it directly. The method and
# scenario are only recoverable from "Folder Name", under the method-prefixed-subfolder
# naming convention statistics_summary.py's own docstring already assumes for cross-method
# comparison (e.g. "fnn_scenario_1_5000_1", "rf_scenario_1_5000_1", ...).
_FOLDER_RE = re.compile(r"^(?P<method>[A-Za-z0-9]+)_scenario_(?P<scenario>\d+)_(?P<n>\d+)_(?P<rep>\d+)$")


def _parse_folder_name(name):
    match = _FOLDER_RE.match(name)
    if not match:
        return None
    return match.group("method"), int(match.group("scenario")), int(match.group("n"))


def build_mse_table(stats_csv):
    """Aggregates statistics_summary.csv's per-replicate rows into one row per
    (scenario, sample size, method): the mean (and its across-replicate SD) of the
    per-replicate mean ROC-curve MSE -- the actual Table 1-style FNN-vs-RF-vs-baselines
    comparison, as opposed to the un-aggregated per-replicate rows statistics_summary.csv
    itself contains.
    """
    df = pd.read_csv(stats_csv)
    parsed = df["Folder Name"].apply(_parse_folder_name)
    skipped = parsed.isna().sum()
    if skipped:
        print(f"Skipping {skipped} row(s) whose Folder Name doesn't match "
              f"'<method>_scenario_<N>_<sample size>_<replicate>' (run statistics_summary.py "
              f"on a --root-dir with method-prefixed subfolders to avoid this).")
    df = df[parsed.notna()].copy()
    df[["Method", "Scenario", "Sample Size"]] = pd.DataFrame(parsed[parsed.notna()].tolist(), index=df.index)

    summary = df.groupby(["Scenario", "Sample Size", "Method"])["Mean"].agg(
        **{"Mean ROC MSE": "mean", "SD Across Runs": "std", "N Replicates": "count"}
    ).reset_index()
    return summary.sort_values(["Scenario", "Sample Size", "Method"])


def write_tables(summary, tex_path, dat_path, caption, label):
    headers = ["Scenario", "Sample Size", "Method", "Mean ROC MSE", "SD Across Runs", "N Replicates"]
    for path in (tex_path, dat_path):
        if os.path.exists(path):
            os.remove(path)  # writers only append; start each run from a clean file
    tex_writer = LatexTableWriter(tex_path, headers, caption=caption, label=label)
    dat_writer = LatexTableDatWriter(dat_path, headers, caption=caption, label=label + "_dat")

    for _, row in summary.iterrows():
        formatted = [
            f"scenario\\_{int(row['Scenario'])}",
            int(row["Sample Size"]),
            row["Method"],
            f"{row['Mean ROC MSE']:.4f}",
            f"{row['SD Across Runs']:.4f}" if pd.notna(row["SD Across Runs"]) else "--",
            int(row["N Replicates"]),
        ]
        tex_writer.append_row(formatted)
        dat_writer.append_row(formatted)

    tex_writer.finalize()
    dat_writer.finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build the per-scenario, per-method ROC-curve MSE comparison table "
                     "(Table 1-style) from statistics_summary.py's statistics_summary.csv."
    )
    parser.add_argument("--stats-csv", default="statistics_summary.csv",
                         help="statistics_summary.csv produced by statistics_summary.py")
    parser.add_argument("--tex-output", default="summary_table_mse.tex")
    parser.add_argument("--dat-output", default="summary_table_mse.dat")
    parser.add_argument("--caption", default="ROC-curve MSE by scenario, sample size, and method")
    parser.add_argument("--label", default="tab:mse_summary")
    args = parser.parse_args()

    summary = build_mse_table(args.stats_csv)
    write_tables(summary, args.tex_output, args.dat_output, args.caption, args.label)
    print(f"Wrote tables: {args.tex_output}, {args.dat_output}")
