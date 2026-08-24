"""Combines R/aroc_bnp_timing.R's per-replicate output/aroc_bnp_timing/*_timing.csv
files (one per scenario/sample-size, single representative replicate) into one sorted
table, for the response letter's ROCnReg::AROC.bnp computation-time comparison
(Reviewer 1, Major Concern 2 -- why the Bayesian nonparametric estimator wasn't
reimplemented/run at the same 100-replicate scale as the other methods).
"""
import argparse
import glob
import os

import pandas as pd

_SCENARIO_N_RE = r"^scenario_(\d+)_(\d+)_\d+_data$"


def main(input_dir, output_csv):
    files = sorted(glob.glob(os.path.join(input_dir, "*_timing.csv")))
    if not files:
        print(f"No *_timing.csv files found under {input_dir}")
        return

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    extracted = df["Scenario"].str.extract(_SCENARIO_N_RE)
    df["Scenario Num"] = extracted[0].astype(int)
    df["Sample Size"] = extracted[1].astype(int)
    df = df.sort_values(["Scenario Num", "Sample Size"])[
        ["Scenario Num", "Sample Size", "Method", "Fit Seconds Total"]
    ]

    df.to_csv(output_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine per-replicate AROC.bnp timing CSVs into one summary table")
    parser.add_argument("--input-dir", default="output/aroc_bnp_timing", help="Directory containing R/aroc_bnp_timing.R's *_timing.csv files")
    parser.add_argument("--output-csv", default="output/aroc_bnp_timing/aroc_bnp_timing_summary.csv", help="Path to write the combined summary table")
    args = parser.parse_args()
    main(args.input_dir, args.output_csv)
