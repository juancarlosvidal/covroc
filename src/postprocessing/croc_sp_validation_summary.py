"""Diffs R/croc_sp_validation.R's output against src/baselines/croc_sp_validation.py's,
per scenario/sample-size, to confirm cROC_sp (the Python port used at scale by
linear_reg_data_simulation.py, Reviewer 1 Major Concern 2) agrees numerically with the
real ROCnReg::cROC.sp. Both are closed-form OLS estimators of the same model, so a
faithful port should match to numerical precision (no MCMC/approximation noise) -- large
diffs here would flag a real discrepancy in the port, not sampling variation.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

_SCENARIO_N_RE = re.compile(r"^scenario_(\d+)_(\d+)_\d+_data$")


def _max_mean_abs_diff(r_file, py_file, column):
    r_df = pd.read_csv(r_file)
    py_df = pd.read_csv(py_file)
    diff = (r_df[column] - py_df[column]).abs()
    return diff.max(), diff.mean(), len(diff)


def main(input_dir, output_csv):
    auc_r_files = sorted(glob.glob(os.path.join(input_dir, "*_auc_R.csv")))
    if not auc_r_files:
        print(f"No *_auc_R.csv files found under {input_dir}")
        return

    rows = []
    for auc_r_file in auc_r_files:
        name = os.path.basename(auc_r_file)[: -len("_auc_R.csv")]
        match = _SCENARIO_N_RE.match(name)
        if not match:
            continue
        scenario_num, sample_size = int(match.group(1)), int(match.group(2))

        auc_py_file = os.path.join(input_dir, f"{name}_auc_python.csv")
        roc_r_file = os.path.join(input_dir, f"{name}_roc_row1_R.csv")
        roc_py_file = os.path.join(input_dir, f"{name}_roc_row1_python.csv")
        if not all(os.path.exists(f) for f in (auc_py_file, roc_r_file, roc_py_file)):
            print(f"Skipping {name}: missing R and/or Python output")
            continue

        auc_max, auc_mean, n_subjects = _max_mean_abs_diff(auc_r_file, auc_py_file, "AUC")
        roc_max, roc_mean, _ = _max_mean_abs_diff(roc_r_file, roc_py_file, "ROC")

        rows.append({
            "Scenario Num": scenario_num,
            "Sample Size": sample_size,
            "N Subjects": n_subjects,
            "AUC Max Abs Diff": auc_max,
            "AUC Mean Abs Diff": auc_mean,
            "ROC Row1 Max Abs Diff": roc_max,
            "ROC Row1 Mean Abs Diff": roc_mean,
        })

    df = pd.DataFrame(rows).sort_values(["Scenario Num", "Sample Size"])
    df.to_csv(output_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize R-vs-Python cROC.sp validation diffs across scenarios")
    parser.add_argument("--input-dir", default="output/croc_sp_validation", help="Directory with R/croc_sp_validation.R's and src/baselines/croc_sp_validation.py's output CSVs")
    parser.add_argument("--output-csv", default="output/croc_sp_validation/croc_sp_validation_summary.csv", help="Path to write the combined diff summary table")
    args = parser.parse_args()
    main(args.input_dir, args.output_csv)
