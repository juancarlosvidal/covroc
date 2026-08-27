"""Naive (pooled, no covariate adjustment) ROC baseline for the NHANES case study
(Reviewer 1, Major Concern 3: "the naive non-covariate adjusted ROC method ... would
better illustrate the differences in the estimated aROC/AUC measures"). Unlike
src/simulation/naive_roc_baseline.py (which operates on the synthetic scenario CSVs and
scores against a known ground truth), this estimates a single pooled ROC(p) curve and
AUC per sex and mortality horizon from the real TAC2 values in data/df_f.csv/df_m.csv --
the same files and target column (TAC2) used by notebooks/nhanes_hetero_residuos.ipynb
for the FNN bootstrap-OOB confidence bands, so the naive AUC is directly comparable to
those covariate-adjusted results.

No covariate adjustment means no age-varying curve to plot alongside the FNN/Random
Forest/R-model age-scatter panels in Figures 2-4 -- the naive estimate is a single
number per sex/horizon, reported here as a summary table rather than a scatter plot.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.integrate import simpson

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation'))
from ground_truth_auc import roc_from_samples

P_GRID = np.linspace(0.001, 0.999, 100)
GENDER_FILES = {"female": "df_f.csv", "male": "df_m.csv"}
HORIZONS = ["tres", "cinco", "ocho"]
TARGET_COL = "TAC2"


def main(input_dir, output_csv):
    rows = []
    for gender, filename in GENDER_FILES.items():
        df = pd.read_csv(os.path.join(input_dir, filename))
        for horizon in HORIZONS:
            sub = df.dropna(subset=[TARGET_COL, horizon])
            y0 = sub.loc[sub[horizon] == 0, TARGET_COL].to_numpy()
            y1 = sub.loc[sub[horizon] == 1, TARGET_COL].to_numpy()
            if len(y0) == 0 or len(y1) == 0:
                print(f"Skipping {gender}/{horizon}: one group is empty")
                continue

            # TAC is protective (lower TAC associates with higher mortality), so the raw
            # AUC from roc_from_samples(y0, y1, ...) -- which assumes higher marker values
            # indicate the "diseased" group -- comes out below 0.5. src/real_data/mlp_reg.py
            # (line ~321, `auc = 1 - auc`) and croc_linear_baseline.py's real-data usage
            # (`1 - cROC_model['AUC']`) both handle this the same way: report 1 - the raw
            # AUC as the discriminative measure. Matched here for consistency (Reviewer 2,
            # Major Comment 2: "state how the direction was handled and use it consistently").
            roc_curve = roc_from_samples(y0, y1, P_GRID)
            auc = 1 - simpson(roc_curve, P_GRID)

            rows.append({
                "Gender": gender,
                "Horizon": horizon,
                "N Healthy": len(y0),
                "N Diseased": len(y1),
                "Naive Pooled AUC": auc,
            })
            print(f"{gender}/{horizon}: N={len(y0)}+{len(y1)}, naive pooled AUC={auc:.4f}")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive (pooled, no covariate adjustment) ROC/AUC baseline for the NHANES case study")
    parser.add_argument("-i", "--input-dir", default="./data", help="Directory containing df_f.csv/df_m.csv")
    parser.add_argument("-o", "--output-csv", default="./output/real_data/naive_roc_summary.csv", help="Path to write the per-sex, per-horizon naive AUC summary")
    args = parser.parse_args()
    main(args.input_dir, args.output_csv)
