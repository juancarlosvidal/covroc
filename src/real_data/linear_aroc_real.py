"""Linear-regression aROC baseline on the NHANES cohort.

Reviewer 1's Major Concern 3 asks for "a linear model approach for aROC" in the
NHANES case study, alongside the naive pooled ROC. The semiparametric additive
model already shown in Figures 2-4 is not a linear model, so this script adds the
linear one, fit with the same location-scale estimator used for the linear baseline
of the simulation study (src/baselines/croc_linear_baseline.py, a port of
ROCnReg::cROC.sp).

Two OLS models per arm give mu_d(x) and a constant sigma_d; the covariate-specific
AUC follows the paper's Gaussian location-scale formula. Covariates are age and BMI,
matching combination 6 of src/real_data/mlp_reg.py, and the analysis is stratified
by sex (separate input files), again matching the case study.

DIRECTION. TAC is protective: decedents have lower TAC. The location-scale formulas
assume higher marker values indicate the diseased arm, so every AUC reported here is
1 - AUC, the same convention applied by src/real_data/mlp_reg.py (`auc = 1 - auc`)
and src/real_data/naive_roc_baseline.py (`1 - simpson(...)`).

The marker column is TAC2, which is what the manuscript calls TAC -- verified against
the manuscript's Table 1: TAC2 reproduces its mean, SD, median and IQR for both sexes
to the cent, whereas the CSV's own TAC column does not.

Run from the repository root:

    python src/real_data/linear_aroc_real.py                 # point estimates
    python src/real_data/linear_aroc_real.py -B 200          # with bootstrap CIs
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines"))
from croc_linear_baseline import cROC_sp  # noqa: E402

GENDER_FILES = {"female": "df_f.csv", "male": "df_m.csv"}
HORIZONS = {"tres": "3-year", "cinco": "5-year", "ocho": "8-year"}
MARKER = "TAC2"
AGE_RAW = "RIDAGEYR.x"
AGE = "AGE"          # patsy cannot parse a dot in a term, so the column is renamed
BMI = "BMI"


def age_grid(df, n_grid, lo_q, hi_q):
    """Grid of covariate profiles: age over an inner quantile range, BMI at median."""
    ages = np.linspace(df[AGE].quantile(lo_q), df[AGE].quantile(hi_q), n_grid)
    return pd.DataFrame({AGE: ages, BMI: np.full(n_grid, df[BMI].median())})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input_dir", default="./data")
    p.add_argument("-o", "--output_csv",
                   default="output/knn_bands_real/linear_aroc_real.csv")
    p.add_argument("-gp", "--grid_points", type=int, default=21)
    p.add_argument("--lo_q", type=float, default=0.025)
    p.add_argument("--hi_q", type=float, default=0.975)
    p.add_argument("-B", "--n_boot", type=int, default=0,
                   help="bootstrap replicates for the CIs (0 = point estimates only)")
    p.add_argument("--transform", choices=["log", "none"], default="log",
                   help="marker transform before the Gaussian location-scale fit. "
                        "The AUC is invariant under a monotone transform of the "
                        "marker, so this does not change the estimand, but raw TAC "
                        "is heavily right-skewed (max 45,105 against a median of 696) "
                        "and the Gaussian assumption is untenable on that scale. "
                        "'log' matches src/real_data/local_knn_bands_real.py.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    rows = []
    for sex, fname in GENDER_FILES.items():
        df_all = pd.read_csv(os.path.join(args.input_dir, fname))
        df_all = df_all.rename(columns={AGE_RAW: AGE})
        for hcol, hlabel in HORIZONS.items():
            df = df_all.dropna(subset=[MARKER, AGE, BMI, hcol]).reset_index(drop=True)
            df[hcol] = df[hcol].astype(int)
            if args.transform == "log":
                df = df[df[MARKER] > 0].reset_index(drop=True)
                df[MARKER] = np.log(df[MARKER])
            n0 = int((df[hcol] == 0).sum())
            n1 = int((df[hcol] == 1).sum())
            if n0 == 0 or n1 == 0:
                print(f"skipping {sex}/{hlabel}: one arm is empty")
                continue

            grid = age_grid(df, args.grid_points, args.lo_q, args.hi_q)
            formula = f"{MARKER} ~ {AGE} + {BMI}"
            res = cROC_sp(formula_h=formula, formula_d=formula,
                          group=hcol, tag_h=0, data=df, newdata=grid,
                          B=args.n_boot, ci_level=0.95)

            # TAC is protective -- see the module docstring.
            auc = 1.0 - res["AUC"][:, 0]
            lo = 1.0 - res["AUC"][:, 2]
            hi = 1.0 - res["AUC"][:, 1]

            for j in range(len(grid)):
                rows.append({
                    "Sex": sex, "Horizon": hlabel,
                    "Age": grid[AGE].iloc[j], "BMI": grid[BMI].iloc[j],
                    "AUC Linear": auc[j],
                    "CI95 Lower": lo[j], "CI95 Upper": hi[j],
                    "N Controls": n0, "N Cases": n1,
                })
            print(f"{sex}/{hlabel}: N={n0}+{n1}  linear aROC AUC "
                  f"{auc.min():.3f}-{auc.max():.3f}  "
                  f"(age {grid[AGE].iloc[0]:.0f}-{grid[AGE].iloc[-1]:.0f}, "
                  f"BMI at median {grid[BMI].iloc[0]:.1f})")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"\n{len(out)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
