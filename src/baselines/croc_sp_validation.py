"""Validates that cROC_sp (croc_linear_baseline.py, a Python port of ROCnReg::cROC.sp)
agrees numerically with the real R package on the same simulation replicate -- confirms
the results linear_reg_data_simulation.py already produced at scale (Reviewer 1, Major
Concern 2) can be trusted without also running the real R package across all 9 scenarios.

One-off diagnostic, not a pipeline step: meant to be diffed against R/croc_sp_validation.R's
output on the identical CSV. Uses newdata=df (every subject's own covariates, shared
between the h/d model evaluations) since that's the only mode ROCnReg::cROC.sp itself
supports -- not the row-paired newdata_h/newdata_d extension linear_reg_data_simulation.py
uses for the ground-truth comparison.
"""
import argparse
import os

import numpy as np
import pandas as pd

from croc_linear_baseline import cROC_sp


def main(csv_file, covariates):
    df = pd.read_csv(csv_file)
    formula = "Y_generated ~ " + " + ".join(covariates)

    print(f"File: {csv_file}\nFormula: {formula}\nN: {len(df)} "
          f"(healthy={(df['mortstat'] == 0).sum()}, diseased={(df['mortstat'] == 1).sum()})\n")

    res = cROC_sp(
        formula_h=formula, formula_d=formula,
        group='mortstat', tag_h=0, data=df, newdata=df,
        p=np.linspace(0, 1, 101), B=0,
    )

    scenario_name = os.path.splitext(os.path.basename(csv_file))[0]
    out_dir = "output/croc_sp_validation"
    os.makedirs(out_dir, exist_ok=True)

    auc_out = pd.DataFrame({"Row": np.arange(1, len(df) + 1), "AUC": res['AUC'][:, 0]})
    auc_file = os.path.join(out_dir, f"{scenario_name}_auc_python.csv")
    auc_out.to_csv(auc_file, index=False)

    roc_out = pd.DataFrame({"p": res['p'], "ROC": res['ROC']['est'][0, :]})
    roc_file = os.path.join(out_dir, f"{scenario_name}_roc_row1_python.csv")
    roc_out.to_csv(roc_file, index=False)

    print(f"Wrote {auc_file} and {roc_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cROC_sp on one replicate for R-vs-Python validation")
    parser.add_argument("csv_file", help="Scenario replicate CSV (e.g. input_real_2/scenario_1/scenario_1_5000_1_data.csv)")
    parser.add_argument("covariates", nargs="+", help="Covariate column names, matching true_dgp.covariate_columns(scenario)")
    args = parser.parse_args()
    main(args.csv_file, args.covariates)
