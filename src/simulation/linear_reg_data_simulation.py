"""Linear and semiparametric-additive aROC baselines for the nine simulation
scenarios, reusing the location-scale cROC_sp estimator from
src/baselines/croc_linear_baseline.py (Reviewer 1, Major Concern 2: "a linear model
approach for aROC" and "the semiparametric additive GAC/ROC model"). Both are the same
estimator; only the formula changes:
  --formula-type linear  ->  Y_generated ~ x_D_1 + ...             (plain OLS)
  --formula-type spline  ->  Y_generated ~ bs(x_D_1, df=K) + ...   (additive splines)

Evaluated against the same ground-truth aROC(p|x) as the FNN/RF/naive baselines
(ground_truth_auc.true_roc_curve), writing roc_mse_values.csv/timing.csv in the same
format so all methods land in the same statistics_summary.py comparison table.
"""
import argparse
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

import true_dgp
import ground_truth_auc
import eval_io

# croc_linear_baseline.py lives in src/baselines/, not alongside this script; this is
# the one cross-directory import in the simulation pipeline, needed because the
# reusable cROC_sp estimator (shared with the real-data NHANES case study) lives there.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
from croc_linear_baseline import cROC_sp

P_GRID = np.linspace(0.001, 0.999, 100)

METHOD_NAMES = {
    'linear': 'Linear',
    'spline': 'Semiparametric additive',
}


def build_formula(target, covariate_cols, formula_type, spline_df):
    if formula_type == 'linear':
        terms = covariate_cols
    else:
        terms = [f"bs({c}, df={spline_df})" for c in covariate_cols]
    return f"{target} ~ " + " + ".join(terms)


def main(input_dir, output_dir, formula_type, spline_df):
    only_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    method = METHOD_NAMES[formula_type]

    for f in only_files:
        file = os.path.join(input_dir, f)
        df = pd.read_csv(file)
        dataset_name = os.path.splitext(os.path.basename(file))[0]
        match = re.match(r'^(scenario_\d+_\d+_\d+)_data$', dataset_name)
        sceminario = match.group(1) if match else dataset_name
        scenario_num = int(re.match(r'^scenario_(\d+)_', f).group(1))

        index_0 = df[df['mortstat'] == 0].copy().reset_index(drop=True)
        index_1 = df[df['mortstat'] == 1].copy().reset_index(drop=True)

        ground_truth_cols = true_dgp.covariate_columns(scenario_num)
        X0_true = index_0[ground_truth_cols].to_numpy(dtype=float)
        X1_true = index_1[ground_truth_cols].to_numpy(dtype=float)

        formula = build_formula('Y_generated', ground_truth_cols, formula_type, spline_df)

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = cROC_sp(
                formula_h=formula, formula_d=formula,
                group='mortstat', tag_h=0, data=df,
                newdata_h=index_0[ground_truth_cols], newdata_d=index_1[ground_truth_cols],
                p=P_GRID, B=0,
            )
        fit_seconds_total = time.perf_counter() - t0

        roc_predicted = list(res['ROC']['est'])
        rng = np.random.default_rng(ground_truth_auc.seed_from_name(sceminario))
        roc_real = list(ground_truth_auc.true_roc_curve(scenario_num, X0_true, X1_true, p=P_GRID, rng=rng))

        eval_io.write_mse_and_timing(output_dir, sceminario, roc_predicted, roc_real, method, fit_seconds_total)
        print(f'{sceminario}: {method} aROC vs. ground truth written to {output_dir}/{sceminario}/')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear / semiparametric-additive aROC baseline')
    parser.add_argument('-i', '--input_dir', default="./input_real_2", help='Input directory (one scenario\'s replicate CSVs)')
    parser.add_argument('-o', '--output_dir', default="./output", help='Output directory')
    parser.add_argument('--formula-type', choices=['linear', 'spline'], default='linear',
                         help='"linear" for a plain OLS aROC, "spline" for the additive-spline semiparametric aROC')
    parser.add_argument('--spline-df', type=int, default=4, help='Degrees of freedom per covariate B-spline (--formula-type spline only)')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.formula_type, args.spline_df)
