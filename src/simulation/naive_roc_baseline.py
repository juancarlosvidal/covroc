"""Naive (pooled/marginal) ROC baseline for the nine simulation scenarios: ignores
covariates entirely and estimates a single ROC(p) curve from the pooled Y values of
each group, via the same empirical-CDF construction used for the ground truth
(ground_truth_auc.roc_from_samples). Requested by Reviewer 1 (Major Concern 2) as the
"naive traditional ROC curve with no covariate adjustment" comparison point.

Evaluated against the same per-subject ground-truth aROC(p|x) as the FNN/RF/linear
baselines (even though its own prediction doesn't vary with x), writing
roc_mse_values.csv/timing.csv in the same format so it lands in the same
statistics_summary.py comparison table.
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd

import true_dgp
import ground_truth_auc
import eval_io

P_GRID = np.linspace(0.001, 0.999, 100)


def main(input_dir, output_dir):
    only_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for f in only_files:
        file = os.path.join(input_dir, f)
        df = pd.read_csv(file)
        dataset_name = os.path.splitext(os.path.basename(file))[0]
        match = re.match(r'^(scenario_\d+_\d+_\d+)_data$', dataset_name)
        sceminario = match.group(1) if match else dataset_name
        scenario_num = int(re.match(r'^scenario_(\d+)_', f).group(1))

        index_0 = df[df['mortstat'] == 0].copy()
        index_1 = df[df['mortstat'] == 1].copy()

        ground_truth_cols = true_dgp.covariate_columns(scenario_num)
        X0_true = index_0[ground_truth_cols].to_numpy(dtype=float)
        X1_true = index_1[ground_truth_cols].to_numpy(dtype=float)

        t0 = time.perf_counter()
        naive_curve = ground_truth_auc.roc_from_samples(
            index_0['Y_generated'].to_numpy(), index_1['Y_generated'].to_numpy(), P_GRID
        )
        fit_seconds_total = time.perf_counter() - t0

        # Same pooled curve for every subject -- the naive estimator doesn't condition on x.
        roc_predicted = [naive_curve] * len(index_0)
        rng = np.random.default_rng(ground_truth_auc.seed_from_name(sceminario))
        roc_real = list(ground_truth_auc.true_roc_curve(scenario_num, X0_true, X1_true, p=P_GRID, rng=rng))

        eval_io.write_mse_and_timing(output_dir, sceminario, roc_predicted, roc_real, 'Naive', fit_seconds_total)
        print(f'{sceminario}: naive pooled ROC vs. ground truth written to {output_dir}/{sceminario}/')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Naive (pooled, no covariate adjustment) ROC baseline')
    parser.add_argument('-i', '--input_dir', default="./input_real_2", help='Input directory (one scenario\'s replicate CSVs)')
    parser.add_argument('-o', '--output_dir', default="./output", help='Output directory')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
