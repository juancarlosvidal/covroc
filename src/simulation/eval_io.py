"""Shared output writer for the simulation-scenario baselines (naive_roc_baseline.py,
linear_reg_data_simulation.py), used the same way mlp_reg_data_simulation_multi.py and
rf_reg_data_simulation_multi_2.py write their own roc_mse_values.csv/timing.csv, so all
methods land in the same per-scenario, per-method comparison table produced by
src/postprocessing/statistics_summary.py (Reviewer 1, Major Concern 1 & 2).
"""
import os
import pandas as pd
from sklearn.metrics import mean_squared_error


def write_mse_and_timing(output_dir, sceminario, roc_predicted, roc_real, method, fit_seconds_total):
    """Writes roc_mse_values.csv (per-subject MSE between predicted and ground-truth
    ROC curves) and timing.csv (one row: scenario/method/fit time) into
    {output_dir}/{sceminario}/, matching the format the FNN/RF scripts use.
    """
    folder = os.path.join(output_dir, sceminario)
    os.makedirs(folder, exist_ok=True)

    mse_values = [mean_squared_error(real, predicted) for predicted, real in zip(roc_predicted, roc_real)]
    mse_df = pd.DataFrame(mse_values, columns=['Mean Squared Error'])
    mse_df.to_csv(os.path.join(folder, "roc_mse_values.csv"), index_label="Object Index")

    timing_df = pd.DataFrame([{
        'Scenario': sceminario,
        'Method': method,
        'Fit Seconds Total': fit_seconds_total,
    }])
    timing_df.to_csv(os.path.join(folder, "timing.csv"), index=False)

    return mse_values
