"""Aggregates the per-scenario roc_mse_values.csv, timing.csv, and mean_std_mse.csv
outputs of the simulation pipeline into summary statistics (mean/median/SD/quantiles of
AUC MSE, of per-replicate fit time, and of the mean-/std-function estimation MSE), the
basis for the finite-sample MSE and computation-time comparison across the nine scenarios
reported in the paper's Supplementary Material (FNN vs. Random Forest vs. semiparametric
benchmark; Reviewer 1, Major Concern 1).

The "MSE SD Across Runs" column here (spread, across the 100 replicates, of the
ROC-curve MSE) is a different quantity from the per-replicate "Std-Function MSE" in
mean_std_mse_summary.csv (MSE of the sigma(x) estimator itself) -- named to avoid the
"MSE Std" vs. "mse_std" collision flagged in Reviewer 1's Minor Concern 4.
"""
#%%
import argparse
import os
import re
import pandas as pd

# Define the root directory where folders are stored
def transform_folder_name(original_name):
    parts = original_name.split('_')
    if len(parts) == 2 and parts[0] == "scenario":
        number = parts[1]
        return f"scenario_{number[0]}_{number[1:]}"
    else:
        return original_name

parser = argparse.ArgumentParser(description="Summarize MSE and timing statistics across output scenario folders")
parser.add_argument("--root-dir", default="output", help="Directory containing one subfolder per scenario/replicate, each with a roc_mse_values.csv and a timing.csv")
parser.add_argument("--output-csv", default="statistics_summary.csv", help="Path to write the MSE summary table")
parser.add_argument("--timing-csv", default="timing_summary.csv", help="Path to write the per-scenario, per-method computation-time summary table")
parser.add_argument("--mean-std-mse-csv", default="mean_std_mse_summary.csv", help="Path to write the per-scenario, per-method, per-group mean-/std-function MSE summary table")
args = parser.parse_args()
root_dir = args.root_dir

# Initialize empty lists to store results
results = []
timing_rows = []
mean_std_mse_rows = []

# Loop through each folder in the root directory
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        csv_file = os.path.join(folder_path, "roc_mse_values.csv")

        # Check if the CSV file exists
        if os.path.exists(csv_file):
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Check if the 'Mean Squared Error' column exists
            if 'Mean Squared Error' in df.columns:
                mse_values = df['Mean Squared Error']

                # Calculate statistics
                stats = {
                    "Folder Name": transform_folder_name(folder),
                    "Mean": mse_values.mean(),
                    "Median": mse_values.median(),
                    "MSE SD Across Runs": mse_values.std(),
                    "Variance": mse_values.var(),
                    "Min": mse_values.min(),
                    "Max": mse_values.max(),
                    "Q1": mse_values.quantile(0.25),
                    "Q3": mse_values.quantile(0.75)
                }

                # Append the results
                results.append(stats)

        timing_file = os.path.join(folder_path, "timing.csv")
        if os.path.exists(timing_file):
            timing_rows.append(pd.read_csv(timing_file))

        mean_std_mse_file = os.path.join(folder_path, "mean_std_mse.csv")
        if os.path.exists(mean_std_mse_file):
            mean_std_mse_rows.append(pd.read_csv(mean_std_mse_file))

# Create a DataFrame from the results
table = pd.DataFrame(results)
# Save the table to a CSV file
table.to_csv(args.output_csv, index=False)

print(f"Table created and saved as {args.output_csv}")

def _scenario_id(scenario):
    match = re.match(r"^(scenario_\d+)_", scenario)
    return match.group(1) if match else scenario

# Aggregate per-replicate computation time (one timing.csv row per replicate folder)
# into per-scenario, per-method statistics -- e.g. FNN vs. Random Forest wall-clock
# fit time (Reviewer 1, Major Concern 1).
if timing_rows:
    all_timing = pd.concat(timing_rows, ignore_index=True)
    all_timing["Scenario ID"] = all_timing["Scenario"].apply(_scenario_id)

    timing_summary = all_timing.groupby(["Scenario ID", "Method"])["Fit Seconds Total"].agg(
        Mean="mean", Median="median", **{"Standard Deviation": "std"}, Min="min", Max="max", N="count"
    ).reset_index()
    timing_summary.to_csv(args.timing_csv, index=False)
    print(f"Table created and saved as {args.timing_csv}")
else:
    print("No timing.csv files found; skipping timing summary.")

# Aggregate per-replicate Mean-Function/Std-Function MSE (one mean_std_mse.csv per
# replicate folder, two rows: healthy/diseased) into per-scenario, per-method,
# per-group statistics across the 100 replicates -- feeds mean_std_mse_boxplots.py
# (Reviewer 1, Minor Concerns 3 & 4).
if mean_std_mse_rows:
    all_mean_std_mse = pd.concat(mean_std_mse_rows, ignore_index=True)
    all_mean_std_mse["Scenario ID"] = all_mean_std_mse["Scenario"].apply(_scenario_id)

    mean_std_mse_summary = all_mean_std_mse.groupby(["Scenario ID", "Method", "Group"])[
        ["Mean-Function MSE", "Std-Function MSE"]
    ].agg(["mean", "median", "std", "min", "max", "count"])
    mean_std_mse_summary.columns = [f"{metric} {stat.capitalize()}" for metric, stat in mean_std_mse_summary.columns]
    mean_std_mse_summary = mean_std_mse_summary.reset_index()
    mean_std_mse_summary.to_csv(args.mean_std_mse_csv, index=False)
    print(f"Table created and saved as {args.mean_std_mse_csv}")
else:
    print("No mean_std_mse.csv files found; skipping mean/std-function MSE summary.")

# %%
