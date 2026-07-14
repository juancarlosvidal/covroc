"""Aggregates the per-scenario roc_mse_values.csv and timing.csv outputs of the
simulation pipeline into summary statistics (mean/median/SD/quantiles of AUC MSE and
of per-replicate fit time), the basis for the finite-sample MSE and computation-time
comparison across the nine scenarios reported in the paper's Supplementary Material
(FNN vs. Random Forest vs. semiparametric benchmark; Reviewer 1, Major Concern 1)."""
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
args = parser.parse_args()
root_dir = args.root_dir

# Initialize empty lists to store results
results = []
timing_rows = []

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
                    "Standard Deviation": mse_values.std(),
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

# Create a DataFrame from the results
table = pd.DataFrame(results)
# Save the table to a CSV file
table.to_csv(args.output_csv, index=False)

print(f"Table created and saved as {args.output_csv}")

# Aggregate per-replicate computation time (one timing.csv row per replicate folder)
# into per-scenario, per-method statistics -- e.g. FNN vs. Random Forest wall-clock
# fit time (Reviewer 1, Major Concern 1).
if timing_rows:
    def _scenario_id(scenario):
        match = re.match(r"^(scenario_\d+)_", scenario)
        return match.group(1) if match else scenario

    all_timing = pd.concat(timing_rows, ignore_index=True)
    all_timing["Scenario ID"] = all_timing["Scenario"].apply(_scenario_id)

    timing_summary = all_timing.groupby(["Scenario ID", "Method"])["Fit Seconds Total"].agg(
        Mean="mean", Median="median", **{"Standard Deviation": "std"}, Min="min", Max="max", N="count"
    ).reset_index()
    timing_summary.to_csv(args.timing_csv, index=False)
    print(f"Table created and saved as {args.timing_csv}")
else:
    print("No timing.csv files found; skipping timing summary.")

# %%
