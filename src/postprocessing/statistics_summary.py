"""Aggregates the per-scenario roc_mse_values.csv outputs of the simulation pipeline
into summary statistics (mean/median/SD/quantiles of AUC MSE), the basis for the
finite-sample MSE comparison across the nine scenarios reported in the paper's
Supplementary Material (FNN vs. Random Forest vs. semiparametric benchmark)."""
#%%
import argparse
import os
import pandas as pd

# Define the root directory where folders are stored
def transform_folder_name(original_name):
    parts = original_name.split('_')
    if len(parts) == 2 and parts[0] == "scenario":
        number = parts[1]
        return f"scenario_{number[0]}_{number[1:]}"
    else:
        return original_name

parser = argparse.ArgumentParser(description="Summarize MSE statistics across output scenario folders")
parser.add_argument("--root-dir", default="output", help="Directory containing one subfolder per scenario, each with a roc_mse_values.csv")
parser.add_argument("--output-csv", default="statistics_summary.csv", help="Path to write the summary table")
args = parser.parse_args()
root_dir = args.root_dir

# Initialize an empty list to store results
results = []

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

# Create a DataFrame from the results
table = pd.DataFrame(results)
# Save the table to a CSV file
table.to_csv(args.output_csv, index=False)

print(f"Table created and saved as {args.output_csv}")

# %%
