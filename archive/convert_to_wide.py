"""Reshapes data_generation.py's long-form scenario CSVs (paired healthy '_bar'
and diseased columns per row) into the wide, group-labeled form ('mortstat' 0/1)
expected by the regression loaders in src/simulation/."""
#%%
#%%
import pandas as pd
import os

# Directory paths
input_dir = r'data_simulation'
output_dir = r'data_simulation_wide'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate through all scenario files
for scenario_file in os.listdir(input_dir):
    if scenario_file.endswith('.csv'):
        # Load the DataFrame
        scenario_path = os.path.join(input_dir, scenario_file)
        df = pd.read_csv(scenario_path)

        # Correct approach to construct the wide-form DataFrame
        # Separate the data for 'bar' and non-'bar' into two separate DataFrames
        bar_data = df.filter(regex='bar').copy()
        non_bar_data = df.filter(regex='^(?!.*bar).*$').copy()

        # Assign 'mortstat' column
        bar_data['mortstat'] = 0
        non_bar_data['mortstat'] = 1

        # Rename columns for consistency
        new_columns = [col.replace('_bar', '') for col in bar_data.columns]
        bar_data.columns = new_columns
        non_bar_data.columns = new_columns

        # Concatenate both DataFrames to create the wide form
        wide_df_final = pd.concat([bar_data, non_bar_data], ignore_index=True)

        # Save the new wide form CSV
        output_path = os.path.join(output_dir, scenario_file)
        wide_df_final.to_csv(output_path, index=False)

# %%
    