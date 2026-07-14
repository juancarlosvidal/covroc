"""Generates the nine synthetic scenarios (Scenarios I-IX) used to benchmark the
FNN-based aROC estimator in the paper's Supplementary Material.

Each scenario draws a healthy group (D=0, columns suffixed '_bar') and a diseased
group (D=1) from Y | X = mu_d(x) + sigma_d(x) * eps under a range of covariate-mean
relationships -- constant, linear, non-linear, and multivariate interaction/mixture
effects -- with sigma_0 = 0.5 and sigma_1 = 1 held fixed except where a scenario
varies it explicitly. These mimic the linear, non-linear, and interaction-driven
covariate effects described in the paper's simulation study.

The conditional mean/std/sampling formulas live in true_dgp.py (the single source of
truth also used by the ground-truth AUC comparison in ground_truth_auc.py) so the two
can never drift apart.
"""
#%%




import numpy as np
import pandas as pd
from scipy.stats import skewnorm, t
import os

import true_dgp

# Create directory if it doesn't exist
os.makedirs("data_simulation", exist_ok=True)



# Scenario I

def generate_scenario_I(n):
    x_D_bar_1 = np.random.normal(size=n)
    x_D_1 = np.random.normal(size=n)
    true_mean_Y_bar = true_dgp.true_mean(1, 0, x_D_bar_1[:, None])
    true_mean_Y = true_dgp.true_mean(1, 1, x_D_1[:, None])
    Y_generated_bar = true_dgp.sample_conditional(1, 0, x_D_bar_1[:, None]).ravel()
    Y_generated = true_dgp.sample_conditional(1, 1, x_D_1[:, None]).ravel()

    healthy_data = pd.DataFrame({
        'Y_generated': Y_generated_bar,
        'True_Mean_Y': true_mean_Y_bar,
        'x_D_1': x_D_bar_1,
        'mortstat': 0
    })
    diseased_data = pd.DataFrame({
        'Y_generated': Y_generated,
        'True_Mean_Y': true_mean_Y,
        'x_D_1': x_D_1,
        'mortstat': 1
    })

    return pd.concat([healthy_data, diseased_data], ignore_index=True)

# Scenario II
def generate_scenario_II(n):
    x_D_bar_1 = np.random.normal(size=n)
    x_D_1 = np.random.normal(size=n)
    true_mean_Y_bar = true_dgp.true_mean(2, 0, x_D_bar_1[:, None])
    true_mean_Y = true_dgp.true_mean(2, 1, x_D_1[:, None])

    Y_generated_bar = true_dgp.sample_conditional(2, 0, x_D_bar_1[:, None]).ravel()
    Y_generated = true_dgp.sample_conditional(2, 1, x_D_1[:, None]).ravel()

    healthy_data = pd.DataFrame({
        'Y_generated': Y_generated_bar,
        'True_Mean_Y': true_mean_Y_bar,
        'x_D_1': x_D_bar_1,
        'mortstat': 0
    })
    diseased_data = pd.DataFrame({
        'Y_generated': Y_generated,
        'True_Mean_Y': true_mean_Y,
        'x_D_1': x_D_1,
        'mortstat': 1
    })

    return pd.concat([healthy_data, diseased_data], ignore_index=True)

# Scenario III
def generate_scenario_III(n):
    x_D_bar_1 = np.random.normal(size=n)
    x_D_1 = np.random.normal(size=n)
    true_mean_Y_bar = true_dgp.true_mean(3, 0, x_D_bar_1[:, None])
    true_mean_Y = true_dgp.true_mean(3, 1, x_D_1[:, None])
    Y_generated_bar = true_dgp.sample_conditional(3, 0, x_D_bar_1[:, None]).ravel()
    Y_generated = true_dgp.sample_conditional(3, 1, x_D_1[:, None]).ravel()


    healthy_data = pd.DataFrame({
        'Y_generated': Y_generated_bar,
        'True_Mean_Y': true_mean_Y_bar,
        'x_D_1': x_D_bar_1,
        'mortstat': 0
    })
    diseased_data = pd.DataFrame({
        'Y_generated': Y_generated,
        'True_Mean_Y': true_mean_Y,
        'x_D_1': x_D_1,
        'mortstat': 1
    })

    return pd.concat([healthy_data, diseased_data], ignore_index=True)

# Scenario IV
def generate_scenario_IV(n):
    x_D_bar_1 = np.random.normal(size=n)
    x_D_1 = np.random.normal(size=n)
    true_mean_Y_bar = true_dgp.true_mean(4, 0, x_D_bar_1[:, None])
    true_mean_Y = true_dgp.true_mean(4, 1, x_D_1[:, None])
    Y_generated_bar = true_dgp.sample_conditional(4, 0, x_D_bar_1[:, None]).ravel()
    Y_generated = true_dgp.sample_conditional(4, 1, x_D_1[:, None]).ravel()

    healthy_data = pd.DataFrame({
        'Y_generated': Y_generated_bar,
        'True_Mean_Y': true_mean_Y_bar,
        'x_D_1': x_D_bar_1,
        'mortstat': 0
    })
    diseased_data = pd.DataFrame({
        'Y_generated': Y_generated,
        'True_Mean_Y': true_mean_Y,
        'x_D_1': x_D_1,
        'mortstat': 1
    })

    return pd.concat([healthy_data, diseased_data], ignore_index=True)

# Add similar structure for other scenarios
# The same structure can be repeated for Scenarios V through IX with the specific `true_mean_Y_bar` and `true_mean_Y` calculations for each scenario.


# Scenario V
def generate_scenario_V(n):
    x_D_bar_1 = np.random.normal(size=n)
    x_D_bar_2 = np.random.normal(size=n)
    x_D_1 = np.random.normal(size=n)
    x_D_2 = np.random.normal(size=n)

    true_mean_Y_bar = true_dgp.true_mean(5, 0, np.column_stack([x_D_bar_1, x_D_bar_2]))
    true_mean_Y = true_dgp.true_mean(5, 1, np.column_stack([x_D_1, x_D_2]))
    Y_generated_bar = true_dgp.sample_conditional(5, 0, np.column_stack([x_D_bar_1, x_D_bar_2])).ravel()
    Y_generated = true_dgp.sample_conditional(5, 1, np.column_stack([x_D_1, x_D_2])).ravel()

    healthy_data = pd.DataFrame({
        'Y_generated': Y_generated_bar,
        'True_Mean_Y': true_mean_Y_bar,
        'x_D_1': x_D_bar_1,
        'x_D_2': x_D_bar_2,
        'mortstat': 0
    })

    diseased_data = pd.DataFrame({
        'Y_generated': Y_generated,
        'True_Mean_Y': true_mean_Y,
        'x_D_1': x_D_1,
        'x_D_2': x_D_2,
        'mortstat': 1
    })

    return pd.concat([healthy_data, diseased_data], ignore_index=True)

# Scenario VI
def generate_scenario_VI(n):
    x_D_bar_1 = np.random.normal(size=n)
    x_D_bar_3 = np.random.binomial(1, 0.5, size=n)
    x_D_1 = np.random.normal(size=n)
    x_D_3 = np.random.normal(size=n)

    true_mean_Y_bar = true_dgp.true_mean(6, 0, np.column_stack([x_D_bar_1, x_D_bar_3]))
    true_mean_Y = true_dgp.true_mean(6, 1, np.column_stack([x_D_1, x_D_3]))

    Y_generated_bar = true_dgp.sample_conditional(6, 0, np.column_stack([x_D_bar_1, x_D_bar_3])).ravel()
    Y_generated = true_dgp.sample_conditional(6, 1, np.column_stack([x_D_1, x_D_3])).ravel()

    healthy_data = pd.DataFrame({
        'Y_generated': Y_generated_bar,
        'True_Mean_Y': true_mean_Y_bar,
        'x_D_1': x_D_bar_1,
        'x_D_3': x_D_bar_3,
        'mortstat': 0
    })

    diseased_data = pd.DataFrame({
        'Y_generated': Y_generated,
        'True_Mean_Y': true_mean_Y,
        'x_D_1': x_D_1,
        'x_D_3': x_D_3,
        'mortstat': 1
    })

    return pd.concat([healthy_data, diseased_data], ignore_index=True)

# Scenario VII
def generate_scenario_VII(n):
    x_D_bar_4 = np.random.uniform(0, 1, size=n)
    x_D_4 = np.random.uniform(0, 1, size=n)

    true_mean_Y_bar = true_dgp.sample_conditional(7, 0, x_D_bar_4[:, None]).ravel()
    true_mean_Y = true_dgp.true_mean(7, 1, x_D_4[:, None])

    Y_generated_bar = true_mean_Y_bar
    Y_generated = true_dgp.sample_conditional(7, 1, x_D_4[:, None]).ravel()

    healthy_data = pd.DataFrame({
        'Y_generated': Y_generated_bar,
        'True_Mean_Y': true_mean_Y_bar,
        'x_D_1': x_D_bar_4,
        'mortstat': 0
    })

    diseased_data = pd.DataFrame({
        'Y_generated': Y_generated,
        'True_Mean_Y': true_mean_Y,
        'x_D_1': x_D_4,
        'mortstat': 1
    })

    return pd.concat([healthy_data, diseased_data], ignore_index=True)

# Scenario VIII
def generate_scenario_VIII(n):
    # Generate four continuous covariates for healthy and diseased groups
    x_D_bar_5 = np.random.normal(size=n)
    x_D_bar_6 = np.random.normal(size=n)
    x_D_bar_7 = np.random.normal(size=n)
    x_D_bar_8 = np.random.normal(size=n)

    x_D_5 = np.random.normal(size=n)
    x_D_6 = np.random.normal(size=n)
    x_D_7 = np.random.normal(size=n)
    x_D_8 = np.random.normal(size=n)

    # Outcome distributions remain the same as Scenario I
    true_mean_Y_bar = true_dgp.true_mean(8, 0, np.column_stack([x_D_bar_5, x_D_bar_6, x_D_bar_7, x_D_bar_8]))
    true_mean_Y = true_dgp.true_mean(8, 1, np.column_stack([x_D_5, x_D_6, x_D_7, x_D_8]))

    Y_generated_bar = true_dgp.sample_conditional(8, 0, np.column_stack([x_D_bar_5, x_D_bar_6, x_D_bar_7, x_D_bar_8])).ravel()
    Y_generated = true_dgp.sample_conditional(8, 1, np.column_stack([x_D_5, x_D_6, x_D_7, x_D_8])).ravel()

    # Create data for the healthy group
    healthy_data = pd.DataFrame({
        'Y_generated': Y_generated_bar,
        'True_Mean_Y': true_mean_Y_bar,
        'x_D_1': x_D_bar_5,
        'x_D_6': x_D_bar_6,
        'x_D_7': x_D_bar_7,
        'x_D_8': x_D_bar_8,
        'mortstat': 0  # Healthy group
    })

    # Create data for the diseased group
    diseased_data = pd.DataFrame({
        'Y_generated': Y_generated,
        'True_Mean_Y': true_mean_Y,
        'x_D_1': x_D_5,
        'x_D_6': x_D_6,
        'x_D_7': x_D_7,
        'x_D_8': x_D_8,
        'mortstat': 1  # Diseased group
    })

    # Combine healthy and diseased data
    return pd.concat([healthy_data, diseased_data], ignore_index=True)

# Scenario IX
def generate_scenario_IX(n):

    x_D_bar_5 = np.random.uniform(-1, 1, size=n)
    x_D_bar_6 = np.random.uniform(-1, 1, size=n)
    x_D_bar_7 = np.random.uniform(-1, 1, size=n)
    x_D_bar_8 = np.random.uniform(-1, 1, size=n)
    x_D_5 = np.random.uniform(-1, 1, size=n)
    x_D_6 = np.random.uniform(-1, 1, size=n)
    x_D_7 = np.random.uniform(-1, 1, size=n)
    x_D_8 = np.random.uniform(-1, 1, size=n)
    true_mean_Y_bar = true_dgp.true_mean(9, 0, np.column_stack([x_D_bar_5, x_D_bar_6, x_D_bar_7, x_D_bar_8]))
    true_mean_Y = true_dgp.true_mean(9, 1, np.column_stack([x_D_5, x_D_6, x_D_7, x_D_8]))

    Y_generated_bar = true_dgp.sample_conditional(9, 0, np.column_stack([x_D_bar_5, x_D_bar_6, x_D_bar_7, x_D_bar_8])).ravel()
    Y_generated = true_dgp.sample_conditional(9, 1, np.column_stack([x_D_5, x_D_6, x_D_7, x_D_8])).ravel()

    healthy_data = pd.DataFrame({
        'Y_generated': Y_generated_bar,
        'True_Mean_Y': true_mean_Y_bar,
        'x_D_1': x_D_bar_5,
        'x_D_6': x_D_bar_6,
        'x_D_7': x_D_bar_7,
        'x_D_8': x_D_bar_8,
        'mortstat': 0
    })

    diseased_data = pd.DataFrame({
        'Y_generated': Y_generated,
        'True_Mean_Y': true_mean_Y,
        'x_D_1': x_D_5,
        'x_D_6': x_D_6,
        'x_D_7': x_D_7,
        'x_D_8': x_D_8,
        'mortstat': 1
    })

    return pd.concat([healthy_data, diseased_data], ignore_index=True)


dir_path = "input_real_2"
os.makedirs(dir_path, exist_ok=True)

# Sample size
sample_sizes = [5000,20000]

# define the number of generations of the same dataset
ng = 100
for g in range (ng):

    # Generating data and saving as CSV files for each scenario
    scenarios_names = ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4',
                       'scenario_5','scenario_6','scenario_7',
                    'scenario_8','scenario_9']

    scenarios = [generate_scenario_I, generate_scenario_II, generate_scenario_III, generate_scenario_IV,generate_scenario_V,generate_scenario_VI,generate_scenario_VII,generate_scenario_VIII,generate_scenario_IX]
    for n in sample_sizes:
        for i, generate_fn in enumerate(scenarios, 1):
            os.makedirs(f"input_real_2/{scenarios_names[i-1]}", exist_ok=True)
            df = generate_fn(n)
            df.to_csv(f"input_real_2/{scenarios_names[i-1]}/scenario_{i}_{n}_{g+1}_data.csv", index=False)

# %%
