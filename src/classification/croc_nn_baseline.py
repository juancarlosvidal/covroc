
#%% Neural Network–Based cROC Estimation (using NN instead of OLS)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import simpson
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scipy.interpolate import interp1d
from scipy.integrate import simpson

from statsmodels.distributions.empirical_distribution import ECDF

# Import your NN training and helper functions:
from train import cv_loop
# (Make sure cv_loop, compute_residues, compute_mean, compute_std are in your path.)
# from your_nn_module import cv_loop, compute_residues, compute_mean, compute_std

# A helper to convert a DataFrame into the dictionary format expected by your NN code.
def prepare_nn_data(df, covariates, target):
    """
    Given a DataFrame and a list of covariates plus a target,
    returns a dictionary with keys 'x', 'y', 'w' (with w = 1s) and a cleaned DataFrame.
    """
    df_clean = df.dropna(subset=covariates + [target])
    x = df_clean[covariates].values.astype(np.float32)
    y = df_clean[target].values.astype(np.float32).reshape(-1, 1)
    w = np.ones_like(y)  # or use weights if available
    return {'x': x, 'y': y, 'w': w}, df_clean
def compute_residues(data, model):
    input = torch.from_numpy(data['x'])
    model.eval()
    with torch.no_grad():
        output = model(input).detach().cpu().numpy()
    # return np.abs(output - data['y'])
    return output - data['y']



def compute_mean(data, model):
    input = torch.from_numpy(data['x'])
    print('Input size {}'.format(input.shape))
    model.eval()
    with torch.no_grad():
        output = model(input).detach().cpu().numpy()
    return output
# NOT PQ NO LA MEDIA?


def compute_std(data, model):
    input = torch.from_numpy(data['x'])
    model.eval()
    with torch.no_grad():
        output = model(input).detach().cpu().numpy()
    # output[output < 0] = 0.001
    return np.sqrt(np.exp(output))
    # return np.sqrt(output)
def compute_ROC_nn(covariates, target, data, group, tag_h, newdata,
                   n_folds=2, n_epochs=800, batch_size=64, 
                   learning_rate=0.001, weight_decay=0.05,
                   p=np.linspace(0, 1, 101)):
    """
    Computes covariate-specific ROC curves and AUC using neural network models
    in place of OLS fits.
    
    Parameters:
      - covariates: list of predictor names (e.g. ['BMI', 'RIDAGEYR_x']).
      - target: response variable name (e.g. 'TAC2').
      - data: full DataFrame containing all subjects.
      - group: column name for grouping (e.g. 'cinco').
      - tag_h: value in `group` indicating a healthy subject.
      - newdata: DataFrame on which to compute the ROC.
      - (n_folds, n_epochs, batch_size, learning_rate, weight_decay): hyperparameters.
      - p: grid of false-positive fractions.
    
    Returns:
      A dictionary with:
         - 'ROC': an array (n_newdata x len(p)) of ROC curves,
         - 'AUC': an array (n_newdata,) of AUC values,
         - 'newdata': the newdata DataFrame (cleaned),
         - 'models': trained NN models (both for mean and variance).
    """
    # Split the data into healthy and diseased groups:
    data_h_df = data[data[group] == tag_h]
    data_d_df = data[data[group] != tag_h]
    
    # Prepare NN data (for both groups and for newdata):
    data_h_nn, data_h_df = prepare_nn_data(data_h_df, covariates, target)
    data_d_nn, data_d_df = prepare_nn_data(data_d_df, covariates, target)
    newdata_nn, newdata_df = prepare_nn_data(newdata, covariates, target)
    
    # --- Train NN Models for Mean Predictions ---
    print("Training NN model for healthy group (mean)...")
    model_h, _ = cv_loop(data_h_nn, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=True)
    print("Training NN model for diseased group (mean)...")
    model_d, _ = cv_loop(data_d_nn, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=True)
    
    # --- Train NN Models for Variance (Residue) Estimation ---
    # First, compute residues on the training data:
    residues_h = compute_residues(data_h_nn, model_h)
    residues_d = compute_residues(data_d_nn, model_d)
    
    # Prepare data dictionaries for residue modeling:
    data_h_resid = {'x': data_h_nn['x'], 'y': residues_h, 'w': data_h_nn['w']}
    data_d_resid = {'x': data_d_nn['x'], 'y': residues_d, 'w': data_d_nn['w']}
    
    print("Training NN model for healthy group (residues)...")
    model_h_resid, _ = cv_loop(data_h_resid, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=True)
    print("Training NN model for diseased group (residues)...")
    model_d_resid, _ = cv_loop(data_d_resid, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=True)
    
    # --- Get Predictions on New Data ---
    # Mean predictions:
    mean_h = compute_mean(newdata_nn, model_h)   # shape: (n_newdata, 1)
    mean_d = compute_mean(newdata_nn, model_d)   # shape: (n_newdata, 1)
    # Standard deviation (from NN variance model):
    std_h = compute_std(newdata_nn, model_h_resid)  # shape: (n_newdata, 1)
    std_d = compute_std(newdata_nn, model_d_resid)  # shape: (n_newdata, 1)
    
    # --- Compute 'a' and 'b' as in the parametric ROC model ---
    a_pred = np.where(mean_d > mean_h,
                      (mean_d - mean_h) / std_d,
                      (mean_h - mean_d) / std_h)
    b_pred = np.where(mean_d > mean_h,
                      std_h / std_d,
                      std_d / std_h)
    
    # --- Compute ROC Curves and AUC ---
    # For each newdata observation, for each FPF value p:
    ROC_curves = []
    AUCs = []
    for i in range(len(a_pred)):
        # For observation i, compute:
        #   ROC(p) = 1 - Phi( b_pred[i]*Phi^{-1}(1-p) - a_pred[i] )
        roc_i = 1 - norm.cdf(norm.ppf(1 - p) * b_pred[i] - a_pred[i])
        ROC_curves.append(roc_i)
        # Compute AUC via numerical integration:
        auc_i = simpson(roc_i, p)
        AUCs.append(auc_i)
    
    ROC_curves = np.array(ROC_curves)  # shape: (n_newdata, len(p))
    AUCs = np.array(AUCs)              # shape: (n_newdata,)
    
    result = {
        'p': p,
        'ROC': ROC_curves,
        'AUC': AUCs,
        'newdata': newdata_df,
        'models': {
            'mean': {'healthy': model_h, 'diseased': model_d},
            'resid': {'healthy': model_h_resid, 'diseased': model_d_resid}
        }
    }
    return result

# --- Example Usage ---

# Run from the repository root. input_real_2/df_gender_2.csv is produced upstream
# by the R data-prep step (see R/aroc_single_model.R).
df2 = pd.read_csv('input_real_2/df_gender_2.csv')
df2 = df2.rename(columns={'RIDAGEYR.x': 'RIDAGEYR_x'})

# Specify the covariates and target variable.
covariates = ['BMI', 'RIDAGEYR_x']
target = 'TAC2'
group = 'cinco'  # grouping variable; healthy subjects have value 0.
tag_h = 0

# Compute the NN-based ROC (and AUC) on the newdata (here we use the same df2).
nn_cROC = compute_ROC_nn(covariates, target, df2, group, tag_h, df2,
                         n_folds=2, n_epochs=800, batch_size=64,
                         learning_rate=0.001, weight_decay=0.05,
                         p=np.linspace(0, 1, 101))

# --- Plotting: For example, 1 - AUC vs. Age ---
plt.figure(figsize=(10, 5))
# Here we assume that the newdata DataFrame contains the column 'RIDAGEYR_x'
plt.scatter(nn_cROC['newdata']['RIDAGEYR_x'], 1- nn_cROC['AUC'], 
            color='blue', alpha=0.6, label='1 - AUC')
plt.xlabel("RIDAGEYR_x")
plt.ylabel("1 - AUC")
plt.title("NN-based cROC Plot")
plt.legend()
plt.grid(True)
plt.savefig("nn_cROC_plot.png")
plt.show()

# %%
