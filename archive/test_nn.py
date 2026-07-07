#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import random
import os 
from scipy.stats import norm
import glob
from data import load_data, CustomDataset, create_dict_2

# --------------------------
# Define a simple feed-forward network
# --------------------------
class Net(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --------------------------
# Helper functions for training/evaluation
# --------------------------
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)
    return total_loss / count

# --------------------------
# Cross-validation loop for neural network training
# --------------------------
def cv_loop_nn(data, n_folds, num_epochs, batch_size, learning_rate, hidden_size, random_seed=0):
    # Convert data to torch tensors (assume data['x'] is numpy array of shape (n_samples, n_features))
    X = torch.from_numpy(data['x']).float()
    # Ensure targets are float and 2D (n_samples, 1)
    y = torch.from_numpy(np.array(data['y']).reshape(-1, 1)).float()
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    fold_losses = []
    models = []
    input_dim = X.shape[1]
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = Net(input_dim, hidden_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        model = train_model(model, train_loader, criterion, optimizer, num_epochs)
        val_loss = evaluate_model(model, val_loader, criterion)
        fold_losses.append(val_loss)
        models.append(model)
    
    avg_loss = np.mean(fold_losses)
    # Optionally, choose the model with the lowest validation loss
    best_model = models[np.argmin(fold_losses)]
    
    return best_model, avg_loss

# --------------------------
# Grid search over neural network hyperparameters
# --------------------------
def grid_search_nn(data, n_folds, batch_size, param_grid, random_seed=0):
    best_model = None
    best_score = float('inf')
    best_params = None
    
    # Loop over each combination in the grid
    for hidden_size in param_grid['hidden_size']:
        for learning_rate in param_grid['learning_rate']:
            for num_epochs in param_grid['num_epochs']:
                print(f"Training NN with hidden_size={hidden_size}, lr={learning_rate}, epochs={num_epochs}")
                model, avg_loss = cv_loop_nn(data, n_folds, num_epochs, batch_size, learning_rate, hidden_size, random_seed)
                print(f"Avg validation MSE: {avg_loss}")
                if avg_loss < best_score:
                    best_score = avg_loss
                    best_model = model
                    best_params = {
                        'hidden_size': hidden_size,
                        'learning_rate': learning_rate,
                        'num_epochs': num_epochs
                    }
    print("Best hyperparameters:", best_params)
    print("Best CV MSE:", best_score)
    return best_model, best_score, best_params

# --------------------------
# Functions to compute predictions, residues, etc.
# --------------------------
def compute_residues(data, model):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(data['x']).float()
        outputs = model(inputs).cpu().numpy().ravel()
    y_true = np.array(data['y']).ravel()
    return (y_true - outputs)**2 

def compute_mean(data, model):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(data['x']).float()
        outputs = model(inputs).cpu().numpy().ravel()
    return outputs

def compute_std(data, model):
    # For standard deviation, assume that 'model' was trained on squared residues.
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(data['x']).float()
        outputs = model(inputs).cpu().numpy().ravel()
    # Avoid division by zero or negative sqrt by replacing negatives with a small value
    outputs = np.where(outputs <= 0, 1e-6, np.sqrt(outputs))
    return outputs

def roc(a, b, res_0, res_1):
    res_0 = res_0.squeeze()
    res_1 = res_1.squeeze()
    ecdf_0 = ECDF(res_0)
    ecdf_1 = ECDF(res_1)

    sample_edf = ecdf_0
    slope_changes = sorted(set(res_0))
    sample_edf_values_at_slope_changes = [sample_edf(item) for item in slope_changes]
    inverted_edf_0 = interp1d(sample_edf_values_at_slope_changes, slope_changes, fill_value="extrapolate")

    p = np.linspace(0.001, 0.999, num=100)
    r = 1 - ecdf_1(inverted_edf_0(1-p) * b - a)
    i1 = simpson(r, p)
    return i1

# --------------------------
# Main program
# --------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Neural Network regression with grid search')
    parser.add_argument('-i', '--input_dir', default="./input_real_2", help='Input directory')
    parser.add_argument('-o', '--output_dir', default="./output", help='Output directory')
    parser.add_argument('-c', '--combination', default=5, type=int, help='Selection of variables')
    parser.add_argument('-f', '--n_folds', default=2, type=int, help='Number of CV folds')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('-e', '--n_epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay for optimizer')
    args = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    n_folds = args.n_folds
    batch_size = args.batch_size
    # Define hyperparameter grid for the NN
    param_grid = {
        'hidden_size': [16, 32, 64],
        'learning_rate': [0.001, 0.0005],
        'num_epochs': [50, 100]
    }

    # Variable combinations (as before)
    comb_1 = ['RIDAGEYR.x', 'BMI']
    comb_2 = comb_1
    comb_3 = comb_1
    comb_4 = comb_1
    comb_5 = comb_1
    comb_6 = ['RIDAGEYR.x', 'BMI']
    comb_7 = comb_1
    all_combinations = [comb_1, comb_2, comb_3, comb_4, comb_5, comb_6, comb_7]
    selected_combination = all_combinations[int(args.combination)]
    target = 'TAC2'

    # Loop over groups (for example: 'tres', 'cinco', 'ocho')
    for var_to_group in ['tres', 'cinco', 'ocho']:
        df = pd.DataFrame()
        onlyfiles = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
        for f in onlyfiles:
            file = os.path.join(args.input_dir, f)
            print(f"Processing file: {file}")
            print(selected_combination)
            df = pd.read_csv(file)
            dataset_name = os.path.splitext(os.path.basename(file))[0]
            df.reset_index(drop=True, inplace=True)
            index_0 = df[df[var_to_group] == 0].copy()
            index_1 = df[df[var_to_group] == 1].copy()
            sceminario = f'{dataset_name}'
            op = os.path.join(args.output_dir, var_to_group, sceminario)
            if not os.path.exists(op):
                os.makedirs(op)
            
            # Prepare DataFrames for healthy and diseased data
            data, data_0, data_1 = load_data(file, selected_combination, target, var_to_group)
            print(f"Data_0 shape: {data_0['x'].shape} - Data_1 shape: {data_1['x'].shape}")

            # Use grid search with NN for each group
            model_0, score_0, params_0 = grid_search_nn(data_0, n_folds, batch_size, param_grid)
            model_1, score_1, params_1 = grid_search_nn(data_1, n_folds, batch_size, param_grid)

            # Print some sample predictions from model_0
            mean_preds_0 = compute_mean(data_0, model_0)
            for i in range(min(5, len(data_0['y']))):
                print(f"Target {data_0['y'][i]} - Prediction {mean_preds_0[i]}")
            
            # Delete all files in the specified folder before saving new plots
            for file_del in glob.glob(os.path.join(op, "*")):
                os.remove(file_del)

            residues_0 = compute_residues(data_0, model_0)
            plt.hist(residues_0, bins=50, alpha=0.5, label='Healthy')
            residues_1 = compute_residues(data_1, model_1)
            plt.hist(residues_1, bins=50, alpha=0.5, label='Diseased')
            plt.legend()
            plt.title('Residue Distributions')
            plt.xlabel('Residue')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(args.output_dir, var_to_group, sceminario, f'Residue Distributions_{dataset_name}_{var_to_group}.png'))
            plt.clf()
            print(f'Mean residues class_0: {np.mean(residues_0)}')
            print(f'Mean residues class_1: {np.mean(residues_1)}')
            print(f'Residues class_0 shape: {residues_0.shape}')
            print(f'Residues class_1 shape: {residues_1.shape}')
            
            # Process residues (train NN on squared errors) for uncertainty estimation
            data_0_r = {
                'x': data_0['x'],
                'y': residues_0,
                'w': data_0['w'],
            }
            model_0_r, score_0_r, _ = grid_search_nn(data_0_r, n_folds, batch_size, param_grid)
            data_1_r = {
                'x': data_1['x'],
                'y': residues_1,
                'w': data_1['w'],
            }
            model_1_r, score_1_r, _ = grid_search_nn(data_1_r, n_folds, batch_size, param_grid)
            
            print(f"Data_0_r shape: {data_0_r['x'].shape} - Data_1_r shape: {data_1_r['x'].shape}")
            mean_0 = compute_mean(data, model_0)
            mean_1 = compute_mean(data, model_1)
            std_0 = compute_std(data, model_0_r)
            std_1 = compute_std(data, model_1_r)
            
            # Compute 'a' and 'b' based on the condition
            a_predicted = np.where(mean_1 > mean_0,
                                   (mean_1 - mean_0) / std_1,
                                   (mean_0 - mean_1) / std_0)
            b_predicted = np.where(mean_1 > mean_0,
                                   std_0 / std_1,
                                   std_1 / std_0)
            
            area = roc(a_predicted[0], b_predicted[0], residues_0, residues_1)
            print(f'AUC: {area}')
            
            # Calculate the conditional ROC values
            def G_D(y):
                return norm.cdf(y)
            p = np.linspace(0.001, 0.999, 100)
            roc_predicted = []
            auc_predicted = []
            for i in range(len(a_predicted)):
                roc_values_predicted = 1 - G_D(norm.ppf(1-p) * b_predicted[i] - a_predicted[i])
                roc_predicted.append(roc_values_predicted)
                plt.plot(p, np.array(roc_values_predicted))
            plt.xlabel('1 - Specificity')
            plt.ylabel('Sensitivity')
            plt.title(f'Conditional ROC Curves for predicted {sceminario}_{dataset_name}')
            plt.savefig(os.path.join(args.output_dir, var_to_group, sceminario, f'roc_plotNN_predicted_{dataset_name}_{var_to_group}.png'))
            plt.clf()
            
            for i in range(len(a_predicted)):
                roc_values_predicted = 1 - G_D(norm.ppf(1-p) * b_predicted[i] - a_predicted[i])
                auc = simpson(roc_values_predicted, p)
                # Alternatively, you could compute AUC per point as in your original code
                auc_predicted.append(auc)
                plt.hist(np.array(auc_predicted), bins=20, alpha=0.7)
            plt.title(f'AUC {sceminario}_{dataset_name}')
            plt.savefig(os.path.join(args.output_dir, var_to_group, sceminario, f'AUC_plotNN_predicted_{dataset_name}_{var_to_group}.png'))
            plt.clf()
            
            grid_size = int(np.sqrt(len(a_predicted)))
            v1 = np.linspace(int(min(df['RIDAGEYR.x'])), int(max(df['RIDAGEYR.x'])), num=grid_size)
            v2 = np.linspace(int(min(df['BMI'])), int(max(df['BMI'])), num=grid_size)
            v1_grid, v2_grid = np.meshgrid(v1, v2)
            auc_grid = np.zeros((len(v2), len(v1)))
            for i in range(len(v2)):
                for j in range(len(v1)):
                    auc_grid[i, j] = roc(a_predicted[i * len(v1) + j], b_predicted[i * len(v1) + j], residues_0, residues_1)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(v1_grid, v2_grid, auc_grid, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_xlabel('RIDAGEYR')
            ax.set_ylabel('BMI')
            ax.set_zlabel('AUC')
            plt.savefig(os.path.join(args.output_dir, var_to_group, sceminario, f'auc_3d_plotNN_predicted_{dataset_name}_{var_to_group}.png'))
            plt.clf()
            
            df['1 - CROC_model'] = auc_predicted
            plt.figure(figsize=(12, 6))
            plt.scatter(df['RIDAGEYR.x'], df['1 - CROC_model'], alpha=0.2, edgecolors='none')
            plt.title('Scatter Plot of 1 - CROC_model vs. RIDAGEYR.x')
            plt.xlabel('RIDAGEYR.x')
            plt.ylabel('1 - CROC_model$(AUC,1,1)$')
            plt.grid(True)
            plt.savefig(os.path.join(args.output_dir, var_to_group, sceminario, f'auc_2d_plotNN_predicted_{dataset_name}_{var_to_group}.png'))
            plt.clf()

# %%
