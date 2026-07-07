import os
import pandas as pd
import numpy as np
import random
import argparse
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from data_simulation_reg import load_data,create_dict_2,create_dict
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from statsmodels.distributions.empirical_distribution import ECDF
import glob
import re
import matplotlib.pyplot as plt

# Definición del MLP parametrizable para regresión
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout):
        """
        Parámetros:
            input_dim: número de variables de entrada.
            hidden_layers: lista con el número de unidades en cada capa oculta.
            dropout: tasa de dropout para cada capa oculta.
        """
        super(MLP, self).__init__()
        layers = []
        in_features = input_dim

        # Construir cada capa oculta
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_units

        # Capa de salida: 1 neurona sin activación para regresión
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, device, learning_rate, weight_decay, num_epochs=50, early_stop_patience=10):
    """
    Entrena el modelo usando MSELoss con sample weights y regularización L2 (weight decay).
    """
    # Usamos MSELoss sin reducción para luego incorporar los sample weights
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = np.inf
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            x_batch, y_batch, sample_weights = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            sample_weights = sample_weights.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch).view(-1, 1)  # tamaño (batch_size, 1)
            loss = criterion(outputs, y_batch.float().view(-1, 1))
            loss = (loss * sample_weights.view(-1, 1)).mean()  # Incorporamos el peso de cada observación
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validación
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_batch, sample_weights = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                sample_weights = sample_weights.to(device)
                outputs = model(x_batch).view(-1, 1)
                loss = criterion(outputs, y_batch.float().view(-1, 1))
                loss = (loss * sample_weights.view(-1, 1)).mean()
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Comprobación del early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0  # reiniciar contador
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping en epoch {epoch + 1} (sin mejora durante {early_stop_patience} épocas consecutivas)")
                break

    # Restaurar el mejor modelo obtenido en validación
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def evaluate_model(model, data_loader, device):
    """
    Evalúa el modelo y acumula las salidas, etiquetas y sample weights de cada observación.
    """
    model.eval()
    losses = []
    criterion = nn.MSELoss(reduction='none')
    all_outputs = []
    all_labels = []
    all_sample_weights = []
    with torch.no_grad():
        for batch in data_loader:
            x_batch, y_batch, sample_weights = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            sample_weights = sample_weights.to(device)
            outputs = model(x_batch).view(-1, 1)
            loss = criterion(outputs, y_batch.float().view(-1, 1))
            loss = (loss * sample_weights.view(-1, 1)).mean()
            losses.append(loss.item())
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_sample_weights.append(sample_weights.cpu().numpy())
    avg_loss = np.mean(losses)
    return avg_loss, np.concatenate(all_outputs), np.concatenate(all_labels), np.concatenate(all_sample_weights)


def compute_residues(data, model):
    device = next(model.parameters()).device
    input = torch.from_numpy(data['x']).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input).detach().cpu().numpy()
    return (output - data['y'])**2


def compute_mean(data, model):
    input = torch.from_numpy(data['x']).to(device)
    print('Input size {}'.format(input.shape))
    model.eval()
    with torch.no_grad():
        output = model(input).detach().cpu().numpy()
    return output
# NOT PQ NO LA MEDIA?


def compute_std(data, model):
    input = torch.from_numpy(data['x']).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input).detach().cpu().numpy()
    # output[output < 0] = 0.001
    #return np.sqrt(np.exp(output))
    output_safe = np.clip(output, 0, None)
    return np.sqrt(output_safe)




def roc(a, b, res_0, res_1):
    res_0 = res_0.squeeze()
    res_1 = res_1.squeeze()
    ecdf_0 = ECDF(res_0)
    ecdf_1 = ECDF(res_1)

    sample_edf = ecdf_0
    slope_changes = sorted(set(res_0))
    sample_edf_values_at_slope_changes = [sample_edf(item) for item in slope_changes]
    inverted_edf_0 = interp1d(sample_edf_values_at_slope_changes, slope_changes)

    p = np.linspace(0.001, 0.999, num=100)
    # aux = inverted_edf_0(1-p) * b - a
    # print('Aux: {}'.format(aux))
    r = 1 - ecdf_1(inverted_edf_0(1-p) * b - a)
    i1 = simpson(r, p)
    # plt.plot(p, r)
    # plt.title('AUC: {}'.format(i1))
    # plt.show()
    #
    # print('AUC: {}'.format(i1))
    return i1


def main(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    match config.get("combination", -1):
        case 0:
            combination = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBDSCHSI_43',
                           'LBXSTR_43', 'LBXSGL_43', 'RIAGENDR', 'LBXGH_39']
        case 1:
            combination = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS',
                           'BPXDI1', 'LBDSCHSI_43', 'LBXSTR_43', 'LBXSGL_43', 'RIAGENDR']
        case 3:
            combination = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBDSCHSI_43',
                           'LBXSTR_43', 'RIAGENDR', 'LBXGH_39']
        case 4:
            combination = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS',
                           'BPXDI1', 'LBDSCHSI_43', 'LBXSTR_43', 'RIAGENDR']
        case 5:
            combination = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS',
                           'BPXDI1', 'RIAGENDR']
        case 6:
            combination = ['x_D_5', 'x_D_6', 'x_D_7', 'x_D_8']   
        case 7:
            combination = ['BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'BPXDI1']
        case _:
            combination = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBDSCHSI_43',
                           'LBXSTR_43', 'LBXSGL_43', 'RIAGENDR', 'LBXGH_39']

    target = 'Y_generated'
    # ['ocho', 'cinco', 'tres']
    for var_to_group in [ 'mortstat']:
        
        onlyfiles = [f for f in os.listdir(config["input_dir"]) if os.path.isfile(os.path.join(config["input_dir"], f))]
        for f in onlyfiles:
            file = os.path.join(config["input_dir"], f)
            df = pd.read_csv(file)
            selected_combination = df.columns[df.columns.str.contains('x_D')].tolist()
            print('Running regression for file {}'.format(file))
            dataset_name = os.path.splitext(os.path.basename(file))[0]
            sceminario = f'{dataset_name}'
            index_0 = df[df['mortstat'] == 0].copy() # Adjust based on how you define groups
            index_1 = df[df['mortstat'] == 1].copy() # Adjust based on how you define groups
            media_verdadera_bar = index_0['True_Mean_Y']
            del index_0['True_Mean_Y']
            media_verdadera = index_1['True_Mean_Y']
            del index_1['True_Mean_Y']
            match = re.search(r'(scenario_[IVXLCDM\d]+)', file)
            match = re.match(r'^(scenario_\d+_\d+_\d+)_data$', dataset_name)

            if match:
                sceminario = match.group(1)
            print (sceminario)
            data, data_0, data_1 = load_data(file, selected_combination, target,var_to_group)
            output_folder = f'{args.output_file}/{var_to_group}/{sceminario}'
            op= f'{args.output_file}/{sceminario}'
            if not os.path.exists(op):
                 os.makedirs(op)
            for file1 in glob.glob(f'{op}/*'):
                os.remove(file1)
            
            output_file = f'{args.output_file}'
            model_0 = split_and_train(data_0, config)
            model_1 = split_and_train(data_1, config)
            
            #residues_0 = compute_residues(data_0, model_0)
            residues_0 = compute_residues(data_0, model_0)
            plt.hist(residues_0, bins=50, alpha=0.5, label='Healthy')
            #residues_1 = compute_residues(data_1, model_1)
            residues_1 = compute_residues(data_1, model_1)
            plt.hist(residues_1, bins=50, alpha=0.5, label='Diseased')


            plt.legend()
            plt.title('Residue Distributions')
            plt.xlabel('Residue')
            plt.ylabel('Frequency')
            plt.savefig(f'{output_file}/{sceminario}/Residue Distributions_{sceminario}.png')
            plt.clf()
            print(f'Mean residues class_0: {np.mean(residues_0)}')
            print(f'Mean residues class_1: {np.mean(residues_1)}')
            print(f'Residues class_0 shape: {residues_0.shape}')
            print(f'Residues class_1 shape: {residues_1.shape}')

            data_0_r = {
                'x': data_0['x'],
                'y': residues_0,
                'w': data_0['w'],
            }
            model_0_r = split_and_train(data_0_r, config)

            data_1_r = {
                'x': data_1['x'],
                'y': residues_1,
                'w': data_1['w'],
            }
            model_1_r = split_and_train(data_1_r, config)

            mean_0 = compute_mean(data_0, model_0)
            mean_1 = compute_mean(data_1, model_1)
            std_0 = compute_std(data_0_r, model_0_r)
            std_1 = compute_std(data_1_r, model_1_r)
        
            results_healthy = pd.DataFrame({
                'Real Mean': media_verdadera_bar,
                'Predicted Mean': mean_0.flatten(),
                'Real Std Dev': 0.5,
                'Predicted Std Dev': std_0.flatten()
            })

            results_diseased = pd.DataFrame({
                'Real Mean': media_verdadera,
                'Predicted Mean': mean_1.flatten(),
                'Real Std Dev': 1,
                'Predicted Std Dev': std_1.flatten()
            })

            plt.figure(figsize=(10, 6))

            # Plot True Mean of Y
            plt.scatter(index_0['x_D_1'], results_healthy['Real Mean'], color='blue', alpha=0.5, label='True Mean of Y', s=15)

            # Plot Generated Y
            plt.scatter(index_0['x_D_1'], data_0['y'], color='green', alpha=0.3, label='Generated Y', s=15)

            # Plot Predicted Y
            plt.scatter(index_0['x_D_1'], results_healthy['Predicted Mean'], color='red', alpha=0.5, label='Predicted Y', s=15)

            # Add plot details
            plt.xlabel('X')
            plt.ylabel('Y values')
            plt.title(f'Comparison of True Mean, Generated Y, and Predicted Y for Healthy for {sceminario}')
            plt.legend()
            plt.savefig(f'{output_file}/{sceminario}/healthy_comparison_plot_{sceminario}.png')
            plt.clf()  # Clear the plot to prevent overlap in subsequent loops


            # Ensure the data for X, True Mean, Generated Y, and Predicted Y are accessible in this section
            plt.figure(figsize=(10, 6))

            # Plot True Mean of Y
            plt.scatter(index_1['x_D_1'], results_diseased['Real Mean'], color='blue', alpha=0.5, label='True Mean of Y', s=15)

            # Plot Generated Y
            plt.scatter(index_1['x_D_1'], data_1['y'], color='green', alpha=0.3, label='Generated Y', s=15)

            # Plot Predicted Y
            plt.scatter(index_1['x_D_1'], results_diseased['Predicted Mean'], color='red', alpha=0.5, label='Predicted Y', s=15)

            # Add plot details
            plt.xlabel('X')
            plt.ylabel('Y values')
            plt.title(f'Comparison of True Mean, Generated Y, and Predicted Y for Diseased for {sceminario}')
            plt.legend()
            plt.savefig(f'{output_file}/{sceminario}/Diseased_comparison_plot_{sceminario}.png')
            plt.clf()  # Clear the plot to prevent overlap in subsequent loops



            # Plot Real Std Dev vs Predicted Std Dev for healthy data
            plt.figure(figsize=(10, 5))
            plt.scatter(results_healthy['Real Std Dev'], results_healthy['Predicted Std Dev'], alpha=0.5, label='Std Dev')
            plt.xlabel('Real Std Dev')
            plt.ylabel('Predicted Std Dev')
            plt.title(f'Real vs Predicted Std Dev for Healthy {sceminario}')
            plt.plot([results_healthy['Real Std Dev'].min(), results_healthy['Real Std Dev'].max()], 
                    [results_healthy['Real Std Dev'].min(), results_healthy['Real Std Dev'].max()], 'r--')  # Diagonal line
            plt.legend()
            plt.savefig(f'{output_file}/{sceminario}/healthy_std_dev_comparison_plot_{sceminario}.png')
            plt.clf()

            # Plot Real Std Dev vs Predicted Std Dev for diseased data
            plt.figure(figsize=(10, 5))
            plt.scatter(results_diseased['Real Std Dev'], results_diseased['Predicted Std Dev'], alpha=0.5, label='Std Dev')
            plt.xlabel('Real Std Dev')
            plt.ylabel('Predicted Std Dev')
            plt.title(f'Real vs Predicted Std Dev for Diseased {sceminario}')
            plt.plot([results_diseased['Real Std Dev'].min(), results_diseased['Real Std Dev'].max()], 
                    [results_diseased['Real Std Dev'].min(), results_diseased['Real Std Dev'].max()], 'r--')  # Diagonal line
            plt.legend()
            plt.savefig(f'{output_file}/{sceminario}/diseased_std_dev_comparison_plot_{sceminario}.png')
            plt.clf()

            
                        # Compute 'a' and 'b' based on the condition
            epsilon = 1e-8  # a small constant to avoid zero division

            a_predicted = np.where(
                mean_1.flatten() > mean_0.flatten(),
                (mean_1.flatten() - mean_0.flatten()) / (std_1.flatten() + epsilon),
                (mean_0.flatten() - mean_1.flatten()) / (std_0.flatten() + epsilon)
                )

            b_predicted = np.where(
            mean_1.flatten() > mean_0.flatten(),
            std_0.flatten() / (std_1.flatten() + epsilon),
            std_1.flatten() / (std_0.flatten() + epsilon)
                            )


            a_real = np.where(results_diseased['Real Mean'].values > results_healthy['Real Mean'].values,
                            (results_diseased['Real Mean'].values - results_healthy['Real Mean'].values) / results_diseased['Real Std Dev'].values,
                            (results_healthy['Real Mean'].values - results_diseased['Real Mean'].values) / results_healthy['Real Std Dev'].values)

            b_real = np.where(results_diseased['Real Mean'].values > results_healthy['Real Mean'].values,
                            results_healthy['Real Std Dev'].values / results_diseased['Real Std Dev'].values,
                            results_diseased['Real Std Dev'].values / results_healthy['Real Std Dev'].values)

            area = roc(a_predicted[0], b_predicted[0], residues_0, residues_1)
            print(f'AUC: {area}')
                    # Calculate the conditional ROC values
            p = np.linspace(0.001, 0.999, 100)
            roc_predicted = []
            from scipy.stats import norm

            def G_D(y):
                return norm.cdf(y)

            for i in range(len(a_predicted)):
                roc_values_predicted =  roc(a_predicted[i], b_predicted[i], residues_0, residues_1)
                roc_values_predicted = 1 - G_D(norm.ppf(1-p) * b_predicted[i] - a_predicted[i])
                roc_predicted.append(roc_values_predicted)
                plt.plot(p, np.array(roc_values_predicted))

            plt.xlabel('1 - Specificity')
            plt.ylabel('Sensitivity')
            plt.title(f'Conditional ROC Curves for predicted {sceminario}')
            plt.savefig(f'{output_file}/{sceminario}/roc_plotnn_predicted_{sceminario}.png')

            plt.clf()

            roc_real = []
            p = np.linspace(0.001, 0.999, 100)

            for i in range(len(a_real)):
                roc_values_real= roc(a_real[i], b_real[i ], residues_0, residues_1)
                roc_values_real= 1 - G_D(norm.ppf(1-p) * b_real[i] - a_real[i])
                roc_real.append(roc_values_real)

                plt.plot(p, np.array(roc_values_real))

            plt.xlabel('1 - Specificity')
            plt.ylabel('Sensitivity')
            plt.title(f'Conditional ROC Curves for real {sceminario}')
            plt.savefig(f'{output_file}/{sceminario}/roc_plotnn_real_{sceminario}.png')

            plt.clf()
                        # Assuming data_0 and data_1 have 'y' values as the real target values
            real_output_0 = media_verdadera_bar.values  # Real values for healthy
            real_output_1 = media_verdadera.values      # Real values for diseased
    
            real_output_0 = real_output_0.reshape(-1, 1)
            real_output_1 = real_output_1.reshape(-1, 1)

            residues_0_real = (data_0['y'] - real_output_0) ** 2
            residues_1_real = (data_1['y'] - real_output_1) ** 2
            residues_0_real = residues_0_real.ravel().reshape(-1, 1)  # Flatten and reshape to single column
            residues_1_real = residues_1_real.ravel().reshape(-1, 1)  # Flatten and reshape to single column


            a = (mean_1 - mean_0) / std_1
            b = std_0 / std_1

            area = roc(a[2], b[2], residues_0, residues_1)
            print('auc {}'.format(area))

            v1 = np.linspace(25, 80, num=51)
            # v1 = np.linspace(25, 75, num=51)
            # v1 = np.linspace(80, 115, num=71)
            # v2 = np.linspace(4.5, 7.0, num=51)
            v2 = np.linspace(90, 140, num=51)
            # v1 = np.linspace(80, 120, num=81)
            [v1_grid, v2_grid] = np.meshgrid(v1, v2)
            input_grid = np.zeros((len(v2), len(v1)))
            for i in range(len(v2)):
                 for j in range(len(v1)):
                     input_grid[i, j] = roc(a_predicted[i * len(v1) + j], b_predicted[i * len(v1) + j], residues_0, residues_1)
            # Plotting
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(v1_grid, v2_grid, input_grid, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_xlabel('RIDAGEYR.x')
            ax.set_ylabel('BMXWAIST')
            ax.set_zlabel('AUC')
            plt.savefig(f'{output_file}/{sceminario}/auc_3d_plot_predicted_{sceminario}.png')
            plt.clf()
            
            # Assuming data_0 and data_1 have 'y' values as the real target values
            real_output_0 = media_verdadera_bar.values  # Real values for healthy
            real_output_1 = media_verdadera.values      # Real values for diseased
    
            real_output_0 = real_output_0.reshape(-1, 1)
            real_output_1 = real_output_1.reshape(-1, 1)

            residues_0_real = (data_0['y'] - real_output_0) ** 2
            residues_1_real = (data_1['y'] - real_output_1) ** 2
            residues_0_real = residues_0_real.ravel().reshape(-1, 1)  # Flatten and reshape to single column
            residues_1_real = residues_1_real.ravel().reshape(-1, 1)  # Flatten and reshape to single column
            auc_grid = np.zeros((len(v2), len(v1)))


            for i in range(len(v2)):
                for j in range(len(v1)):
                    auc_grid[i, j] = roc(a_real[i * len(v1) + j], b_real[i * len(v1) + j], residues_0_real, residues_1_real)
            # Plotting
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(v1_grid, v2_grid, auc_grid, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_xlabel('RIDAGEYR.x')
            ax.set_ylabel('BMXWAIST')
            ax.set_zlabel('AUC')
            plt.savefig(f'{args.output_file}/{sceminario}/auc_3d_plot_real_{sceminario}.png')
            plt.clf()
            # Calculate Euclidean Distance between predicted and real ROC curves for each point
            distances = [np.linalg.norm(np.array(roc_pred) - np.array(roc_real_)) for roc_pred, roc_real_ in zip(roc_predicted, roc_real)]
            print(f"Euclidean distances between predicted and real ROC curves: {distances}")

            distances_df = pd.DataFrame(distances, columns=['Euclidean Distance'])
            distances_filename = f'{args.output_file}/{sceminario}/euclidean_distances{sceminario}.csv'
            distances_df.to_csv(distances_filename, index_label="Object Index")

            # Calculate Mean Squared Error between predicted and real ROC curves

            # Assuming roc_predicted and roc_real are lists of arrays, where each array represents an ROC curve
            mse_values = []
            for predicted, real in zip(roc_predicted, roc_real):
                # Calculate Mean Squared Error between predicted and real ROC values
                mse = mean_squared_error(real, predicted)
                mse_values.append(mse)

            # Convert MSE values to a DataFrame and save to CSV
            mse_df = pd.DataFrame(mse_values, columns=['Mean Squared Error'])
            mse_filename = f'{args.output_file}/{sceminario}/roc_mse_values_{sceminario}.csv'
            mse_df.to_csv(mse_filename, index_label="Object Index")
            print(len(roc_predicted))
            print(len(roc_real))

            # Plot MSE values
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(mse_values)), mse_values, marker='o', linestyle='-', color='b', label='MSE between Predicted and Real ROC Curves')
            plt.xlabel('Object Index')
            plt.ylabel('Mean Squared Error')
            plt.title(f'MSE between Predicted and Real ROC Curves for {sceminario}')
            plt.legend()
            plt.savefig(f'{output_file}/{sceminario}/roc_mse_plot_{sceminario}.png')
            plt.clf()

            # Print MSE values
            print(f"Mean Squared Errors between predicted and real ROC curves: {mse_values}")


def split_and_train(data, config):
    # DIVIDIR ESTO -> CREAR NUEVA FUNC Y LLAMARLA
    X, y, sample_weights = data['x'], data['y'], data['w']
    d = X.shape[1]

    # Dividir en train+validation (80%) y test (20%)
    X_trainval, X_test, y_trainval, y_test, sw_trainval, sw_test = train_test_split(
        X, y, sample_weights, test_size=0.20, random_state=42
    )

    # Dentro del 80%: 64% train y 16% validación (respectivamente, 80%*0.8 y 80%*0.2)
    X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
        X_trainval, y_trainval, sw_trainval, test_size=0.20, random_state=42
    )

    # Convertir a tensores (para regresión, las etiquetas son float)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    sw_train_tensor = torch.tensor(sw_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    sw_val_tensor = torch.tensor(sw_val, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    sw_test_tensor = torch.tensor(sw_test, dtype=torch.float32)

    # Crear datasets para train, validación y test
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, sw_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor, sw_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, sw_test_tensor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validación cruzada en el conjunto de entrenamiento (train) con k folds
    k_folds = config["k_folds"]
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    indices = np.arange(len(train_dataset))
    fold_results = []

    print("----- VALIDACIÓN CRUZADA -----")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\nFold {fold + 1}/{k_folds}")
        fold_train_subset = Subset(train_dataset, train_idx)
        fold_val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(fold_train_subset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(fold_val_subset, batch_size=config["batch_size"], shuffle=False)

        # Inicializar el modelo para este fold
        model = MLP(input_dim=d, hidden_layers=config["hidden_layers"], dropout=config["dropout"])
        model.to(device)

        model = train_model(model, train_loader, val_loader, device,
                            learning_rate=config["learning_rate"],
                            weight_decay=config["weight_decay"],
                            num_epochs=config["num_epochs"],
                            early_stop_patience=config["early_stop_patience"])

        # Evaluar en el fold de validación
        val_loss, _, _, _ = evaluate_model(model, val_loader, device)
        print(f"Fold {fold + 1} - Validation Loss: {val_loss:.4f}")
        fold_results.append(val_loss)

    print("\nResultados de la validación cruzada:")
    print(f"Promedio Loss: {np.mean(fold_results):.4f} | Std: {np.std(fold_results):.4f}")

    # Entrenamiento final con Train + Validation (80% de los datos)
    print("\n----- ENTRENAMIENTO FINAL (Train+Val) -----")
    X_trainval_tensor = torch.cat([X_train_tensor, X_val_tensor])
    y_trainval_tensor = torch.cat([y_train_tensor, y_val_tensor])
    sw_trainval_tensor = torch.cat([sw_train_tensor, sw_val_tensor])
    trainval_dataset = TensorDataset(X_trainval_tensor, y_trainval_tensor, sw_trainval_tensor)

    trainval_loader = DataLoader(trainval_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    final_model = MLP(input_dim=d, hidden_layers=config["hidden_layers"], dropout=config["dropout"])
    final_model.to(device)

    # Entrenamiento final usando todos los datos de train y validación
    final_model = train_model(final_model, trainval_loader, trainval_loader, device,
                            learning_rate=config["learning_rate"],
                            weight_decay=config["weight_decay"],
                            num_epochs=config["num_epochs"],
                            early_stop_patience=config["early_stop_patience"])

    # # Evaluar en el conjunto test y obtener las salidas, etiquetas y sample weights
    # test_loss, test_outputs, test_labels, test_sample_weights = evaluate_model(final_model, test_loader, device)
    # print(f"\nTest Loss: {test_loss:.4f}")
    #
    # # Calcular métricas de regresión utilizando los sample weights
    # # Se calculan las métricas ponderando cada error individual
    # mse = mean_squared_error(test_labels, test_outputs, sample_weight=test_sample_weights)
    # rmse = np.sqrt(mse)
    # mae = mean_absolute_error(test_labels, test_outputs, sample_weight=test_sample_weights)
    # r2 = r2_score(test_labels, test_outputs, sample_weight=test_sample_weights)
    #
    # print("\n----- Métricas finales en el conjunto test (ponderadas) -----")
    # print(f"RMSE: {rmse:.4f}")
    # print(f"MAE: {mae:.4f}")
    # print(f"R^2: {r2:.4f}\n\n")
    #
    # result = {
    #     "RMSE": rmse,
    #     "MAE": mae,
    #     "R2": r2,
    #     "Test_Loss": test_loss,
    # }
    #
    # return final_model, result
    return final_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Programa de regresión con MLP')
    parser.add_argument('-i', '--input_dir', default="./data/reg_real", help='Directorio de entrada')
    parser.add_argument('-o', '--output_file', default="./output/sim1_mlp_regression.csv", help='Archivo de salida')
    parser.add_argument('-c', '--combination', type=int, default=6, help='Selección de variables')
    parser.add_argument('-f', '--k_folds', type=int, default=5, help='Número de folds para cross validation')
    parser.add_argument('-e', '--num_epochs', type=int, default=50, help='Número de épocas')
    parser.add_argument('-es', '--early_stop_patience', type=int, default=10, help='Early stopping')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Tamaño de batch')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Tasa de aprendizaje')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4, help='Regularización L2 (weight decay)')
    parser.add_argument('-dr', '--dropout', type=float, default=0.2, help='Tasa de dropout')
    parser.add_argument('-hl', '--hidden_layers', type=ast.literal_eval, default="[64, 32]",
                        help="Capas ocultas como '[64, 32]'")
    args = parser.parse_args()

    config = {
        "input_dir": args.input_dir,
        "output_file": args.output_file,
        "combination": args.combination,
        "hidden_layers": args.hidden_layers,  # Lista con el tamaño de cada capa oculta
        "dropout": args.dropout,  # Tasa de dropout
        "learning_rate": args.learning_rate,  # Tasa de aprendizaje
        "weight_decay": args.weight_decay,  # Regularización L2 (weight decay)
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "k_folds": args.k_folds,  # Número de folds para cross validation
        "early_stop_patience": args.early_stop_patience  # Número de épocas sin mejora para early stopping
    }

main(config)

#%%
