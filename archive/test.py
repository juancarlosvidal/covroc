#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from statsmodels.distributions.empirical_distribution import ECDF
from train import cv_loop
from data import load_data, CustomDataset, create_dict_2
import random
import os 
from scipy.stats import norm
import re
from scipy.stats import skewnorm, t
import glob
from sklearn.metrics import mean_squared_error


def compute_residues(data, model):
    input_data = torch.from_numpy(data['x'])
    model.eval()
    with torch.no_grad():
        output = model(input_data).detach().cpu().numpy()
    return  (data['y'] - output)**2 

def compute_mean(data, model):
    input_data = torch.from_numpy(data['x'])
    model.eval()
    with torch.no_grad():
        output = model(input_data).detach().cpu().numpy()
    return output

def compute_std(data, model):
    input_data = torch.from_numpy(data['x'])
    model.eval()
    with torch.no_grad():
        output = model(input_data).detach().cpu().numpy()
    output = np.where(output <=0, 0,np.sqrt( output))

    return output

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


'''
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i', '--input_dir', default="./input_real_2", help='Input_directory')
    parser.add_argument('-o', '--output_dir', default="./output", help='Output_directory')
    parser.add_argument('-c', '--combination', default=5, type=int, help='Selection of variables')
    parser.add_argument('-f', '--n_folds', default=2, type=int, help='Number of cv folders')
    parser.add_argument('-e', '--n_epochs', default=800, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    n_folds = args.n_folds
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = 0.001
    weight_decay = 0.05
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
    var_to_group = 'ocho'

    df = pd.DataFrame()
    onlyfiles = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    for f in onlyfiles:
        file = os.path.join(args.input_dir, f)
        print(f"Processing file: {file}")
        print (selected_combination)
        df = pd.read_csv(file)

        dataset_name = os.path.splitext(os.path.basename(file))[0]

        df.reset_index(drop=True, inplace=True)
        index_0 = df[df[var_to_group] == 0].copy() # Adjust based on how you define groups
        index_1 = df[df[var_to_group] == 1].copy() # Adjust based on how you define groups

        sceminario = f'{dataset_name}'

        op= f'{args.output_dir}/{sceminario}'
        if not os.path.exists(op):
            os.makedirs(op)
        # Prepare DataFrames for healthy and diseased data
        data, data_0, data_1 = load_data(file, selected_combination,target ,var_to_group)
        print(f"Data_0 shape: {data_0['x'].shape} - Data_1 shape: {data_1['x'].shape}")

        model_0, metrics_0 = cv_loop(data_0, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=True)
        model_1, metrics_1 = cv_loop(data_1, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=True)

        input = torch.from_numpy(data_0['x'])
        model_0.eval()
        with torch.no_grad():
            output = model_0(input).detach().cpu().numpy()
        for i in range(len(data_0['y'])):
            print('Target {} - output {}'.format(data['y'][i], output[i]))
        
                # Delete all files in the specified folder before saving the new plot
        output_folder = f'{args.output_dir}/{sceminario}'
        for file in glob.glob(f'{output_folder}/*'):
            os.remove(file)

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
        plt.savefig(f'{args.output_dir}/{sceminario}/Residue Distributions_{dataset_name}_{var_to_group}.png')
        plt.clf()
        print(f'Mean residues class_0: {np.mean(residues_0)}')
        print(f'Mean residues class_1: {np.mean(residues_1)}')
        print(f'Residues class_0 shape: {residues_0.shape}')
        print(f'Residues class_1 shape: {residues_1.shape}')

        # Process residues
        data_0_r = {
            'x': data_0['x'],
            'y': residues_0 ,
            'w': data_0['w'],
        }
        model_0_r, metrics_0_r = cv_loop(data_0_r, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=True)

        data_1_r = {
            'x': data_1['x'],
            'y':residues_1 ,
            'w': data_1['w'],
        }
        model_1_r, metrics_1_r = cv_loop(data_1_r, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=True)
        print(f"Data_0 shape: {data_0_r['x'].shape} - Data_1 shape: {data_1_r['x'].shape}")
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
        plt.savefig(f'{args.output_dir}/{sceminario}/roc_plotnn_predicted_{dataset_name}_{var_to_group}.png')

        plt.clf()
        
        for i in range(len(a_predicted)):
            roc_values_predicted = 1 - G_D(norm.ppf(1-p) * b_predicted[i] - a_predicted[i])
            auc = simpson(roc_values_predicted, p)
            auc_predicted.append(auc)
            plt.hist(np.array(auc_predicted))

        plt.title(f'AUC {sceminario}_{dataset_name}')
        plt.savefig(f'{args.output_dir}/{sceminario}/AUC_plotnn_predicted_{dataset_name}_{var_to_group}.png')

        plt.clf()
        grid_size = int(np.sqrt(len(a_predicted)))

        v1 = np.linspace(int(min(df['RIDAGEYR.x'])), int(max(df['RIDAGEYR.x'])), num=grid_size)
        v2 = np.linspace(int(min(df['BMI'])),int( max(df['BMI'])), num=grid_size)
        [v1_grid, v2_grid] = np.meshgrid(v1, v2)
        
        auc_grid = np.zeros((len(v2), len(v1)))
        for i in range(len(v2)):
            for j in range(len(v1)):
                auc_grid[i, j] =  roc(a_predicted[i * len(v1) + j], b_predicted[i * len(v1) + j], residues_0, residues_1)
        # Plotting
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(v1_grid, v2_grid, auc_grid, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('RIDAGEYR')
        ax.set_ylabel('BMI')
        ax.set_zlabel('AUC')
        plt.savefig(f'{args.output_dir}/{sceminario}/auc_3d_plot_predicted_{dataset_name}_{var_to_group}.png')
        plt.clf()

# 0 healthy bar/ 1 diseased nor bar

# Example DataFrame, replace 'df' with your actual DataFrame containing the 'RIDAGEYR.x' and ensure it matches auc_predicted in order
ages = df['RIDAGEYR.x'].unique()
aucs = auc_predicted  # Ensure this is a list of AUC values corresponding to the ages in 'ages'

# Create a DataFrame from the ages and AUCs
data_to_plot = pd.DataFrame({
    'Age': ages,
    'AUC': aucs
})

# Sort by age if needed
data_to_plot.sort_values('Age', inplace=True)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(data_to_plot['Age'], data_to_plot['AUC'], marker='o', linestyle='-')
plt.title('AUC vs Age')
plt.xlabel('Age')
plt.ylabel('AUC')
plt.grid(True)
plt.savefig(f'{args.output_dir}/{sceminario}/AUC_vs_Age_plot_{dataset_name}_{var_to_group}.png')
plt.show()

'''
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i', '--input_dir', default="./input_real_2", help='Input_directory')
    parser.add_argument('-o', '--output_dir', default="./output", help='Output_directory')
    parser.add_argument('-c', '--combination', default=5, type=int, help='Selection of variables')
    parser.add_argument('-f', '--n_folds', default=2, type=int, help='Number of cv folders')
    parser.add_argument('-e', '--n_epochs', default=800, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    n_folds = args.n_folds
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = 0.001
    weight_decay = 0.05
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

    # Loop over the values of var_to_group
    # ['ocho', 'cinco', 'tres']
    for var_to_group in ['tres']:
        df = pd.DataFrame()
        onlyfiles = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
        for f in onlyfiles:
            file = os.path.join(args.input_dir, f)
            print(f"Processing file: {file}")
            print (selected_combination)
            df = pd.read_csv(file)

            dataset_name = os.path.splitext(os.path.basename(file))[0]

            df.reset_index(drop=True, inplace=True)
            index_0 = df[df[var_to_group] == 0].copy() # Adjust based on how you define groups
            index_1 = df[df[var_to_group] == 1].copy() # Adjust based on how you define groups

            sceminario = f'{dataset_name}'

            op= f'{args.output_dir}/{var_to_group}/{sceminario}'
            if not os.path.exists(op):
                os.makedirs(op)
            # Prepare DataFrames for healthy and diseased data
            data, data_0, data_1 = load_data(file, selected_combination,target ,var_to_group)
            print(f"Data_0 shape: {data_0['x'].shape} - Data_1 shape: {data_1['x'].shape}")

            model_0, metrics_0 = cv_loop(data_0, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=True)
            model_1, metrics_1 = cv_loop(data_1, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=True)

            input = torch.from_numpy(data_0['x'])
            model_0.eval()
            with torch.no_grad():
                output = model_0(input).detach().cpu().numpy()
            for i in range(len(data_0['y'])):
                print('Target {} - output {}'.format(data['y'][i], output[i]))
            
                    # Delete all files in the specified folder before saving the new plot
            output_folder = f'{args.output_dir}/{var_to_group}/{sceminario}'
            for file in glob.glob(f'{output_folder}/*'):
                os.remove(file)

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
            plt.savefig(f'{args.output_dir}/{var_to_group}/{sceminario}/Residue Distributions_{dataset_name}_{var_to_group}.png')
            plt.clf()
            print(f'Mean residues class_0: {np.mean(residues_0)}')
            print(f'Mean residues class_1: {np.mean(residues_1)}')
            print(f'Residues class_0 shape: {residues_0.shape}')
            print(f'Residues class_1 shape: {residues_1.shape}')

            # Process residues
            data_0_r = {
                'x': data_0['x'],
                'y': residues_0 ,
                'w': data_0['w'],
            }
            model_0_r, metrics_0_r = cv_loop(data_0_r, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=True)

            data_1_r = {
                'x': data_1['x'],
                'y':residues_1 ,
                'w': data_1['w'],
            }
            model_1_r, metrics_1_r = cv_loop(data_1_r, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=True)
            print(f"Data_0 shape: {data_0_r['x'].shape} - Data_1 shape: {data_1_r['x'].shape}")
            mean_0 = compute_mean(data_0_r, model_0)
            mean_1 = compute_mean(data_1_r, model_1)
            std_0 = compute_std(data_0_r, model_0_r)
            std_1 = compute_std(data_1_r, model_1_r)

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
            plt.savefig(f'{args.output_dir}/{var_to_group}/{sceminario}/roc_plotnn_predicted_{dataset_name}_{var_to_group}.png')

            plt.clf()
            
            for i in range(len(a_predicted)):
                roc_values_predicted = 1 - G_D(norm.ppf(1-p) * b_predicted[i] - a_predicted[i])
                auc = simpson(roc_values_predicted, p)

                #auc = roc(a_predicted[i ], b_predicted[i ], residues_0, residues_1)
                auc_predicted.append(auc)
                plt.hist(np.array(auc_predicted))

            plt.title(f'AUC {sceminario}_{dataset_name}')
            plt.savefig(f'{args.output_dir}/{var_to_group}/{sceminario}/AUC_plotnn_predicted_{dataset_name}_{var_to_group}.png')

            plt.clf()
            grid_size = int(np.sqrt(len(a_predicted)))

            v1 = np.linspace(int(min(df['RIDAGEYR.x'])), int(max(df['RIDAGEYR.x'])), num=grid_size)
            v2 = np.linspace(int(min(df['BMI'])),int( max(df['BMI'])), num=grid_size)
            [v1_grid, v2_grid] = np.meshgrid(v1, v2)
            
            auc_grid = np.zeros((len(v2), len(v1)))
            for i in range(len(v2)):
                for j in range(len(v1)):
                    auc_grid[i, j] =  roc(a_predicted[i * len(v1) + j], b_predicted[i * len(v1) + j], residues_0, residues_1)
            # Plotting
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(v1_grid, v2_grid, auc_grid, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_xlabel('RIDAGEYR')
            ax.set_ylabel('BMI')
            ax.set_zlabel('AUC')
            plt.savefig(f'{args.output_dir}/{var_to_group}/{sceminario}/auc_3d_plot_predicted_{dataset_name}_{var_to_group}.png')
            plt.clf()

            df['1 - CROC_model'] = auc_predicted

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.scatter(df['RIDAGEYR.x'], df['1 - CROC_model'], alpha=0.2, edgecolors='none')
            plt.title('Scatter Plot of 1 - CROC_model vs. RIDAGEYR.x')
            plt.xlabel('RIDAGEYR.x')
            plt.ylabel('1 - CROC_model$(AUC,1,1)')
            plt.grid(True)
            plt.savefig(f'{args.output_dir}/{var_to_group}/{sceminario}/auc_2d_plot_predicted_{dataset_name}_{var_to_group}.png')
            plt.clf()




# %%
