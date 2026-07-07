#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
from sklearn.model_selection import GridSearchCV

def compute_residues(data, model):
    input_data = data['x']
    output = model.predict(input_data).ravel() 
  
    y_true = np.array(data['y']).ravel()  
    return  (y_true - output)**2 

def compute_mean(data, model):
    input_data = data['x']
    output = model.predict(input_data)
    return output

def compute_std(data, model):
    input_data = data['x']
    output = model.predict(input_data)
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
    inverted_edf_0 = interp1d(sample_edf_values_at_slope_changes, slope_changes)

    p = np.linspace(0.001, 0.999, num=100)
    r = 1 - ecdf_1(inverted_edf_0(1-p) * b - a)
    i1 = simpson(r, p)
    return i1



def cv_loop(data, n_folds, n_estimators=100, random_state=0):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4]    # Minimum number of samples required at each leaf node
    }
    
    # Initialize the base model
    base_model = RandomForestRegressor(random_state=random_state)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',  # Use negative MSE for scoring
        cv=n_folds,                        # Use the same number of folds as in KFold
        n_jobs=-1,                         # Use all available CPU cores
        verbose=1                          # Print progress
    )
    
    # Fit GridSearchCV on the training data
    grid_search.fit(data['x'], data['y'].ravel()  )
    
    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert negative MSE back to positive
    
    
    return best_model, best_score
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i', '--input_dir', default="./input_real_2", help='Input_directory')
    parser.add_argument('-o', '--output_dir', default="./output", help='Output_directory')
    parser.add_argument('-c', '--combination', default=5, type=int, help='Selection of variables')
    parser.add_argument('-f', '--n_folds', default=2, type=int, help='Number of cv folders')
    parser.add_argument('-e', '--n_estimators', default=100, type=int, help='Number of estimators for Random Forest')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    n_folds = args.n_folds
    n_estimators = args.n_estimators
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

    for var_to_group in ['tres','cinco','ocho']:
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

            op= f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}'
            if not os.path.exists(op):
                os.makedirs(op)
            # Prepare DataFrames for healthy and diseased data
            data, data_0, data_1 = load_data(file, selected_combination,target ,var_to_group)
            print(f"Data_0 shape: {data_0['x'].shape} - Data_1 shape: {data_1['x'].shape}")

            model_0, metrics_0 = cv_loop(data_0, n_folds, n_estimators)
            model_1, metrics_1 = cv_loop(data_1, n_folds, n_estimators)

            output = model_0.predict(data_0['x'])
            for i in range(len(data_0['y'])):
                print('Target {} - output {}'.format(data['y'][i], output[i]))
            
            # Delete all files in the specified folder before saving the new plot
            output_folder = f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}'
            for file in glob.glob(f'{output_folder}/*'):
                os.remove(file)

            residues_0 = compute_residues(data_0, model_0)
            plt.hist(residues_0, bins=50, alpha=0.5, label='Healthy')
            residues_1 = compute_residues(data_1, model_1)
            plt.hist(residues_1, bins=50, alpha=0.5, label='Diseased')

            plt.legend()
            plt.title('Residue Distributions')
            plt.xlabel('Residue')
            plt.ylabel('Frequency')
            plt.savefig(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/Residue Distributions_{dataset_name}_{var_to_group}.png')
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
            model_0_r, metrics_0_r = cv_loop(data_0_r, n_folds, n_estimators)

            data_1_r = {
                'x': data_1['x'],
                'y':residues_1 ,
                'w': data_1['w'],
            }
            model_1_r, metrics_1_r = cv_loop(data_1_r, n_folds, n_estimators)
            print(f"Data_0 shape: {data_0_r['x'].shape} - Data_1 shape: {data_1_r['x'].shape}")
            mean_0 = compute_mean(data, model_0)
            mean_1 = compute_mean(data, model_1)
            std_0 = compute_std(data, model_0_r)
            std_1 = compute_std(data, model_1_r)



            # Compute 'a' and 'b' based on the condition
            epsilon = 1e-8
            a_predicted = np.where(mean_1 > mean_0,
                                (mean_1 - mean_0) / (std_1 + epsilon),
                                (mean_0 - mean_1) / (std_0 + epsilon))

            b_predicted = np.where(mean_1 > mean_0,
                                std_0 / (std_1 + epsilon),
                                std_1 / (std_0 + epsilon))

               
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
                #roc_values_predicted= roc(a_predicted[i ], b_predicted[i], residues_0, residues_1)
                roc_predicted.append(roc_values_predicted)
                plt.plot(p, np.array(roc_values_predicted))

            plt.xlabel('1 - Specificity')    
            plt.ylabel('Sensitivity')
            plt.title(f'Conditional ROC Curves for predicted {sceminario}_{dataset_name}')
            plt.savefig(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/roc_plotnn_predicted_{dataset_name}_{var_to_group}.png')

            plt.clf()
            
            for i in range(len(a_predicted)):
                roc_values_predicted = 1 - G_D(norm.ppf(1-p) * b_predicted[i] - a_predicted[i])
                auc = simpson(roc_values_predicted, p)
                auc= roc(a_predicted[i ], b_predicted[i], residues_0, residues_1)
                auc_predicted.append(auc)
                plt.hist(np.array(auc_predicted))

            plt.title(f'AUC {sceminario}_{dataset_name}')
            plt.savefig(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/AUC_plotnn_predicted_{dataset_name}_{var_to_group}.png')

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
            plt.savefig(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/auc_3d_plot_predicted_{dataset_name}_{var_to_group}.png')
            plt.clf()

            df['1 - CROC_model'] = auc_predicted

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.scatter(df['RIDAGEYR.x'], df['1 - CROC_model'], alpha=0.2, edgecolors='none')
            plt.title('Scatter Plot of 1 - CROC_model vs. RIDAGEYR.x')
            plt.xlabel('RIDAGEYR.x')
            plt.ylabel('1 - CROC_model$(AUC,1,1)')
            plt.grid(True)
            plt.savefig(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/auc_2d_plot_predicted_{dataset_name}_{var_to_group}.png')
            plt.clf()






