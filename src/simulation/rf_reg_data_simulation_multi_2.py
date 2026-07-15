"""Random Forest counterpart to mlp_reg_data_simulation_multi.py, used in the paper's
simulation study as the ensemble-learning benchmark for the two-stage location-scale
aROC estimator: a RandomForestRegressor estimates mu_d(x) per group d in {0,1}
(cv_loop), a second forest is trained on the squared residuals to estimate sigma_d(x)^2,
and roc() integrates the resulting aROC(p|x) = 1 - Phi(b(x)*Phi^-1(1-p) - a(x)) using
the empirical residual CDFs. Relative to the FNN estimator, this benchmark is reported
in the paper to achieve higher finite-sample error and less stable estimates,
particularly under non-linear and interaction covariate effects.
"""
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
import time
from scipy.stats import norm
import glob
from data_simulation_reg import load_data,create_dict_2,create_dict
from sklearn.model_selection import GridSearchCV
import re
import true_dgp
import ground_truth_auc
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
    fixed_params = {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100}
    all_metrics = []
    all_models = []
    for train_idx, val_idx in kf.split(data['x']):
        x_train, y_train = data['x'][train_idx], data['y'][train_idx]
        x_val, y_val = data['x'][val_idx], data['y'][val_idx]
        model = RandomForestRegressor(random_state=random_state, **fixed_params)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        mse = mean_squared_error(y_val, preds)
        all_metrics.append(mse)
        all_models.append(model)
    best_idx = np.argmin(all_metrics)
    best_model = all_models[best_idx]
    best_score = all_metrics[best_idx]
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
    comb_1 = ['x_D_5', 'x_D_6', 'x_D_7', 'x_D_8']   
    comb_2 = comb_1
    comb_3 = comb_1
    comb_4 = comb_1
    comb_5 = comb_1
    comb_6 = ['x_D_5', 'x_D_6', 'x_D_7', 'x_D_8']   
    comb_7 = comb_1
    all_combinations = [comb_1, comb_2, comb_3, comb_4, comb_5, comb_6, comb_7]
    selected_combination = all_combinations[int(args.combination)]
    target = 'Y_generated'

    for var_to_group in ['mortstat']:
        df = pd.DataFrame()
        summary_rows = []  
        onlyfiles = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
        for f in onlyfiles:
            file = os.path.join(args.input_dir, f)
            print(f"Processing file: {file}")
            print (selected_combination)
            df = pd.read_csv(file)
            selected_combination = df.columns[df.columns.str.contains('x_D')].tolist()
            dataset_name = os.path.splitext(os.path.basename(file))[0]

            df.reset_index(drop=True, inplace=True)
            index_0 = df[df[var_to_group] == 0].copy() # Adjust based on how you define groups
            index_1 = df[df[var_to_group] == 1].copy() # Adjust based on how you define groups
            media_verdadera_bar = index_0['True_Mean_Y']
            del index_0['True_Mean_Y']
            media_verdadera = index_1['True_Mean_Y']
            del index_1['True_Mean_Y']
            match = re.search(r'(scenario_[IVXLCDM\d]+)', file)
            match = re.match(r'^(scenario_\d+_\d+_\d+)_data$', dataset_name)

            if match:
                sceminario = match.group(1)
            print (sceminario)
            scenario_num = int(re.match(r'^scenario_(\d+)_', f).group(1))
            ground_truth_cols = true_dgp.covariate_columns(scenario_num)
            X0_true = index_0[ground_truth_cols].to_numpy(dtype=float)
            X1_true = index_1[ground_truth_cols].to_numpy(dtype=float)

            op= f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}'
            if not os.path.exists(op):
                os.makedirs(op)
            # Prepare DataFrames for healthy and diseased data
            data, data_0, data_1 = load_data(file, selected_combination,target ,var_to_group)
            print(f"Data_0 shape: {data_0['x'].shape} - Data_1 shape: {data_1['x'].shape}")

            fit_seconds_total = 0.0
            t0 = time.perf_counter()
            model_0, metrics_0 = cv_loop(data_0, n_folds, n_estimators)
            model_1, metrics_1 = cv_loop(data_1, n_folds, n_estimators)
            fit_seconds_total += time.perf_counter() - t0

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
            t0 = time.perf_counter()
            model_0_r, metrics_0_r = cv_loop(data_0_r, n_folds, n_estimators)

            data_1_r = {
                'x': data_1['x'],
                'y':residues_1 ,
                'w': data_1['w'],
            }
            model_1_r, metrics_1_r = cv_loop(data_1_r, n_folds, n_estimators)
            fit_seconds_total += time.perf_counter() - t0
            print(f"Data_0 shape: {data_0_r['x'].shape} - Data_1 shape: {data_1_r['x'].shape}")
            mean_0 = compute_mean(data_0, model_0)
            mean_1 = compute_mean(data_1, model_1)
            std_0 = compute_std(data_0_r, model_0_r)
            std_1 = compute_std(data_1_r, model_1_r)


            # true_dgp.true_std returns a per-subject array (Scenario 3's std varies with
            # x; every other scenario's is constant, broadcast to the same shape) or None
            # for Scenario 7's healthy arm (skew-normal/t mixture, no closed-form std);
            # becomes NaN below, which is honest -- the ground-truth ROC/AUC comparison
            # further down goes through ground_truth_auc, which handles that scenario via
            # Monte Carlo instead of this column.
            real_std_0 = true_dgp.true_std(scenario_num, 0, X0_true)
            real_std_1 = true_dgp.true_std(scenario_num, 1, X1_true)

            results_healthy = pd.DataFrame({
                'Real Mean': media_verdadera_bar,
                'Predicted Mean': mean_0.flatten(),
                'Real Std Dev': real_std_0,
                'Predicted Std Dev': std_0.flatten()
            })

            results_diseased = pd.DataFrame({
                'Real Mean': media_verdadera,
                'Predicted Mean': mean_1.flatten(),
                'Real Std Dev': real_std_1,
                'Predicted Std Dev': std_1.flatten()
            })

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
            escaped_base = sceminario.replace('_', r'\_')  
            # now escaped_base == 'data\_1\_10\_20'

            # 2) Then add your suffix (also escaping its underscore)
            healthy_tex  = escaped_base + r'\_healthy'   # 'data\_1\_10\_20\_healthy'
            diseased_tex = escaped_base + r'\_diseased'  # 'data\_1\_10\_20\_diseased'

            # Mean-Function MSE: predicted mean vs the *true* mean (not the raw noisy Y,
            # which would inflate this by the irreducible noise variance). Std-Function
            # MSE: predicted std vs the true std (NaN for Scenario 7's healthy arm, which
            # has no closed-form std). Renamed from the old bare "MSE" and newly added,
            # respectively (Reviewer 1, Minor Concern 4 -- see mlp_reg_data_simulation_multi.py).
            mean_function_mse_0 = mean_squared_error(media_verdadera_bar.to_numpy().ravel(), mean_0.ravel())
            mean_function_mse_1 = mean_squared_error(media_verdadera.to_numpy().ravel(), mean_1.ravel())
            std_function_mse_0 = (mean_squared_error(real_std_0, std_0.ravel())
                                   if real_std_0 is not None else float('nan'))
            std_function_mse_1 = mean_squared_error(real_std_1, std_1.ravel())

            summary_rows.append({
            "Scenario":        healthy_tex,
            "Real Mean":       media_verdadera_bar.mean(),
            "Predicted Mean":  mean_0.mean(),
            "Real Std Dev":    real_std_0.mean() if real_std_0 is not None else float('nan'),
            "Predicted Std Dev": std_0.mean(),
            "Generated Y":     data_0['y'].mean(),
            "Residues":        residues_0.mean(),
            "Mean-Function MSE": mean_function_mse_0,
            "Std-Function MSE": std_function_mse_0
               })
            summary_rows.append({
            "Scenario":        diseased_tex,
            "Real Mean":       media_verdadera.mean(),
            "Predicted Mean":  mean_1.mean(),
            "Real Std Dev":    real_std_1.mean(),
            "Predicted Std Dev": std_1.mean(),
            "Generated Y":     data_1['y'].mean(),
            "Residues":        residues_1.mean(),
            "Mean-Function MSE": mean_function_mse_1,
            "Std-Function MSE": std_function_mse_1
            })

            mean_std_mse_df = pd.DataFrame([
                {'Scenario': sceminario, 'Group': 'healthy', 'Method': 'RF',
                 'Mean-Function MSE': mean_function_mse_0, 'Std-Function MSE': std_function_mse_0},
                {'Scenario': sceminario, 'Group': 'diseased', 'Method': 'RF',
                 'Mean-Function MSE': mean_function_mse_1, 'Std-Function MSE': std_function_mse_1},
            ])
            mean_std_mse_df.to_csv(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/mean_std_mse.csv', index=False)


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

            # Ground truth aROC(p|x): exact closed form for the eight Gaussian
            # scenarios, Monte Carlo (via the true DGP) for Scenario 7's non-Gaussian
            # healthy arm. RF previously had no ground-truth comparison at all
            # (Reviewer 1, Major Concern 1).
            roc_real = list(ground_truth_auc.true_roc_curve(scenario_num, X0_true, X1_true, p=p))

            for roc_values_real in roc_real:
                plt.plot(p, np.array(roc_values_real))

            plt.xlabel('1 - Specificity')
            plt.ylabel('Sensitivity')
            plt.title(f'Conditional ROC Curves for real {sceminario}_{dataset_name}')
            plt.savefig(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/roc_plotnn_real_{dataset_name}_{var_to_group}.png')

            plt.clf()

            # Mean Squared Error between predicted and ground-truth ROC curves, per
            # subject -- matches the roc_mse_values.csv format written by
            # mlp_reg_data_simulation_multi.py, aggregated by
            # src/postprocessing/statistics_summary.py.
            mse_values = [mean_squared_error(real, predicted) for predicted, real in zip(roc_predicted, roc_real)]
            mse_df = pd.DataFrame(mse_values, columns=['Mean Squared Error'])
            mse_df.to_csv(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/roc_mse_values.csv', index_label="Object Index")

            # Per-replicate computation time (Reviewer 1, Major Concern 1).
            timing_df = pd.DataFrame([{
                'Scenario': sceminario,
                'Method': 'RF',
                'Fit Seconds Total': fit_seconds_total,
            }])
            timing_df.to_csv(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/timing.csv', index=False)

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

            v1 = np.linspace(25, 80, num=51)
            # v1 = np.linspace(25, 75, num=51)
            # v1 = np.linspace(80, 115, num=71)
            # v2 = np.linspace(4.5, 7.0, num=51)
            v2 = np.linspace(90, 140, num=51)
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
            plt.savefig(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/healthy_comparison_plot_{sceminario}.png')
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
            plt.savefig(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/Diseased_comparison_plot_{sceminario}.png')
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
            plt.savefig(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/healthy_std_dev_comparison_plot_{sceminario}.png')
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
            plt.savefig(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/diseased_std_dev_comparison_plot_{sceminario}.png')
            plt.clf()

            df_summary = pd.DataFrame(summary_rows,
            columns=[
                'Scenario','Real Mean','Predicted Mean','Real Std Dev',
                'Predicted Std Dev','Generated Y','Residues','Mean-Function MSE','Std-Function MSE'
            ]
            )
        # ensure output folder exists
            os.makedirs(f'{args.output_dir}/{var_to_group+"_rf"}', exist_ok=True)

        # 1) Write LaTeX table
            tex_path = os.path.join(f'{args.output_dir}/{var_to_group+"_rf"}', "summary_rf_table.tex")
            with open(tex_path, "a", encoding="utf-8") as f:
                f.write("\\begin{table}[ht]\n  \\centering\n")
                f.write(df_summary.to_latex(index=False, escape=True, column_format="|l|" + "r|"*8))
                f.write("\\caption{Summary of Healthy vs Diseased}\n")
                f.write("\\label{tab:summary}\n")
                f.write("\\end{table}\n")

        # 2) Write DAT table (same rows, but using & separators)
            dat_path = os.path.join(f'{args.output_dir}/{var_to_group+"_rf"}', "summary_rf_table.dat")
            with open(dat_path, "a", encoding="utf-8") as f:
                f.write("\\begin{table}[ht]\n  \\centering\n")
                f.write("  \\begin{tabular}{|" + " l |"*9 + "}\n    \\hline\n")
                # header
                f.write("    " + " & ".join(df_summary.columns) + " \\\\\n    \\hline\n")
                # rows
                for _, row in df_summary.iterrows():
                    # escape underscores
                    scen = row["Scenario"]
                    vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row[1:]]
                    f.write(f"    {scen} & " + " & ".join(vals) + " \\\\\n    \\hline\n")
                f.write("  \\end{tabular}\n")
                f.write("\\caption{Summary of Healthy vs Diseased (DAT)}\n")
                f.write("\\label{tab:summary_dat}\n")
                f.write("\\end{table}\n")

            # Combine original input data with prediction results
            df_healthy_full = pd.concat([index_0.reset_index(drop=True), results_healthy.reset_index(drop=True)], axis=1)
            df_diseased_full = pd.concat([index_1.reset_index(drop=True), results_diseased.reset_index(drop=True)], axis=1)

            # Save to CSV
            df_healthy_full.to_csv(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/full_results_healthy_{sceminario}.csv', index=False)
            df_diseased_full.to_csv(f'{args.output_dir}/{var_to_group+"_rf"}/{sceminario}/full_results_diseased_{sceminario}.csv', index=False)






