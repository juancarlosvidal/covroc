"""Semiparametric linear-regression baseline for the covariate-adjusted ROC curve,
a Python port of the location-scale cROC.sp estimator from the R package ROCnReg
(Rodriguez-Alvarez, related to Related Work / Statistical Methods for Covariate-
Adjusted ROC). Two OLS models (statsmodels) estimate mu_h(x)/sigma_h and
mu_d(x)/sigma_d for the healthy and diseased groups; a(x) and b(x) follow the same
formulas as the paper's Gaussian location-scale model, and cROC_sp() adds bootstrap
confidence intervals by resampling the OLS residuals (do_boost_roc). This linear
model is the semiparametric benchmark shown to underperform the FNN and Random
Forest approaches under non-linear/interaction covariate effects.
"""
#%%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from scipy.integrate import simps
import matplotlib.pyplot as plt

# --- Helper functions ---

def pauccontrol():
    # Default pauc control parameters (modify as needed)
    return {"compute": False, "focus": "FPF", "value": 0.1}



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

def compute_ROC(formula_h, formula_d, data_h, data_d, newdata, est_cdf, pauc, p):


    """
    Fits two linear models (for healthy and diseased subjects), computes the covariate‐specific ROC curve 
    and AUC according to the chosen method (normal or empirical).
    """
    # p: numpy array of FPF values
    # Extract marker (the response variable) from the left‐hand side of formula_h
    marker = formula_h.split("~")[0].strip()
    
    # Fit linear models using the provided formulas
    fit_h = smf.ols(formula=formula_h, data=data_h).fit()
    sigma_h = np.sqrt(fit_h.mse_resid)
    fit_d = smf.ols(formula=formula_d, data=data_d).fit()
    sigma_d = np.sqrt(fit_d.mse_resid)
    
    # Get the union of coefficient names
    coeff_names = list(fit_h.params.index)
    for name in fit_d.params.index:
        if name not in coeff_names:
            coeff_names.append(name)
    beta_h = pd.Series(0, index=coeff_names)
    beta_d = pd.Series(0, index=coeff_names)
    beta_h.update(fit_h.params)
    beta_d.update(fit_d.params)
    
    # Compute “ROC” coefficients as in R: (beta.h - beta.d)/sigma_d and add b = sigma_h/sigma_d
    beta_ROC = (beta_h - beta_d) / sigma_d
    beta_ROC['b'] = sigma_h / sigma_d

    # Compute predictions on newdata:
    a = (fit_h.predict(newdata) - fit_d.predict(newdata)) / sigma_d
    b_val = sigma_h / sigma_d  # scalar

    if est_cdf == "normal":
        # Compute the ROC curve:
        # For each newdata row, for each false-positive fraction p, compute:
        #    ROC = 1 - Phi( a + b * Phi^(-1)(1-p) )
        q = norm.ppf(1 - p)  # quantiles for each p
        # Outer sum: shape (n_newdata, len(p))
        M = np.outer(a, np.ones_like(q)) + np.outer(np.ones_like(a), b_val * q)
        cROC = 1 - norm.cdf(M)
        # AUC for each newdata row
        cAUC = 1 - norm.cdf(a / np.sqrt(1 + b_val**2))
        if pauc.get('compute', False):
            # pAUC: additional computations would be needed here
            cpAUC = np.full_like(cAUC, np.nan)
    else:
        # The “empirical” branch is not fully detailed here.
        # (One would compute the empirical CDFs of the residuals, use np.percentile for quantiles, etc.)
        raise NotImplementedError("The 'empirical' est_cdf option is not implemented in this example.")

    res = {
        'p': p,
        'ROC': cROC,  # matrix of ROC estimates (n_newdata x len(p))
        'AUC': cAUC,  # vector of AUC estimates (length n_newdata)
        'fit': {'h': fit_h, 'd': fit_d},
        'coeff': {'h': fit_h.params, 'd': fit_d.params, 'ROC': beta_ROC}
    }
    if pauc.get('compute', False):
        res['pAUC'] = cpAUC
    return res

def do_boost_roc(i, formula_h, formula_d, data_h, data_d, newdata, croc, est_cdf, pauc, p, marker):
    """
    One bootstrap replication: resample the residuals (with replacement) and recalculate the ROC curve.
    """
    # Create bootstrap copies
    data_boot_h = data_h.copy()
    data_boot_d = data_d.copy()
    # Resample residuals from the original fits
    res_h = croc['fit']['h'].resid.values
    res_d = croc['fit']['d'].resid.values
    res_h_b = np.random.choice(res_h, size=len(res_h), replace=True)
    res_d_b = np.random.choice(res_d, size=len(res_d), replace=True)
    
    # Replace the marker values with fitted values plus bootstrapped residuals
    data_boot_h[marker] = croc['fit']['h'].fittedvalues + res_h_b
    data_boot_d[marker] = croc['fit']['d'].fittedvalues + res_d_b
    
    # Recalculate ROC using the bootstrapped data
    obj_boot = compute_ROC(formula_h, formula_d, data_boot_h, data_boot_d, newdata, est_cdf, pauc, p)
    res = {
        'ROC': obj_boot['ROC'],
        'AUC': obj_boot['AUC'],
        'coeff_h': obj_boot['coeff']['h'],
        'coeff_d': obj_boot['coeff']['d'],
        'coeff_ROC': obj_boot['coeff']['ROC']
    }
    if pauc.get('compute', False):
        res['pAUC'] = obj_boot['pAUC']
    return res

def cROC_sp(formula_h, formula_d, group, tag_h, data, newdata=None, 
            est_cdf="normal", pauc=None, p=None, B=1000, ci_level=0.95, 
            parallel="no", ncpus=1, cl=None):
    """
    Main function to compute the covariate‐specific ROC (cROC) and its bootstrap confidence intervals.
    
    Parameters:
      - formula_h, formula_d: formulas (as strings) for the healthy and diseased groups.
      - group: name of the grouping variable in data.
      - tag_h: value in `group` indicating a healthy subject.
      - data: pandas DataFrame with all variables.
      - newdata: DataFrame on which to compute the ROC; if None, data is used.
      - est_cdf: either "normal" or "empirical" (only "normal" is fully implemented here).
      - pauc: dictionary of pAUC options (see pauccontrol).
      - p: numpy array of false-positive fractions (default is 101 equally spaced points from 0 to 1).
      - B: number of bootstrap replications.
      - ci_level: confidence level (e.g. 0.95).
      
    Returns:
      A dictionary with the estimated ROC curves, AUC values, bootstrap confidence intervals,
      fitted models, and coefficients.
    """
    if pauc is None:
        pauc = pauccontrol()
    if p is None:
        p = np.linspace(0, 1, 101)
    if est_cdf not in ["normal", "empirical"]:
        raise ValueError("est_cdf must be either 'normal' or 'empirical'")
    if newdata is None:
        newdata = data.copy()
    
    # Extract the marker (response variable) from the formula (assumes "response ~ predictors")
    marker = formula_h.split("~")[0].strip()
    
    # Ensure the necessary variables are in the data and newdata.
    # Here we assume that the covariates are those appearing to the right of "~".
    rhs_vars_h = [v.strip() for v in formula_h.split("~")[1].split("+")]
    rhs_vars_d = [v.strip() for v in formula_d.split("~")[1].split("+")]
    covars = list(set(rhs_vars_h + rhs_vars_d))
    required_vars = [marker, group] + covars
    for var in required_vars:
        if var not in data.columns:
            raise ValueError(f"Variable {var} is not in data")
        if var not in newdata.columns:
            raise ValueError(f"Variable {var} is not in newdata")
    
    # Remove rows with missing values in the required variables
    data_clean = data.dropna(subset=required_vars)
    
    # Split the data into healthy and diseased groups
    data_h = data_clean[data_clean[group] == tag_h].copy()
    data_d = data_clean[data_clean[group] != tag_h].copy()
    
    # Compute the ROC and AUC using the original data
    res_fit = compute_ROC(formula_h, formula_d, data_h, data_d, newdata, est_cdf, pauc, p)
    croc = res_fit
    cROC_est = croc['ROC']      # shape: (n_newdata, len(p))
    cAUC_est = croc['AUC']      # shape: (n_newdata,)
    
    coeff_h_est = croc['fit']['h'].params
    coeff_d_est = croc['fit']['d'].params
    coeff_ROC_est = croc['coeff']['ROC']
    
    # Bootstrap replications if requested (B > 0)
    if B > 0:
        boot_results = []
        for i in range(B):
            boot_res = do_boost_roc(i, formula_h, formula_d, data_h, data_d, newdata, 
                                      croc, est_cdf, pauc, p, marker)
            boot_results.append(boot_res)
        
        # Stack bootstrap results for ROC and AUC:
        # boot_ROC: shape (n_newdata, len(p), B)
        boot_ROC = np.stack([br['ROC'] for br in boot_results], axis=-1)
        # boot_AUC: shape (n_newdata, B)
        boot_AUC = np.stack([br['AUC'] for br in boot_results], axis=-1)
        
        # For the coefficients, collect each bootstrap replication into a DataFrame
        coeff_h_boot = pd.concat([br['coeff_h'] for br in boot_results], axis=1)
        coeff_d_boot = pd.concat([br['coeff_d'] for br in boot_results], axis=1)
        coeff_ROC_boot = pd.concat([br['coeff_ROC'] for br in boot_results], axis=1)
    else:
        boot_ROC = boot_AUC = None

    # Compute bootstrap confidence intervals
    alpha = (1 - ci_level) / 2
    auc_ci_lower = np.percentile(boot_AUC, 100 * alpha, axis=-1)
    auc_ci_upper = np.percentile(boot_AUC, 100 * (1 - alpha), axis=-1)
    roc_ci_lower = np.percentile(boot_ROC, 100 * alpha, axis=-1)
    roc_ci_upper = np.percentile(boot_ROC, 100 * (1 - alpha), axis=-1)
    
    # For each coefficient, compute the quantiles across bootstrap replications.
    coeff_h_ci_lower = coeff_h_boot.quantile(alpha, axis=1)
    coeff_h_ci_upper = coeff_h_boot.quantile(1 - alpha, axis=1)
    coeff_d_ci_lower = coeff_d_boot.quantile(alpha, axis=1)
    coeff_d_ci_upper = coeff_d_boot.quantile(1 - alpha, axis=1)
    coeff_ROC_ci_lower = coeff_ROC_boot.quantile(alpha, axis=1)
    coeff_ROC_ci_upper = coeff_ROC_boot.quantile(1 - alpha, axis=1)
    
    # Organize the results in a dictionary (similar in spirit to the R object)
    result = {
        'call': {
            'formula_h': formula_h,
            'formula_d': formula_d,
            'group': group,
            'tag_h': tag_h,
            'est_cdf': est_cdf,
            'p': p,
            'B': B,
            'ci_level': ci_level
        },
        'newdata': newdata,
        'data': data,
        'marker': marker,
        'group': group,
        'tag_h': tag_h,
        'formula': {'h': formula_h, 'd': formula_d},
        'est_cdf': est_cdf,
        'p': p,
        'ci_fit': B > 0,
        'ci_level': ci_level,
        'ROC': {'est': cROC_est, 'ql': roc_ci_lower, 'qh': roc_ci_upper},
        'AUC': np.column_stack((cAUC_est, auc_ci_lower, auc_ci_upper)),
        'fit': croc['fit'],
        'coeff': {
            'h': pd.DataFrame({'est': coeff_h_est, 'ql': coeff_h_ci_lower, 'qh': coeff_h_ci_upper}),
            'd': pd.DataFrame({'est': coeff_d_est, 'ql': coeff_d_ci_lower, 'qh': coeff_d_ci_upper}),
            'ROC': pd.DataFrame({'est': coeff_ROC_est, 'ql': coeff_ROC_ci_lower, 'qh': coeff_ROC_ci_upper})
        }
    }
    if pauc.get('compute', False):
        # pAUC normalization (placeholder)
        if pauc.get('focus', 'FPF') == "FPF":
            pAUC_est_norm = croc['pAUC'] / pauc.get('value', 0.1)
        else:
            pAUC_est_norm = croc['pAUC'] / (1 - pauc.get('value', 0.1))
        result['pAUC'] = pAUC_est_norm
    return result

# --- Example usage ---

# Suppose df2 is a pandas DataFrame containing the columns:
#   - 'TAC2' (the biomarker),
#   - 'BMI' and 'RIDAGEYR_x' (covariates),
#   - 'cinco' (a grouping variable where healthy subjects have value 0).
#
# The following code fits the cROC model and then creates a plot analogous to the R code.

# Run from the repository root. input_real_2/df_gender_2.csv is produced upstream
# by the R data-prep step (see R/aroc_single_model.R).
df2 = pd.read_csv('input_real_2/df_gender_2.csv')
df2 = df2.rename(columns={'RIDAGEYR.x': 'RIDAGEYR_x'})

cROC_model = cROC_sp(
    formula_h="TAC2 ~ BMI + RIDAGEYR_x",
    formula_d="TAC2 ~ BMI + RIDAGEYR_x",
    group="cinco",
    tag_h=0,
    data=df2,
    newdata=df2,
    p=np.linspace(0, 1, 101),
    B=10  # using 10 bootstrap replications for a quick example
)

# Create the plot.
# In the R code the plot is: plot(df2$RIDAGEYR.x, 1 - cROC_model$AUC[,1])
# Here we assume the covariate is named 'RIDAGEYR_x' in df2.
plt.figure()
# cROC_model['AUC'] is a 2D array with three columns: [estimate, lower CI, upper CI]
plt.scatter(df2['RIDAGEYR_x'], 1 - cROC_model['AUC'][:, 0], color='blue', label='1 - AUC')
plt.xlabel("RIDAGEYR_x")
plt.ylabel("1 - AUC")
plt.title("cROC.sp Plot")
plt.legend()
plt.savefig("cROC_sp_plot.png")
plt.show()
# %%
