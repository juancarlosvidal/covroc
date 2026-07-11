# covroc

Code accompanying the paper:

> **Covariate-Adjusted ROC Analysis Using Neural Networks for Biomarker Evaluation**
> Ziad Akram Ali Hammouri, Yating Zou, Rahul Ghosal, Juan C. Vidal, Marcos Matabuena
> Preprint (DOI / venue to be added upon publication)
>
> If you use this code, please cite the paper above.

This repository implements and compares covariate-adjusted ROC (aROC/cROC) curve estimation methods for biomarker evaluation: a feedforward neural network (FNN) approach, a Random Forest regression approach, an NN-based cROC baseline, a linear-regression baseline, and the semiparametric `AROC.sp`/`cROC.sp` estimators from the R package [`ROCnReg`](https://cran.r-project.org/package=ROCnReg). It also includes a classification pipeline with temperature scaling used in the paper.

**Method.** Both the FNN and Random Forest pipelines follow the paper's two-stage semiparametric approach (Methods, *Proposed framework: two-stage Semi-Parametric Neural Network Approach*): under a Gaussian location-scale model `Y | (X=x, D=d) ~ mu_d(x) + sigma_d(x) * eps`, a first regression stage estimates the conditional mean `mu_d(x)` for each group `d in {0,1}` (controls/cases), and a second stage estimates the conditional variance `sigma_d(x)^2` from the squared residuals of the first. The covariate-specific ROC curve is then obtained from `a(x) = (mu_1(x) - mu_0(x)) / sigma_1(x)` and `b(x) = sigma_0(x) / sigma_1(x)` via `aROC(p|x) = 1 - Phi(b(x) * Phi^-1(1-p) - a(x))`, with the AUC integrated numerically (`roc()` in `mlp_reg.py`/`mlp_reg_data_simulation_multi.py`/`rf_reg_data_simulation_multi_2.py`) using the empirical CDFs of the training residuals rather than assuming Gaussian errors. This code produces point estimates of the aROC/AUC surface; the subject-level bootstrap, cross-fitting, and out-of-bag aggregation used for the paper's confidence bands (Methods, *Uncertainty Quantification for Estimated ROC Curves*) are not included here, with the exception of the residual-bootstrap confidence intervals in the linear-regression baseline (`src/baselines/croc_linear_baseline.py`).

**Case study.** The real-data experiments evaluate total activity count (TAC), a proxy for daily step count derived from NHANES 2011–2014 accelerometry (MIMS units), as a biomarker for all-cause mortality at 3-, 5-, and 8-year horizons, adjusted for age, sex, and BMI (`n = 5,006`; see Table 1 of the paper). The corresponding group/target variables in the real-data pipeline are named in Spanish: `tres` (3-year), `cinco` (5-year), and `ocho` (8-year) mortality status, plus `mortstat` for overall mortality; the biomarker column is `TAC`/`TAC2`. Variables derived from NHANES (`mortstat`, `RIDAGEYR`/Age, `BMI`, `Cancer`, `TAC`) are included under `data/`.

## Repository structure

```
data/                    CSV datasets (NHANES-derived, public/anonymized)
src/
  simulation/             Simulated-data pipeline (wired into hpc/ SLURM scripts)
    data_generation.py            generates the 9 simulation scenarios (Scenarios I-IX, Supplementary
                                   Material) with linear, non-linear, and interaction covariate effects
    data_simulation_reg.py        dataset loader for the simulation pipeline
    mlp_reg_data_simulation_multi.py   two-stage FNN mean/variance regression + aROC/AUC entrypoint
    rf_reg_data_simulation_multi_2.py  same two-stage pipeline with a Random Forest regressor
  real_data/              Regression pipeline on real (NHANES-derived) data
    data_reg_real.py              dataset loader; TAC/TAC2 biomarker, age/BMI/sex covariates
    mlp_reg.py                    two-stage FNN pipeline producing the age x BMI AUC surfaces
                                   (Figures 2-4: 'tres'/'cinco'/'ocho' = 3-/5-/8-year mortality)
  classification/         Classification pipeline with temperature scaling
    data.py, models.py, losses.py, metrics.py, temperature_scaling.py, train.py
    croc_nn_baseline.py            NN-based cROC baseline (uses train.cv_loop)
  baselines/
    croc_linear_baseline.py        semiparametric linear-regression cROC baseline with residual
                                    bootstrap confidence intervals (Python port of ROCnReg::cROC.sp)
  postprocessing/
    convert_to_wide.py             reshape simulation output to wide form
    statistics_summary.py          aggregate MSE statistics across output folders
    write_latex_table.py           helper to build LaTeX tables from results
R/
  aroc_batch_scenarios.R    AROC.sp/cROC.sp over all scenario CSVs in input_real_2/, incl. 3D surface plots
  aroc_crude_vs_adjusted.R  crude vs. confounder-adjusted ROC comparison
  aroc_single_model.R       single AROC.sp model + NHANES data prep + batch AROC/cROC runs
hpc/
  py_gen.sh, py_mlp.sh, py_rfo.sh    SLURM job scripts (submitted via sbatch)
  run_gen.sh, run_mlp.sh, run_rfo.sh convenience wrappers around sbatch
archive/
  superseded prototypes and early drafts, kept for reference (see archive/README.md)
```

## Setup

```bash
conda env create -f environment.yml
conda activate covroc
```

This installs the CPU build of PyTorch by default; for GPU/CUDA support (used by the `hpc/` SLURM jobs), follow the [PyTorch install instructions](https://pytorch.org/get-started/locally/) for your CUDA version instead, e.g. `conda install pytorch pytorch-cuda=<version> -c pytorch -c nvidia`.

R scripts additionally require the `ROCnReg`, `splines`, `plotly`, `htmlwidgets`, `webshot2`, and `dplyr` packages.

## Running the pipeline

All commands below are run from the repository root.

**1. Generate simulated data** (writes to `data_simulation/` and `input_real_2/`):
```bash
python src/simulation/data_generation.py
```

**2. Train/evaluate MLP or Random Forest regression** on a scenario folder:
```bash
python src/simulation/mlp_reg_data_simulation_multi.py -i input_real_2/scenario_1 -o output_real/scenario_1 -c 6 -f 5 -e 800 -b 64
python src/simulation/rf_reg_data_simulation_multi_2.py -i input_real_2/scenario_1 -o output_real/scenario_1 -c 6 -f 2 -e 800 -b 64
```
Repeat for `scenario_1` .. `scenario_9`. On a SLURM cluster, use the wrappers in `hpc/` instead (set `PYTHON=/path/to/python` to override the default interpreter path):
```bash
./hpc/run_mlp.sh ./input_real_2/scenario_1 ./output_real/scenario_1
./hpc/run_rfo.sh ./input_real_2/scenario_1 ./output_real/scenario_1
```

**3. Real-data regression** (see `python src/real_data/mlp_reg.py -h` for its `-i`/`-o` and hyperparameter flags, which follow the same pattern as step 2):
```bash
python src/real_data/mlp_reg.py -i <input_dir> -o <output_file>
```

**4. cROC baselines:**
```bash
python src/baselines/croc_linear_baseline.py
python src/classification/croc_nn_baseline.py
```

**5. Classification with temperature scaling:**
```bash
python src/classification/train.py
```

**6. AROC/cROC comparison in R** (run from repository root so relative paths resolve):
```bash
Rscript R/aroc_crude_vs_adjusted.R
Rscript R/aroc_batch_scenarios.R
Rscript R/aroc_single_model.R
```

**7. Post-processing:**
```bash
python src/postprocessing/convert_to_wide.py
python src/postprocessing/statistics_summary.py --root-dir output
python src/postprocessing/write_latex_table.py
```

## Notes

- The `R/` scripts originally used hardcoded local paths; they now assume they're run from the repository root. `aroc_single_model.R` also references a `data_analysis_TD_2003_19.rda` file that is not included in this repository.
- Directories produced at runtime (`data_simulation/`, `input_real_2/`, `output/`, `output_real/`, etc.) are git-ignored.
- `archive/` contains earlier/superseded versions of scripts, kept for reference only — see `archive/README.md`.
