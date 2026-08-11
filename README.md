# covroc

Code accompanying the paper:

> **Covariate-Adjusted ROC Analysis Using Neural Networks for Biomarker Evaluation**
> Ziad Akram Ali Hammouri, Yating Zou, Rahul Ghosal, Juan C. Vidal, Marcos Matabuena
> Preprint (DOI / venue to be added upon publication)
>
> If you use this code, please cite the paper above.

This repository implements and compares covariate-adjusted ROC (aROC/cROC) curve estimation methods for biomarker evaluation: a feedforward neural network (FNN) approach, a Random Forest regression approach, an NN-based cROC baseline, a naive (pooled, no covariate adjustment) ROC baseline, a linear-regression baseline, a semiparametric additive (spline) baseline, and the semiparametric `AROC.sp`/`cROC.sp` estimators from the R package [`ROCnReg`](https://cran.r-project.org/package=ROCnReg). It also includes a classification pipeline with temperature scaling used in the paper.

**Method.** Both the FNN and Random Forest pipelines follow the paper's two-stage semiparametric approach (Methods, *Proposed framework: two-stage Semi-Parametric Neural Network Approach*): under a Gaussian location-scale model `Y | (X=x, D=d) ~ mu_d(x) + sigma_d(x) * eps`, a first regression stage estimates the conditional mean `mu_d(x)` for each group `d in {0,1}` (controls/cases), and a second stage estimates the conditional variance `sigma_d(x)^2` from the squared residuals of the first. The covariate-specific ROC curve is then obtained from `a(x) = (mu_1(x) - mu_0(x)) / sigma_1(x)` and `b(x) = sigma_0(x) / sigma_1(x)` via `aROC(p|x) = 1 - Phi(b(x) * Phi^-1(1-p) - a(x))`, with the AUC integrated numerically (`roc()` in `mlp_reg.py`/`mlp_reg_data_simulation_multi.py`/`rf_reg_data_simulation_multi_2.py`) using the empirical CDFs of the training residuals rather than assuming Gaussian errors. This code produces point estimates of the aROC/AUC surface; the subject-level bootstrap, cross-fitting, and out-of-bag aggregation used for the paper's confidence bands (Methods, *Uncertainty Quantification for Estimated ROC Curves*) are implemented on top of this same two-stage estimator in `src/simulation/bootstrap_crossfit_oob.py` (see below), plus the residual-bootstrap confidence intervals in the linear-regression baseline (`src/baselines/croc_linear_baseline.py`).

**Ground-truth comparison in the simulation study.** Since the true data-generating process is known for each of the 9 simulation scenarios, `src/simulation/ground_truth_auc.py` computes the true population-level aROC(p|x)/AUC(x) (exact closed form for the 8 Gaussian scenarios, Monte Carlo for Scenario 7's skew-normal/Student-t mixture arm) and every method -- FNN, Random Forest, naive, linear, and semiparametric-additive -- is scored against it via per-subject MSE, plus per-replicate wall-clock fit time, both aggregated by `src/postprocessing/statistics_summary.py` into a single comparison table across methods and scenarios. The Bayesian nonparametric estimator (`ROCnReg::AROC.bnp`) is not included in this Python pipeline: it has no equivalent Python implementation (it's a Dirichlet-process-mixture model specific to that package's authors), and reimplementing it from scratch was judged out of scope; if needed, run it via `ROCnReg` in R and report at least its single-replicate computation time, as the associated review response allows.

**Mean-/std-function estimation metrics (Reviewer 1, Minor Concerns 3 & 4).** Separately from the ROC-curve MSE above, `mlp_reg_data_simulation_multi.py`/`rf_reg_data_simulation_multi_2.py` also report, per replicate and group, `Mean-Function MSE` (predicted vs. true conditional mean) and `Std-Function MSE` (predicted vs. true conditional std, NaN for Scenario 7's healthy arm) into a `mean_std_mse.csv`, replacing the old bare `"MSE"` column (previously computed against the raw noisy `Y` rather than the true mean). `statistics_summary.py` aggregates these into `mean_std_mse_summary.csv`, and `mean_std_mse_boxplots.py` plots them per scenario/sample-size with FNN and RF compared side by side (reproducing the old Figures 10-18, which only showed the FNN). The `"Standard Deviation"` column of the ROC-curve-MSE table is renamed `"MSE SD Across Runs"` so it's never confused with the differently-defined `Std-Function MSE`.

**Known caveat -- Scenario 7's healthy arm, Mean-Function MSE.** Scenario 7's healthy (D=0) arm has no closed-form mean (skew-normal/Student-t mixture -- see `true_dgp.true_mean`'s docstring), so `data_generation.py`'s `True_Mean_Y` placeholder for that arm is a single Monte Carlo draw from the conditional distribution rather than an actual mean (`Y_generated_bar = true_mean_Y_bar` in `generate_scenario_VII`). `Mean-Function MSE` there therefore measures deviation from one noisy sample, not from a true expectation, and comes out roughly one to two orders of magnitude larger than every other scenario/group's `Mean-Function MSE` (e.g. ~0.12 vs. ~0.0007-0.03 elsewhere) -- an artifact of the metric's definition for this one arm, not a modeling or coverage problem. It doesn't affect the paper's primary ROC-curve/AUC comparison, which already handles Scenario 7 correctly via proper large-sample Monte Carlo in `ground_truth_auc.py`. Not fixed for now (would need `Mean-Function MSE`'s reference value to average many Monte Carlo draws instead of one, which means regenerating Scenario 7's data); if reporting this table, footnote Scenario 7's healthy-arm `Mean-Function MSE` as not comparable to the other rows.

**Simulation scenario corrections (Reviewer 1, Minor Concern 1).** Per the Supplementary Material, covariates are `U(-1, 1)` in every scenario; the code previously drew most of them from `Normal(0, 1)` (Scenarios I-VI, VIII) or `U(0, 1)` (Scenario VII) instead -- now fixed in `data_generation.py`, with two deliberate, documented exceptions: Scenario VI's third covariate stays `Bernoulli(0.5)` (it's used as a 0/1 interaction switch, not a continuous effect) and Scenario VII's covariate stays `U(0, 1)` (its mixture weight `exp(-2x)` is only a valid probability for `x >= 0`). Separately, Scenario III's own formula in the Supplementary Material -- titled "Covariate Effect on Both Mean and Variance" -- actually has constant variance, identical in structure to Scenario II; there was no existing spec to fix it against, so `true_dgp.py` now defines a new, explicitly-designed `sigma(x) = base * exp(0.3*x)` for that scenario only. **Both changes alter the generated data**, so Table 1 and Figures 10-18 in the Supplementary Material are stale until the full 9-scenario x 2-sample-size x 100-replicate sweep (`hpc/py_gen.sh` then the training scripts) is rerun.

**Case study.** The real-data experiments evaluate total activity count (TAC), a proxy for daily step count derived from NHANES 2011–2014 accelerometry (MIMS units), as a biomarker for all-cause mortality at 3-, 5-, and 8-year horizons, adjusted for age, sex, and BMI (`n = 5,006`; see Table 1 of the paper). The corresponding group/target variables in the real-data pipeline are named in Spanish: `tres` (3-year), `cinco` (5-year), and `ocho` (8-year) mortality status, plus `mortstat` for overall mortality; the biomarker column is `TAC`/`TAC2`. Variables derived from NHANES (`mortstat`, `RIDAGEYR`/Age, `BMI`, `Cancer`, `TAC`) are included under `data/`, including sex-stratified extracts `df_f.csv`/`df_m.csv` (female/male) used by `notebooks/nhanes_hetero_residuos.ipynb`.

**Bootstrap + cross-fitting + OOB uncertainty quantification (Reviewer 2, Major Comment 4).** `src/simulation/bootstrap_crossfit_oob.py` implements the subject-level bootstrap combined with group-aware `K`-fold cross-fitting and out-of-bag (OOB) aggregation described in the paper's Methods (*Uncertainty Quantification for Estimated ROC Curves*), on top of the **same two-stage mean/variance estimator** used everywhere else in this repository for the FNN point estimates (`mlp_reg_data_simulation_multi.py`'s `MLP`/`train_model`/`compute_mean`/`compute_std`/`compute_residues`) rather than a separately-trained joint-likelihood heteroscedastic network -- so the confidence bands quantify uncertainty in the same estimator whose point estimates are reported in Table 1 and Figures 2-18. Within each bootstrap replicate, each cross-fitting fold's mean and variance models are trained only on that fold's train partition, and standardized residuals are collected only from the held-out validation rows, fixing the in-sample residual-variance bias Reviewer 2's Major Comment 3 raises. Two entry points share this engine: `bootstrap_crossfit_oob_shared`, used by `notebooks/nhanes_hetero_residuos.ipynb` for the real-data NHANES case study (one dataset split into two groups, evaluated at a shared covariate profile `x`, run from the repository root so `./data`, `src/real_data`, and `src/simulation` imports resolve); and `bootstrap_crossfit_oob_paired`, used by `src/simulation/coverage_bootstrap_crossfit.py` for the 9 simulation scenarios (two independently-drawn covariate pools, evaluated row-paired against `src/simulation/ground_truth_auc.py`'s known ground-truth AUC). The latter writes a per-subject `coverage.csv` (true AUC, estimated AUC mean/95% CI, `covered` boolean, OOB sample count), which `statistics_summary.py` aggregates into `coverage_summary.csv` -- the empirical pointwise coverage rate Major Comment 4 explicitly asks the simulation study to report. Scope of this pass: **pointwise** coverage only (not simultaneous/uniform bands) and **FNN** only (not Random Forest), matching the notebook's original real-data scope. Since the confidence bands and every other reported result now use the same two-stage procedure, the recommended fix for Major Comment 3's other half (the main text describing a "joint Gaussian likelihood" while the Supplementary Material's Algorithm 1 describes two separate models) is to correct the main text to describe the two-stage procedure as authoritative, rather than switching the paper to a joint-likelihood model.

## Repository structure

```
data/                    CSV datasets (NHANES-derived, public/anonymized)
src/
  simulation/             Simulated-data pipeline (wired into hpc/ SLURM scripts)
    data_generation.py            generates the 9 simulation scenarios (Scenarios I-IX, Supplementary
                                   Material) with linear, non-linear, and interaction covariate effects
    true_dgp.py                    single source of truth for each scenario's true conditional
                                   mean/std/sampling distribution; used by both data_generation.py
                                   and the ground-truth comparison below
    ground_truth_auc.py            true (population-level) aROC(p|x)/AUC per scenario, built from
                                   true_dgp.py -- exact closed form for the 8 Gaussian scenarios,
                                   Monte Carlo for Scenario 7's non-Gaussian mixture arm
    eval_io.py                      shared roc_mse_values.csv/timing.csv writer used by the baselines below
    data_simulation_reg.py        dataset loader for the simulation pipeline
    mlp_reg_data_simulation_multi.py   two-stage FNN mean/variance regression + aROC/AUC vs. ground truth
    rf_reg_data_simulation_multi_2.py  same two-stage pipeline with a Random Forest regressor
    naive_roc_baseline.py          naive/pooled ROC baseline (no covariate adjustment) vs. ground truth
    linear_reg_data_simulation.py  linear and semiparametric-additive (spline) aROC baselines vs. ground
                                   truth, reusing cROC_sp from src/baselines/croc_linear_baseline.py
    bootstrap_crossfit_oob.py      shared bootstrap + K-fold cross-fitting + OOB confidence-interval
                                   engine (two-stage estimator), used by the notebook below and by
                                   coverage_bootstrap_crossfit.py
    coverage_bootstrap_crossfit.py pointwise bootstrap-OOB coverage of the FNN two-stage estimator
                                   against the ground-truth AUC, per simulation scenario replicate
  real_data/              Regression pipeline on real (NHANES-derived) data
    data_reg_real.py              dataset loader; TAC/TAC2 biomarker, age/BMI/sex covariates
    mlp_reg.py                    two-stage FNN pipeline producing the age x BMI AUC surfaces
                                   (Figures 2-4: 'tres'/'cinco'/'ocho' = 3-/5-/8-year mortality)
  classification/         Classification pipeline with temperature scaling
    data.py, models.py, losses.py, metrics.py, temperature_scaling.py, train.py
    croc_nn_baseline.py            NN-based cROC baseline (uses train.cv_loop)
  baselines/
    croc_linear_baseline.py        semiparametric linear-regression cROC estimator (cROC_sp/compute_ROC)
                                    with residual bootstrap confidence intervals (Python port of
                                    ROCnReg::cROC.sp), used both for the real-data NHANES case study
                                    (its own __main__ block) and, via src/simulation/
                                    linear_reg_data_simulation.py, for the simulation-scenario baselines
  postprocessing/
    statistics_summary.py          aggregate MSE, timing, mean-/std-function MSE, and pointwise
                                   bootstrap-OOB coverage statistics across output folders
    mean_std_mse_boxplots.py       per-scenario Mean-/Std-Function MSE boxplots, methods compared side by side
    write_latex_table.py           per-scenario, per-method ROC-MSE comparison table (.tex/.dat)
                                   from statistics_summary.csv
R/
  aroc_batch_scenarios.R    AROC.sp/cROC.sp over all scenario CSVs in input_real_2/, incl. 3D surface plots
  aroc_crude_vs_adjusted.R  crude vs. confounder-adjusted ROC comparison
  aroc_single_model.R       single AROC.sp model + NHANES data prep + batch AROC/cROC runs
notebooks/
  nhanes_hetero_residuos.ipynb  subject-level bootstrap + cross-fitting + OOB confidence intervals
                                 for the NHANES case study (uses src/simulation/bootstrap_crossfit_oob.py;
                                 see note below)
hpc/
  py_gen.sh, py_mlp.sh, py_rfo.sh, py_cov.sh, py_post.sh    SLURM job scripts (submitted via sbatch;
                                                             py_post.sh is CPU-only, no --gres=gpu)
  py_cov_array.sh              SLURM array version of py_cov.sh -- one replicate per array task,
                               so coverage_bootstrap_crossfit.py's replicates run concurrently
                               across GPUs instead of sequentially in a single job; use this, not
                               py_cov.sh, for a real per-scenario coverage sweep
  run_gen.sh, run_mlp.sh, run_rfo.sh, run_cov.sh, run_cov_array.sh, run_post.sh
                               convenience wrappers around sbatch
  combine_outputs.sh          symlinks each method's per-replicate folders into one method-prefixed
                              root for cross-method aggregation (no sbatch needed, just file I/O)
archive/
  superseded prototypes and early drafts, kept for reference (see archive/README.md)
```

## Setup

### Python

```bash
conda env create -f environment.yml
conda activate covroc
```

This installs the CPU build of PyTorch by default; for GPU/CUDA support (used by the `hpc/` SLURM jobs), follow the [PyTorch install instructions](https://pytorch.org/get-started/locally/) for your CUDA version instead, e.g. `conda install pytorch pytorch-cuda=<version> -c pytorch -c nvidia`.

### R

R is deliberately **not** managed via `environment.yml`. The conda-forge `r-base` build leaves `CC17` undefined in its `Makeconf` (confirmed on r-base 4.3, 4.4, and 4.6), so packages requiring a C17 compiler fail to build through it — including `cubature`, a dependency of `ROCnReg` — regardless of which extra conda-forge compiler packages are added.

Install R directly instead (e.g. via the [official CRAN installer](https://cran.r-project.org/) or your OS's package manager), which ships a complete toolchain and compiles `cubature` fine, then install the packages the `R/` scripts need:

```r
install.packages(c("ROCnReg", "plotly", "htmlwidgets", "webshot2", "dplyr"))
```

(`splines` ships with base R, no separate install needed.)

## Running the pipeline

All commands below are run from the repository root.

**1. Generate simulated data** (writes to `input_real_2/`, already in the wide, `mortstat`-labeled form the regression loaders expect):
```bash
python src/simulation/data_generation.py
```

**2. Train/evaluate every method** on a scenario folder, each scored against the true aROC(p|x) and timed:
```bash
python src/simulation/mlp_reg_data_simulation_multi.py -i input_real_2/scenario_1 -o output_real/scenario_1 -c 6 -f 5 -e 800 -b 64
python src/simulation/rf_reg_data_simulation_multi_2.py -i input_real_2/scenario_1 -o output_real/scenario_1 -c 6 -f 2 -e 800 -b 64
python src/simulation/naive_roc_baseline.py -i input_real_2/scenario_1 -o output_real/scenario_1
python src/simulation/linear_reg_data_simulation.py -i input_real_2/scenario_1 -o output_real/scenario_1 --formula-type linear
python src/simulation/linear_reg_data_simulation.py -i input_real_2/scenario_1 -o output_real/scenario_1 --formula-type spline
```
Each writes `roc_mse_values.csv`/`timing.csv` per replicate folder; running more than one method into the *same* `-o` overwrites files if folder names collide (they share the `scenario_<N>_<n>_<g>` naming), so give each method its own `-o` root when comparing them (see step 7).

Repeat for `scenario_1` .. `scenario_9`. On a SLURM cluster, use the wrappers in `hpc/` instead (set `PYTHON=/path/to/python` to override the default interpreter path):
```bash
./hpc/run_mlp.sh ./input_real_2/scenario_1 ./output_real/scenario_1
./hpc/run_rfo.sh ./input_real_2/scenario_1 ./output_real/scenario_1
```

**2b. Pointwise bootstrap-OOB coverage** of the FNN two-stage estimator against the ground-truth AUC, per scenario folder (Reviewer 2, Major Comment 4; each cross-fitting fold trains its own mean/variance model pair, so this is far more compute-heavy than step 2 -- `-b`/`-k` control how many bootstrap replicates/cross-fitting folds, `min_samples=6` OOB draws per subject are needed before a CI is reported, so `-b` needs to be large enough that most subjects clear that bar). **Measured cost at `-b 150 -k 5 -e 800`: ~2.45h/replicate at n=5000, ~9.1h/replicate at n=20000** -- and `coverage_bootstrap_crossfit.py` processes every file in `-i` sequentially in one GPU allocation, so pointing it at a full 100-replicate scenario folder means one GPU grinding through it for weeks, not an oversight to just "let run longer". Coverage doesn't need 100 replicates -- each one already contributes thousands of rows, so a much smaller subset (e.g. 15, n=5000 only) still gives a stable per-scenario coverage rate:
```bash
python src/simulation/coverage_bootstrap_crossfit.py -i input_real_2/scenario_1 -o output/coverage/scenario_1 -b 150 -k 5 -e 800
```
(run against a folder holding only the replicate CSVs you actually want processed, not the full 100 -- e.g. copy/symlink a subset into its own directory first).

On a SLURM cluster, **use the array form**, `hpc/py_cov_array.sh`/`run_cov_array.sh`, not `py_cov.sh`/`run_cov.sh`: it runs one replicate per array task (so replicates train concurrently across GPUs instead of queued one after another in a single job), automatically restricts to `*_5000_*` files, and skips replicates that already have a `coverage.csv` (safe to re-run/extend):
```bash
./hpc/run_cov_array.sh input_real_2/scenario_1 output/coverage/scenario_1        # first 15 n=5000 replicates
./hpc/run_cov_array.sh input_real_2/scenario_1 output/coverage/scenario_1 15 5   # same, capped at 5 concurrent tasks
```
Repeat per scenario. Writes `coverage.csv`/`timing.csv` per replicate folder, aggregated by `statistics_summary.py` in step 7b. `py_cov.sh`/`run_cov.sh` (whole-folder, single job) still exist for ad hoc runs against a small, already-curated input folder -- not for a full scenario sweep.

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

**7a. Combine each method's output into one root.** Steps 2/2b run each method into its own `-o` (e.g. `output/fnn/scenario_1`, `output/rf/scenario_1`, `output/coverage/scenario_1`, ...), since sharing one `-o` across methods collides replicate-folder names. `statistics_summary.py`/`mean_std_mse_boxplots.py`/`write_latex_table.py` (7b below) instead expect one root containing all of them together, one method-prefixed copy per replicate folder (`fnn_scenario_1_5000_1/`, `rf_scenario_1_5000_1/`, ...). `hpc/combine_outputs.sh` builds that root via symlinks (instant, no data duplication -- safe to run directly, no `sbatch` needed):
```bash
./hpc/combine_outputs.sh output output/combined fnn rf coverage
```
(list whichever method subfolders you actually have under `output/` -- e.g. add `naive linear spline` once those are run too).

**7b. Post-processing.** `statistics_summary.py` aggregates every replicate folder under `--root-dir` into `roc_mse_values.csv`-based MSE statistics, (from each folder's `timing.csv`) a `timing_summary.csv` of per-scenario, per-method fit time, (from each folder's `mean_std_mse.csv`, written only by the FNN/RF scripts) a `mean_std_mse_summary.csv`, and (from each folder's `coverage.csv`, written only by `coverage_bootstrap_crossfit.py`) a `coverage_summary.csv` of per-scenario, per-method empirical pointwise coverage rate. `mean_std_mse_boxplots.py` then plots the FNN-vs-RF Mean-/Std-Function MSE comparison per scenario (one PNG per scenario/sample-size, ~18 total for the full 9-scenario x 2-sample-size sweep), and `write_latex_table.py` turns `statistics_summary.csv`'s per-replicate rows -- parsing the method out of each `<method>_scenario_<N>_<n>_<replicate>` folder name -- into a per-scenario, per-method `summary_table_mse.tex`/`.dat` (mean ROC-curve MSE and its across-replicate SD):
```bash
python src/postprocessing/statistics_summary.py --root-dir output/combined --output-csv output/combined/statistics_summary.csv --timing-csv output/combined/timing_summary.csv --mean-std-mse-csv output/combined/mean_std_mse_summary.csv --coverage-csv output/combined/coverage_summary.csv
python src/postprocessing/mean_std_mse_boxplots.py --root-dir output/combined --output-dir output/combined/mean_std_mse_boxplots
python src/postprocessing/write_latex_table.py --stats-csv output/combined/statistics_summary.csv --tex-output output/combined/summary_table_mse.tex --dat-output output/combined/summary_table_mse.dat
```
This step is pure pandas/matplotlib aggregation over already-produced CSVs -- no GPU, and much lighter than steps 2/2b/3 (seconds to low minutes even at full scale). On a SLURM cluster where the login node isn't meant for even light processing, `hpc/py_post.sh` runs it as a small CPU-only job (no `--gres=gpu`, so it doesn't queue behind GPU jobs for no reason) -- pass it the *combined* directory from 7a, not `.`, since `run_post.sh` (like the other `run_*.sh` wrappers) always resolves its argument relative to the repository root, not to whatever directory your shell happens to be in when you invoke it:
```bash
./hpc/run_post.sh output/combined
```
(writes the three summary CSVs, the boxplots, and `summary_table_mse.tex`/`.dat` under the given root dir).

**8. Uncertainty quantification for the real-data case study** (bootstrap + cross-fitting + OOB using the same two-stage estimator as the rest of the pipeline, via `bootstrap_crossfit_oob_shared`, per group/mortality-horizon; writes per-subject AUC + 95% CI CSVs and smoothed AUC-vs-age plots under `./output`, git-ignored):
```bash
jupyter notebook notebooks/nhanes_hetero_residuos.ipynb
```

## Notes

- The `R/` scripts originally used hardcoded local paths; they now assume they're run from the repository root. `aroc_single_model.R` also references a `data_analysis_TD_2003_19.rda` file that is not included in this repository.
- Directories produced at runtime (`input_real_2/`, `output/`, `output_real/`, etc.) are git-ignored.
- `archive/` contains earlier/superseded versions of scripts, kept for reference only — see `archive/README.md`.
