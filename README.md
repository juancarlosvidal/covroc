# covroc

Code accompanying the paper:

> **Covariate-Adjusted ROC Analysis Using Neural Networks for Biomarker Evaluation**
> Ziad Akram Ali Hammouri, Yating Zou, Rahul Ghosal, Juan C. Vidal, Marcos Matabuena
> Preprint (DOI / venue to be added upon publication)
>
> If you use this code, please cite the paper above.

This repository implements and compares covariate-adjusted ROC (aROC/cROC) curve estimation methods for biomarker evaluation: a feedforward neural network (FNN) approach, a Random Forest regression approach, an NN-based cROC baseline, a naive (pooled, no covariate adjustment) ROC baseline, a linear-regression baseline, a semiparametric additive (spline) baseline, and the semiparametric `AROC.sp`/`cROC.sp` estimators from the R package [`ROCnReg`](https://cran.r-project.org/package=ROCnReg). It also includes a classification pipeline with temperature scaling used in the paper.

**Method (point estimation).** Both the FNN and Random Forest pipelines follow the paper's two-stage semiparametric approach (Methods, *Model Estimation via Neural Networks*): under a Gaussian location-scale model `Y | (X=x, D=d) ~ mu_d(x) + sigma_d(x) * eps`, a first regression stage estimates the conditional mean `mu_d(x)` for each group `d in {0,1}` (controls/cases) by weighted least squares, and a second stage of the same architecture estimates the conditional variance from the squared residuals of the first, recovering `sigma_d(x) = sqrt(max(v_d(x), 0))`. The covariate-specific ROC curve then follows from `a(x) = (mu_1(x) - mu_0(x)) / sigma_1(x)` and `b(x) = sigma_0(x) / sigma_1(x)` via `aROC(p|x) = 1 - Phi(b(x) * Phi^-1(1-p) - a(x))`, and the AUC from `Phi(a(x) / sqrt(1 + b(x)^2))`. Uncertainty quantification is a separate procedure, described below.

**Uncertainty quantification: local kNN.** The confidence intervals reported in the paper come from `src/simulation/coverage_local_knn.py` (simulation study) and `src/real_data/local_knn_bands_real.py` (NHANES case study). The network is used only to define a neighbourhood: each arm is summarized by its fitted `(mu_d(x), sigma_d(x))`, the `k_d(x)` nearest held-out subjects of each arm in that summary -- or, with `-loc covariate`, in the raw covariate space -- form the local sample, and the conditional AUC is the Mann-Whitney statistic over those `k_1 x k_0` pairs, with a DeLong variance and a Wilson fallback where that variance degenerates at perfect local separation (`auc_interval`). Because the network enters only through the neighbourhood, the interval inherits no distributional assumption from it. The two localization spaces were both measured and agree closely (mean 95% coverage 0.926 either way, largest per-scenario difference 0.007). The simulation run writes a per-target `coverage.csv` (true AUC, estimate, CI at each requested level, `Covered`, fallback flag) over a fixed covariate grid; the real-data run writes `bands_<horizon>_<sex>.csv` with the local AUC and its intervals, the FNN plug-in at the same profile, and neighbourhood diagnostics (realized age span, IQR and median offset, pooled and per arm). Survey weights are supported in the real-data path via a Hajek-weighted Mann-Whitney statistic and the matching weighted DeLong variance (`weighted_delong_auc_var`, which reduces exactly to `delong_auc_var` under equal weights), with Kish's effective sample size (`kish_neff`) replacing `k` in the saturation fallback.

**Superseded: bootstrap + cross-fitting + OOB.** `src/simulation/bootstrap_crossfit_oob.py` and `coverage_bootstrap_crossfit.py` implement a subject-level bootstrap with group-aware `K`-fold cross-fitting and out-of-bag aggregation, on top of the same two-stage estimator. They are kept because the paper reports why this route was abandoned: empirical pointwise coverage of a nominal 95% interval ranges from 0.753 to 0.996 across the nine scenarios (`output/combined/coverage_summary.csv`), conservative where the estimation problem is easy and anticonservative where it is hard. **No number in the paper comes from this procedure.**

**Ground-truth comparison in the simulation study.** Since the true data-generating process is known for each of the 9 simulation scenarios, `src/simulation/ground_truth_auc.py` computes the true population-level aROC(p|x)/AUC(x) (exact closed form for the 8 Gaussian scenarios, Monte Carlo for Scenario 7's skew-normal/Student-t mixture arm) and every method -- FNN, Random Forest, naive, linear, and semiparametric-additive -- is scored against it via per-subject MSE, plus per-replicate wall-clock fit time, both aggregated by `src/postprocessing/statistics_summary.py` into a single comparison table across methods and scenarios. The Bayesian nonparametric estimator (`ROCnReg::AROC.bnp`) is not included in this Python pipeline: it has no equivalent Python implementation (it's a Dirichlet-process-mixture model specific to that package's authors), and reimplementing it from scratch was judged out of scope; its wall-clock cost is instead reported via `R/aroc_bnp_timing.R` (`hpc/r_bnp_timing_array.sh`/`run_bnp_timing_array.sh`, aggregated by `src/postprocessing/aroc_bnp_timing_summary.py`/`hpc/py_bnp_summary.sh` into `output/aroc_bnp_timing/aroc_bnp_timing_summary.csv`). Measured across the 9 scenarios: ~316-434s at n=5000 and ~1282-1569s at n=20000, one to two orders of magnitude slower than every other method's `timing_summary.csv` entries, which is the basis for not running it at the other methods' 100-replicate scale. Note that Scenario 8 at n=20000 has no timing run; any table reporting it should leave that cell empty. Scenario 7 is the only scenario whose ground truth is built by Monte Carlo rather than in closed form, and the only one that was scored against a fixed arm order before the ground truth was oriented the same way every estimator is; `rescore_scenario7_orientation.py` recomputes its ROC-curve MSE from stored predictions under both conventions, without refitting.

**`cROC.sp` Python port validation.** `src/baselines/croc_linear_baseline.py`'s `cROC_sp` (the semiparametric linear/spline baseline that `linear_reg_data_simulation.py` runs at scale for all 9 scenarios) is a Python port of `ROCnReg::cROC.sp`, not a call into the R package itself. `R/croc_sp_validation.R` and `src/baselines/croc_sp_validation.py` (run via `hpc/croc_sp_validation_array.sh`/`run_croc_sp_validation_array.sh`, diffed by `src/postprocessing/croc_sp_validation_summary.py`/`hpc/py_croc_sp_validation_summary.sh`) confirm the two agree to floating-point precision (~1e-15/1e-16 max absolute difference in AUC and in the ROC(p) curve) across all 9 scenarios and both sample sizes -- both are closed-form OLS estimators of the same model, so this is an exact-match check, not a statistical one. `R/aroc_batch_scenarios.R` (real-data-only: wrong directory layout and column names for the simulation scenarios) was never part of this comparison.

**Biomarker direction.** TAC is protective: decedents have systematically lower values, so a raw AUC computed under the Methods' higher-is-worse convention comes out below 0.5. Two conventions exist in this repository and they are not the same thing. `compute_ROC`/`cROC_sp` in `croc_linear_baseline.py` default to `adaptive_direction=True`, which reports `max(AUC, 1-AUC)` per covariate profile and therefore never falls below 0.5; `src/real_data/naive_roc_baseline.py` applies an unconditional `1 - AUC`. `local_knn_bands_real.py` takes `--orientation auto|fixed` and defaults to `auto` (the max convention), matching the Random Forest and GAM panels. Any table that mixes them will disagree with itself at profiles where the arms barely separate -- check which one produced a given number before comparing.

**Mean-/std-function estimation metrics.** Separately from the ROC-curve MSE above, `mlp_reg_data_simulation_multi.py`/`rf_reg_data_simulation_multi_2.py` also report, per replicate and group, `Mean-Function MSE` (predicted vs. true conditional mean) and `Std-Function MSE` (predicted vs. true conditional std, NaN for Scenario 7's healthy arm) into a `mean_std_mse.csv`. `statistics_summary.py` aggregates these into `mean_std_mse_summary.csv`, and `mean_std_mse_boxplots.py` plots them per scenario/sample-size with FNN and RF side by side. The `"Standard Deviation"` column of the ROC-curve-MSE table is named `"MSE SD Across Runs"` so it is never confused with the differently-defined `Std-Function MSE`.

**Known caveat -- Scenario 7's healthy arm, Mean-Function MSE.** Scenario 7's healthy (D=0) arm has no closed-form mean (skew-normal/Student-t mixture -- see `true_dgp.true_mean`'s docstring), so `data_generation.py`'s `True_Mean_Y` placeholder for that arm is a single Monte Carlo draw from the conditional distribution rather than an actual mean. `Mean-Function MSE` there therefore measures deviation from one noisy sample, not from a true expectation, and comes out roughly one to two orders of magnitude larger than every other scenario/group's (e.g. ~0.12 vs. ~0.0007-0.03 elsewhere) -- an artifact of the metric's definition for this one arm, not a modeling problem. It does not affect the paper's primary ROC-curve/AUC comparison, which handles Scenario 7 correctly via large-sample Monte Carlo in `ground_truth_auc.py`. If reporting this table, footnote Scenario 7's healthy-arm `Mean-Function MSE` as not comparable to the other rows.

**Simulation scenario definitions.** Per the Supplementary Material, covariates are `U(-1, 1)` in every scenario; the code previously drew most of them from `Normal(0, 1)` (Scenarios I-VI, VIII) or `U(0, 1)` (Scenario VII) instead -- now fixed in `data_generation.py`, with two deliberate, documented exceptions: Scenario VI's third covariate stays `Bernoulli(0.5)` (a 0/1 interaction switch, not a continuous effect) and Scenario VII's covariate stays `U(0, 1)` (its mixture weight `exp(-2x)` is only a valid probability for `x >= 0`). Separately, Scenario III's formula in the Supplementary Material -- titled "Covariate Effect on Both Mean and Variance" -- had constant variance; `true_dgp.py` now defines `sigma(x) = base * exp(0.3*x)` for that scenario. **Both changes alter the generated data**; the full 9-scenario x 2-sample-size x 100-replicate sweep has since been rerun against the corrected data-generating process.

**Case study.** The real-data experiments evaluate total activity count (TAC), a proxy for daily step count derived from NHANES 2011-2014 accelerometry (MIMS units), as a biomarker for all-cause mortality at 3-, 5-, and 8-year horizons. Age and BMI are the covariates and **the analysis is stratified by sex**: separate models are fitted for the 2,385 women and the 2,621 men, and sex is never an input to any model (`n = 5,006` in total; see Table 1 of the paper). Group/target variables are named in Spanish: `tres` (3-year), `cinco` (5-year) and `ocho` (8-year) mortality status, plus `mortstat` for overall mortality.

**Which biomarker column to use.** `data/df_f.csv` and `data/df_m.csv` each contain two columns, `TAC` and `TAC2`. **Every number in the paper uses `TAC2`** (female mean 680.75, male mean 735.04 -- Table 1). The column literally named `TAC` is a different quantity on a different scale (female median ~12,469 against ~660 for `TAC2`) and is *not* a monotone transform of it (Spearman correlation ~0.5), so the two do not give the same AUC. The `df_f_full.csv`/`df_m_full.csv` variants carry the same rows plus the NHANES survey-design columns (`wtmec4yr_adj`, `SDMVPSU`, `SDMVSTRA`), merged by `src/real_data/merge_survey_design.py`; they are what the survey-weighted analysis reads.

## Repository structure

```
data/                    CSV datasets (NHANES-derived, public/anonymized)
                         df_f.csv / df_m.csv            sex-stratified extracts; biomarker column TAC2
                         df_f_full.csv / df_m_full.csv  same rows plus the survey-design columns
                         data_0.csv is NOT included: it is the 33 MB raw NHANES extract that
                         merge_survey_design.py reads to build the _full variants. Those are
                         committed directly, so the pipeline runs without it; the design
                         variables can be obtained from NHANES if you want to rebuild them.
src/
  simulation/             Simulated-data pipeline (wired into hpc/ SLURM scripts)
    data_generation.py            generates the 9 simulation scenarios (Scenarios I-IX, Supplementary
                                   Material) with linear, non-linear, and interaction covariate effects
    true_dgp.py                    single source of truth for each scenario's true conditional
                                   mean/std/sampling distribution
    ground_truth_auc.py            true (population-level) aROC(p|x)/AUC per scenario
    eval_io.py                      shared roc_mse_values.csv/timing.csv writer used by the baselines
    data_simulation_reg.py        dataset loader for the simulation pipeline
    mlp_reg_data_simulation_multi.py   two-stage FNN mean/variance regression + aROC/AUC vs. ground truth
    rf_reg_data_simulation_multi_2.py  same two-stage pipeline with a Random Forest regressor
    naive_roc_baseline.py          naive/pooled ROC baseline (no covariate adjustment) vs. ground truth
    linear_reg_data_simulation.py  linear and semiparametric-additive (spline) aROC baselines
    sandwich_delta_ci.py           fit_single(): one MLP on the full sample, used by the local-kNN
                                   pipelines; also a sandwich/delta CI experiment (not in the paper)
    coverage_local_knn.py          CURRENT inference procedure: local kNN AUC, DeLong variance with
                                   Wilson fallback, Hajek-weighted variants, pointwise coverage per
                                   scenario against the ground-truth AUC
    bootstrap_crossfit_oob.py      SUPERSEDED bootstrap + K-fold cross-fitting + OOB engine
    coverage_bootstrap_crossfit.py SUPERSEDED pointwise bootstrap-OOB coverage
    rescore_scenario7_orientation.py  re-scores stored replicate outputs against the re-oriented
                                   Scenario VII ground truth, without refitting anything
    imbalance_small_n_experiment.py  n=500 with a 90/10 split, Scenarios I and IV
  real_data/              Pipeline on real (NHANES-derived) data
    data_reg_real.py              dataset loader and column normalization
    merge_survey_design.py        merges the NHANES survey-design columns into df_*_full.csv
    local_knn_bands_real.py       CURRENT case-study entry point: two-stage FNN with 5-fold
                                   cross-fitting, local kNN AUC and intervals, weighted and
                                   unweighted; writes bands_<horizon>_<sex>.csv
    plot_nhanes_bands.py          every NHANES figure in the paper and Supplementary
    linear_aroc_real.py           linear covariate-adjusted ROC benchmark (Supplementary Table 9)
    naive_roc_baseline.py         naive pooled ROC/AUC benchmark (Supplementary Table 10)
    select_k_per_group.py         outcome-blind choice of k per arm from neighbourhood age spread
    summarize_bias_experiment.py  aggregates a multi-seed sweep over one training hyperparameter
                                   (the learning-rate sweep behind the case study's lr)
    mlp_reg.py                    SUPERSEDED original-submission FNN pipeline. Its variance stage is
                                   defective: it regresses on log(r^4) although its own docstring says
                                   log(r^2), recovers sigma with sqrt() rather than sqrt(exp(.)), and
                                   returns sigma = 0 whenever the prediction is negative. No figure or
                                   table in the paper comes from it. Kept for provenance only.
  classification/         Classification pipeline with temperature scaling
    data.py, models.py, losses.py, metrics.py, temperature_scaling.py, train.py
    croc_nn_baseline.py            NN-based cROC baseline (uses train.cv_loop)
  baselines/
    croc_linear_baseline.py        semiparametric linear-regression cROC estimator (cROC_sp/compute_ROC),
                                    Python port of ROCnReg::cROC.sp, used both for the NHANES case study
                                    and for the simulation-scenario baselines
  postprocessing/
    statistics_summary.py          aggregate MSE, timing, mean-/std-function MSE and coverage statistics
    mean_std_mse_boxplots.py       per-scenario Mean-/Std-Function MSE boxplots, methods side by side
    write_latex_table.py           per-scenario, per-method ROC-MSE comparison table (.tex/.dat)
R/
  aroc_batch_scenarios.R    AROC.sp/cROC.sp over all scenario CSVs in input_real_2/
  aroc_crude_vs_adjusted.R  crude vs. confounder-adjusted ROC comparison
  aroc_single_model.R       single AROC.sp model + NHANES data prep
  aroc_bnp_timing.R         ROCnReg::AROC.bnp wall-clock timing, one replicate per scenario
  croc_sp_validation.R      reference cROC.sp values for the Python-port check
notebooks/
  nhanes_hetero_residuos.ipynb  SUPERSEDED. Bootstrap + cross-fitting + OOB intervals for the
                                 NHANES case study. The case study in the paper is produced
                                 by local_knn_bands_real.py.
hpc/
  py_gen.sh, py_mlp.sh, py_rfo.sh, py_post.sh         SLURM job scripts (submitted via sbatch)
  py_mlp_array.sh                                      array version, one replicate per task
  py_localknn.sh, run_localknn_all.sh                  local-kNN coverage sweep (current procedure)
  py_knn_bands_real.sh                                 NHANES bands, all six horizon/sex combinations
  py_cov.sh, py_cov_array.sh, run_cov*.sh              SUPERSEDED bootstrap-OOB coverage
  py_lr_sweep_array.sh, run_lr_sweep_array.sh          learning-rate sweep behind the case study's lr
  combine_outputs.sh                                   symlinks each method's folders into one root
archive/
  superseded prototypes and early drafts, kept for reference (see archive/README.md)
```

## Setup

### Python

```bash
conda env create -f environment.yml
conda activate covroc
```

`environment.yml` covers the Python pipeline only. The `R/` scripts additionally need `ROCnReg`, `np` and their dependencies, installed separately from CRAN.

### R

```r
install.packages(c("ROCnReg", "np"))
```

## Running the pipeline

All commands are run from the repository root.

**1. Generate simulated data** (writes to `input_real_2/`):

```bash
./hpc/run_gen.sh
```

**2. Train/evaluate every method** on a scenario folder, each scored against the true aROC(p|x) and timed:

```bash
./hpc/run_mlp_array.sh input_real_2/scenario_1 output/fnn/scenario_1
./hpc/run_rfo.sh       input_real_2/scenario_1 output/rf/scenario_1
python src/simulation/naive_roc_baseline.py      -i input_real_2/scenario_1 -o output/naive/scenario_1
python src/simulation/linear_reg_data_simulation.py -i input_real_2/scenario_1 -o output/linear/scenario_1
```

**3. Pointwise coverage of the local-kNN intervals** (the procedure reported in the paper), per scenario folder:

```bash
./hpc/run_localknn_all.sh input_real_2 output/knn
```

Defaults are a 21-point fixed covariate grid, `k in {5,10,15,20,50,100,300,1000}`, nominal levels 0.95/0.80/0.50, and model-space localization; `-loc covariate` switches to the raw covariate space. A fixed grid rather than subject targets is used so that each grid point contributes one independent Bernoulli trial per replicate.

**4. Real-data case study** -- two-stage FNN, local kNN bands, all six horizon/sex combinations, unweighted and survey-weighted:

```bash
python src/real_data/local_knn_bands_real.py --all -k 30 -o output/knn_bands_real
python src/real_data/local_knn_bands_real.py --all -k 30 -o output/knn_bands_real_w -fs _full --band_weights
```

`-fs _full` reads `df_*_full.csv` (the survey-design variants) and `--band_weights` applies the weights to the local statistic as well as to the networks. `-k` accepts either a single value per arm or `k0:k1` pairs.

**5. Case-study benchmarks** (Supplementary Tables 9 and 10):

```bash
python src/real_data/linear_aroc_real.py  -o output/knn_bands_real/linear_aroc_real.csv
python src/real_data/naive_roc_baseline.py -i data -o output/knn_bands_real
```

**6. Figures.** Each figure has its own flag; the script draws nothing without one. `python src/real_data/plot_nhanes_bands.py -h` lists them all. For example:

```bash
python src/real_data/plot_nhanes_bands.py -i output/knn_bands_real -d data \
    -o output/knn_bands_real --per_group --subject_scatter
```

**7. Combine and post-process.** Steps 2 and 3 write each method into its own `-o`; `combine_outputs.sh` symlinks them into one method-prefixed root, which the aggregators expect:

```bash
./hpc/combine_outputs.sh output/combined output/fnn output/rf output/naive output/linear output/knn
./hpc/run_post.sh output/combined
```

**8. cROC baselines and the R comparison:**

```bash
python src/baselines/croc_linear_baseline.py -i data -o output_real
Rscript R/aroc_single_model.R
```

## Notes

- The `R/` scripts assume they are run from the repository root. `aroc_single_model.R` references a `data_analysis_TD_2003_19.rda` file that is not included here.
- Directories produced at runtime (`input_real_2/`, `output/`, `output_real/`, etc.) are git-ignored.
- `archive/` contains earlier/superseded versions of scripts, kept for reference only -- see `archive/README.md`.
