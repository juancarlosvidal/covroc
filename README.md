# covroc

Code accompanying the paper:

> [PAPER TITLE] — [AUTHORS] — [JOURNAL / VENUE, YEAR]
> [DOI / LINK]
>
> If you use this code, please cite: [CITATION]

This repository implements and compares covariate-adjusted ROC (cROC/AROC) curve estimation methods: an MLP-based regression approach, a Random Forest regression approach, an NN-based cROC baseline, a linear-regression baseline, and the semiparametric `AROC.sp`/`cROC.sp` estimators from the R package [`ROCnReg`](https://cran.r-project.org/package=ROCnReg). It also includes a classification pipeline with temperature scaling used in the paper.

The real-data experiments use variables derived from NHANES (`mortstat`, `RIDAGEYR`/Age, `BMI`, `Cancer`), included under `data/`.

## Repository structure

```
data/                    CSV datasets (NHANES-derived, public/anonymized)
src/
  simulation/             Simulated-data pipeline (wired into hpc/ SLURM scripts)
    data_generation.py            generates the 9 simulation scenarios
    data_simulation_reg.py        dataset loader for the simulation pipeline
    mlp_reg_data_simulation_multi.py   MLP regression training/eval entrypoint
    rf_reg_data_simulation_multi_2.py  Random Forest regression training/eval entrypoint
  real_data/              Regression pipeline on real (NHANES-derived) data
    data_reg_real.py
    mlp_reg.py
  classification/         Classification pipeline with temperature scaling
    data.py, models.py, losses.py, metrics.py, temperature_scaling.py, train.py
    croc_nn_baseline.py            NN-based cROC baseline (uses train.cv_loop)
  baselines/
    croc_linear_baseline.py        linear-regression (statsmodels) cROC baseline
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
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

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
