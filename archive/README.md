# Archive

Earlier prototypes and superseded script versions, kept for reference / provenance. Nothing here is maintained or guaranteed to run — the active code lives under `src/`, `R/`, and `hpc/`.

| File | Why it's archived |
|---|---|
| `data_simulation.py` | Earlier version of `src/simulation/data_generation.py`; the latter is the one used by `hpc/py_gen.sh` and covers more scenarios/generations. |
| `similation.py`, `similation_old.py` | Ad-hoc scripts for building a simulation grid, from before `data_generation.py` covered scenario generation. |
| `mlp_reg_data_simulation.py` | Earlier version of `src/simulation/mlp_reg_data_simulation_multi.py`; the "multi" version is the one used by `hpc/py_mlp.sh`. |
| `rf_reg_data_simulation_multi.py` | Earlier version (v1) of `src/simulation/rf_reg_data_simulation_multi_2.py`, which is the one used by `hpc/py_rfo.sh`. |
| `test.py`, `test_nn.py`, `test_rf.py` | Early prototypes of the MLP/RF regression pipelines. Despite the name, these are not unit tests — they import from a loader (`data.py`) that predates `data_reg_real.py`/`data_simulation_reg.py`, and were superseded by the pipelines in `src/simulation/` and `src/real_data/`. |
| `data2.py`, `data_cla_real.py` | Dataset loader variants not imported by any active script. |
| `convert_to_wide.py` | Reshaped `data_simulation.py`'s old long-form scenario CSVs (paired healthy `_bar`/diseased columns in one row) into wide, `mortstat`-labeled form. `data_generation.py` now writes wide form directly to `input_real_2/`, so this step -- and the `data_simulation/`/`data_simulation_wide/` directories it read from and wrote to -- is no longer part of the pipeline; nothing downstream reads its output. |
| `coverage_sandwich.py`, `bootstrap_crossfit_parametric.py`, `coverage_subsampling.py` | Alternative routes to a confidence interval for the covariate-specific AUC (a sandwich/delta construction, a parametric/Gaussian bootstrap, and subsampling), explored alongside the bootstrap-OOB procedure. None is used in the paper, which reports the local-kNN intervals of `src/simulation/coverage_local_knn.py`. Their SLURM wrappers (`py_sandwich*.sh`, `run_sandwich*.sh`, `py_subsampling_array.sh`, `run_subsampling_array.sh`) are archived with them. |
| `variance_decomposition_diagnostic.py`, `py_variance_diag.sh` | Diagnostic that split the bootstrap-OOB interval width into its estimation and resampling components. Served its purpose during the development of the superseded procedure; nothing in the paper depends on it. |
| `coverage_band_plot.py`, `py_coverage_band_plots.sh`, `run_coverage_band_plots.sh` | Plotted the bootstrap-OOB intervals of one representative replicate per scenario. The figures it produced do not appear in the paper, which reports coverage as a table rather than as per-replicate bands. |
