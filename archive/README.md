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
