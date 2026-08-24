# Validates that src/baselines/croc_linear_baseline.py's cROC_sp is a faithful port of
# ROCnReg::cROC.sp (Reviewer 1, Major Concern 2's semiparametric baseline, already run at
# scale for the 9 simulation scenarios via src/simulation/linear_reg_data_simulation.py).
# Not a new benchmark result for the response letter -- this confirms the two
# implementations' point estimates agree on the same data, so linear_reg_data_simulation.py's
# already-run results can be trusted without also running the real R package at scale
# (see the README's note on why R/aroc_batch_scenarios.R was never wired into the
# simulation study: wrong directory layout and real-data-only formula/columns).
#
# cROC.sp only supports one shared `newdata` (no newdata.h/newdata.d split, confirmed via
# args(cROC.sp)) -- unlike the Python port's row-paired newdata_h/newdata_d extension used
# for the ground-truth comparison, this validation uses newdata = df (every subject's own
# covariates, regardless of their actual group), the one mode both implementations support.
#
# Run from the repository root, e.g.:
#   Rscript R/croc_sp_validation.R input_real_2/scenario_1/scenario_1_5000_1_data.csv x_D_1
library(ROCnReg)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript croc_sp_validation.R <scenario_replicate.csv> <covariate1> [covariate2 ...]\n",
       "  (the covariate names must match true_dgp.covariate_columns(scenario) in the Python code)")
}
csv_file <- args[1]
covariates <- args[2:length(args)]

df <- read.csv(csv_file)
formula_str <- paste("Y_generated ~", paste(covariates, collapse = " + "))
formula_obj <- as.formula(formula_str)

cat(sprintf(
  "File: %s\nFormula: %s\nN: %d (healthy=%d, diseased=%d)\n\n",
  csv_file, formula_str, nrow(df), sum(df$mortstat == 0), sum(df$mortstat == 1)
))

# B = 0: point estimate only, matching how linear_reg_data_simulation.py calls the
# Python port (no bootstrap CI needed for this comparison).
model <- cROC.sp(
  formula.h = formula_obj,
  formula.d = formula_obj,
  group = "mortstat",
  tag.h = 0,
  data = df,
  newdata = df,
  p = seq(0, 1, length.out = 101),
  B = 0
)

scenario_name <- tools::file_path_sans_ext(basename(csv_file))
out_dir <- "output/croc_sp_validation"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# AUC is a 3-column [est, ql, qh] matrix; ql/qh are NA since B = 0.
auc_out <- data.frame(Row = seq_len(nrow(df)), AUC = model$AUC[, 1])
auc_file <- file.path(out_dir, paste0(scenario_name, "_auc_R.csv"))
write.csv(auc_out, auc_file, row.names = FALSE)

# Full ROC(p) curve for the first row only, to compare curve shape, not just the AUC scalar.
roc_out <- data.frame(p = model$p, ROC = model$ROC$est[1, ])
roc_file <- file.path(out_dir, paste0(scenario_name, "_roc_row1_R.csv"))
write.csv(roc_out, roc_file, row.names = FALSE)

cat(sprintf("Wrote %s and %s\n", auc_file, roc_file))
