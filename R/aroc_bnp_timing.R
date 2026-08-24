# Reports ROCnReg::AROC.bnp's wall-clock computation time on a single simulation-scenario
# replicate (Reviewer 1, Major Concern 2). The Bayesian nonparametric AROC estimator was
# declined for a full Python reimplementation -- no equivalent in the Python ecosystem --
# so the response letter instead reports its cost on one representative replicate,
# alongside the FNN/RF/naive/linear/spline timings already collected in
# timing_summary.csv (see src/simulation/*_reg_data_simulation*.py /
# naive_roc_baseline.py). Not a full comparison sweep -- see the README's Post-processing
# section for that.
#
# Uses data_generation.py's actual scenario CSV columns (Y_generated, x_D_*, mortstat),
# NOT the real-data NHANES columns (TAC/BMI/RIDAGEYR) the other R/ scripts use.
#
# Run from the repository root, e.g.:
#   Rscript R/aroc_bnp_timing.R input_real_2/scenario_1/scenario_1_5000_1_data.csv x_D_1
#   Rscript R/aroc_bnp_timing.R input_real_2/scenario_5/scenario_5_5000_1_data.csv x_D_1 x_D_2
#   Rscript R/aroc_bnp_timing.R input_real_2/scenario_8/scenario_8_5000_1_data.csv x_D_1 x_D_6 x_D_7 x_D_8
library(ROCnReg)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript aroc_bnp_timing.R <scenario_replicate.csv> <covariate1> [covariate2 ...]\n",
       "  (the covariate names must match true_dgp.covariate_columns(scenario) in the Python code)")
}
csv_file <- args[1]
covariates <- args[2:length(args)]

df <- read.csv(csv_file)
formula_str <- paste("Y_generated ~", paste(covariates, collapse = " + "))
cat(sprintf(
  "File: %s\nFormula: %s\nN: %d (healthy=%d, diseased=%d)\n\n",
  csv_file, formula_str, nrow(df), sum(df$mortstat == 0), sum(df$mortstat == 1)
))

# Verified against ROCnReg 1.0-9's actual AROC.bnp signature (formula.h, group, tag.h,
# data, ..., p, ...) -- unlike AROC.sp/cROC.sp used elsewhere in R/, the first argument
# is named formula.h, not formula (only argument that starts with "formula" here, so
# formula = ... would partial-match fine too, but spelled out explicitly to not rely on
# that). Every other argument (standardise, ci.level, prior.h, mcmc, ...) is left at its
# default, matching how AROC.sp is called elsewhere in R/ for the timing comparison.
elapsed <- system.time({
  aroc_bnp_model <- AROC.bnp(
    formula.h = as.formula(formula_str),
    group = "mortstat",
    tag.h = 0,
    data = df,
    p = seq(0, 1, length.out = 101)
  )
})

cat("--- AROC.bnp timing ---\n")
print(elapsed)
cat(sprintf("\nElapsed (wall clock) seconds: %.3f\n", elapsed["elapsed"]))

# Same Scenario/Method/Fit Seconds Total shape timing.csv already uses elsewhere in the
# pipeline (src/simulation/eval_io.py), so this can be dropped into the same comparison
# if wanted, without a bespoke format just for this one method.
scenario_name <- tools::file_path_sans_ext(basename(csv_file))
out <- data.frame(
  Scenario = scenario_name, Method = "AROC.bnp",
  `Fit Seconds Total` = unname(elapsed["elapsed"]), check.names = FALSE
)
out_dir <- "output/aroc_bnp_timing"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_file <- file.path(out_dir, paste0(scenario_name, "_timing.csv"))
write.csv(out, out_file, row.names = FALSE)
cat(sprintf("\nWrote %s\n", out_file))
