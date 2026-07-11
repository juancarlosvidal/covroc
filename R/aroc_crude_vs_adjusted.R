# Illustrates the paper's central motivation (Introduction / The Need for
# Covariate-Adjusted ROC Analysis): the "crude" pooled ROC curve for Age vs.
# mortality ignores covariates and can misrepresent discriminative performance,
# whereas standardizing Age against confounders (Cancer, BMI) before computing the
# adjusted ROC curve corrects for this population heterogeneity.
# Load necessary libraries
library(splines)  # For basis splines in confounder adjustment

# Step 1: Read the dataset
# Run this script from the repository root.
data_path <- "data/data_analysis_TD_2003_19.csv"
data_real <- read.csv(data_path)

# Check the first few rows
head(data_real)

# Step 2: Crude ROC curve (ignoring confounders)
roc_crude <- adjusted.ROC(status = "mortstat",
                          variable = "Age",
                          confounders = ~1,  # No confounder adjustment
                          database = data_real,
                          precision = seq(0.1, 0.9, by = 0.1))

# Step 3: Standardize the Age variable based on confounders (Cancer, BMI)
lm_std <- lm(Age ~ Cancer + BMI, data = data_real[data_real$mortstat == 0, ])
data_real$Age_std <- (data_real$Age - (lm_std$coef[1] +
                                         lm_std$coef[2] * data_real$Cancer +
                                         lm_std$coef[3] * data_real$BMI)) / sd(lm_std$residuals)

# Step 4: Adjusted ROC curve (with confounders Cancer and BMI)
roc_adjusted <- adjusted.ROC(status = "mortstat",
                             variable = "Age_std",
                             confounders = ~ bs(Cancer, df = 3) + BMI,
                             database = data_real,
                             precision = seq(0.1, 0.9, by = 0.1))

# Step 5: Plot the ROC curves
plot(1 - roc_crude$table$sp, roc_crude$table$se, 
     ylim = c(0, 1), xlim = c(0, 1), 
     ylab = "Sensitivity", xlab = "1 - Specificity", 
     type = "l", lty = 1, col = 1, lwd = 2, 
     main = "Crude vs Adjusted ROC Curves")

lines(1 - roc_adjusted$table$sp, roc_adjusted$table$se, 
      col = 2, lwd = 2, lty = 2)

abline(0, 1, lty = 2, col = "gray")

legend("bottomright", 
       lty = 1:2, lwd = 2, col = 1:2, 
       legend = c(paste("Crude estimation, (AUC=", round(roc_crude$auc, 2), ")", sep=""),
                  paste("Adjusted estimation, (AUC=", round(roc_adjusted$auc, 2), ")", sep="")))
