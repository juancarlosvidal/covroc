#%%
# Semiparametric AROC.sp benchmark (BMI adjusted for Age and Cancer status, group =
# mortstat) -- the kernel-smoothing counterpart the paper's FNN estimator is
# compared against for a single covariate combination.
# Run this script from the repository root.
library('ROCnReg')
data.path <- "data/data_analysis_TD_2003_19.csv"
df <- read.csv(data.path)

aroc_model <- AROC.sp (formula=BMI ~ Age + Cancer ,
                        group = 'mortstat',
                        tag.h  = 0 ,
                        data = df,
                        p  = seq(0, 1, length.out = 101))
plot(aroc_model)

#%%
# NHANES data prep for the TAC-mortality case study (Results from the NHANES
# 2011-2014 Dataset): collapses the per-minute activity matrix (act_mat) into the
# total activity count (TAC) biomarker used throughout the paper, then writes the
# pooled dataset plus Cancer/Gender-stratified subsets consumed by the batch
# AROC.sp/cROC.sp runs below and by the Python pipelines in src/real_data/.
# Requires data_analysis_TD_2003_19.rda (not included in this repo) placed under data/.
load('data/data_analysis_TD_2003_19.rda')

library(dplyr)

data_analysis$act_mat <- apply (data_analysis$act_mat , 1 , sum)
data_analysis <- data_analysis %>% rename(TAC = act_mat)

write.csv(data_analysis[,c("mortstat", "RIDAGEYR", "BMI", "TAC")], "input_real_2/todos_datos.csv", row.names = FALSE)

# Filter rows where Cancer is "Yes" and Gender is "Male"
filtered_data_male <- data_analysis[data_analysis$Cancer == "Yes" & data_analysis$Gender == "Male",
                               c("mortstat", "RIDAGEYR", "BMI", "TAC")]

# Save the filtered data to a CSV file
write.csv(filtered_data_male, "input_real_2/male_data_con_cancer.csv", row.names = FALSE)

# Filter rows where Cancer is "Yes" and Gender is "Female"
filtered_data_female <- data_analysis[data_analysis$Cancer == "Yes" & data_analysis$Gender == "Female",
                               c("mortstat", "RIDAGEYR", "BMI", "TAC")]

# Save the filtered data to a CSV file
write.csv(filtered_data_female, "input_real_2/female_data_con_cancer.csv", row.names = FALSE)

# Filter rows where Cancer is "No" and Gender is "Male"
filtered_data_male <- data_analysis[data_analysis$Cancer == "No" & data_analysis$Gender == "Male",
                               c("mortstat", "RIDAGEYR", "BMI", "TAC")]

# Save the filtered data to a CSV file
write.csv(filtered_data_male, "input_real_2/male_data_sin_cancer.csv", row.names = FALSE)

# Filter rows where Cancer is "No" and Gender is "Female"
filtered_data_female <- data_analysis[data_analysis$Cancer == "No" & data_analysis$Gender == "Female",
                               c("mortstat", "RIDAGEYR", "BMI", "TAC")]

# Save the filtered data to a CSV file
write.csv(filtered_data_female, "input_real_2/female_data_sin_cancer.csv", row.names = FALSE)



# AROC.sp/cROC.sp batch run over the Cancer/Gender-stratified TAC subsets written
# above -- the semiparametric benchmark for the TAC~BMI+RIDAGEYR aROC surfaces.
#%%
library('ROCnReg')
library('plotly')
library('htmlwidgets')  # To save interactive HTML files
library('webshot2')     # To save plots as PNG

# Set the directory containing the CSV files (run from repository root)
input_dir <- "input_real_2/"
output_dir <- "output/"

# List all CSV files in the input directory
csv_files <- list.files(input_dir, pattern = "\\.csv$", full.names = TRUE)

# Loop through each CSV file
for (csv_file in csv_files) {
  # Extract the base name of the file (without directory and extension)
  file_name <- tools::file_path_sans_ext(basename(csv_file))

  # Create a subdirectory for this file's plots
  file_output_dir <- file.path(output_dir, file_name)
  if (!dir.exists(file_output_dir)) {
    dir.create(file_output_dir, recursive = TRUE)
  }

  # Read the CSV file
  df <- read.csv(csv_file)

  # Create an AROC.sp model
  aroc_model <- AROC.sp(
    formula = TAC ~ BMI + RIDAGEYR,
    group = 'mortstat',
    tag.h = 0,
    data = df,
    p = seq(0, 1, length.out = 101)
  )
  
  # Save the AROC.sp plot
  png(file.path(file_output_dir, paste0(file_name, "_AROC.sp_model_plot.png")), width = 800, height = 600)
  plot(aroc_model, main = paste("AROC.sp Model for", file_name, "Data in R"))
  dev.off()
  
  # Create a cROC.sp model
  cROC_model <- cROC.sp(
    formula.h = TAC ~ BMI + RIDAGEYR,
    formula.d = TAC ~ BMI + RIDAGEYR,
    group = 'mortstat',
    tag.h = 0,
    data = df,
    p = seq(0, 1, length.out = 101),
    B = 10,
    newdata = df
  )
  
  # Save the cROC.sp plot
  png(file.path(file_output_dir, paste0(file_name, "_cROC.sp_model_plot.png")), width = 800, height = 600)
  plot(df$RIDAGEYR, 1 - cROC_model$AUC[,1], main = paste("cROC.sp Model for", file_name, "Data in R"))
  dev.off()
  
  # Create the 3D surface plot
  y <- df$RIDAGEYR
  x <- df$BMI
  z_length <- length(cROC_model$AUC[,1])
  x_length <- length(unique(x))
  y_length <- length(unique(y))
  
  # Ensure z matrix matches dimensions of unique x and y
  z <- matrix(1 - cROC_model$AUC[,1][1:(x_length * y_length)], 
              nrow = x_length, 
              ncol = y_length, 
              byrow = TRUE)
  
  # Create 3D Surface Plot
  fig <- plot_ly(
    x = ~unique(x),
    y = ~unique(y),
    z = ~z,
    type = "surface",
    colorscale = "Viridis"
  ) %>%
    layout(
      scene = list(
        xaxis = list(title = "BMI", autorange = "reversed"),
        yaxis = list(title = "RIDAGEYR"),
        zaxis = list(title = "1 - AUC")
      ),
      title = paste("3D Surface Plot for", file_name, "Data")
    )

  # Save the plot as an interactive HTML file (not self-contained)
  html_file <- file.path(file_output_dir, paste0(file_name, "_3D_plot.html"))
  htmlwidgets::saveWidget(fig, html_file, selfcontained = FALSE)

  # Save the plot as a static PNG
  png_file <- file.path(file_output_dir, paste0(file_name, "_3D_plot.png"))
  webshot2::webshot(html_file, png_file)
}
 

# Same AROC.sp/cROC.sp batch run, generalized to the nine simulation-scenario CSVs
# (Y_generated ~ all covariate columns) -- the semiparametric benchmark used in the
# finite-sample comparison against the FNN and Random Forest estimators.
#%%

library('ROCnReg')
library('plotly')
library('htmlwidgets')  # To save interactive HTML files
library('webshot2')     # To save plots as PNG

# Set the directory containing the CSV files (run from repository root)
input_dir <- "input_real_2/"
output_dir <- "output/"

# List all CSV files in the input directory
csv_files <- list.files(input_dir, pattern = "\\.csv$", full.names = TRUE)

# Loop through each CSV file
for (csv_file in csv_files) {
  # Extract the base name of the file (without directory and extension)
  file_name <- tools::file_path_sans_ext(basename(csv_file))

  # Create a subdirectory for this file's plots
  file_output_dir <- file.path(output_dir, file_name)
  if (!dir.exists(file_output_dir)) {
    dir.create(file_output_dir, recursive = TRUE)
  }

  # Read the CSV file
  df <- read.csv(csv_file)

  # Drop the 'True_Mean_Y' column if it exists
  if ("True_Mean_Y" %in% names(df)) {
    df <- df[, !names(df) %in% "True_Mean_Y"]
  }
  
  # Identify target and feature columns
  target <- "Y_generated"
  features <- setdiff(names(df), c(target, "mortstat"))
  
  # Use the first and second columns for the plots
  x_col <- features[1]
  y_col <- features[2]

  # Create an AROC.sp model
  aroc_model <- AROC.sp(
    formula = as.formula(paste(target, "~", paste(features, collapse = "+"))),
    group = 'mortstat',
    tag.h = 0,
    data = df,
    p = seq(0, 1, length.out = 101)
  )
  
  # Save the AROC.sp plot
  png(file.path(file_output_dir, paste0(file_name, "_AROC.sp_model_plot.png")), width = 800, height = 600)
  plot(aroc_model, main = paste("AROC.sp Model for", file_name, "Data in R"))
  dev.off()
  
  # Create a cROC.sp model
  cROC_model <- cROC.sp(
    formula.h = as.formula(paste(target, "~", paste(features, collapse = "+"))),
    formula.d = as.formula(paste(target, "~", paste(features, collapse = "+"))),
    group = 'mortstat',
    tag.h = 0,
    data = df,
    p = seq(0, 1, length.out = 101),
    B = 10,
    newdata = df
  )

  # Save the cROC.sp plot
  png(file.path(file_output_dir, paste0(file_name, "_cROC.sp_model_plot.png")), width = 800, height = 600)
  plot(df[[y_col]], 1 - cROC_model$AUC[,1], main = paste("cROC.sp Model for", file_name, "Data in R"))
  dev.off()

  # Create the 3D surface plot
  x <- df[[x_col]]
  y <- df[[y_col]]
  z_length <- length(cROC_model$AUC[,1])
  x_length <- length(unique(x))
  y_length <- length(unique(y))

  # Ensure z matrix matches dimensions of unique x and y
  z <- matrix(1 - cROC_model$AUC[,1][1:(x_length * y_length)], 
              nrow = x_length, 
              ncol = y_length, 
              byrow = TRUE)

  # Create 3D Surface Plot
  fig <- plot_ly(
    x = ~unique(x),
    y = ~unique(y),
    z = ~z,
    type = "surface",
    colorscale = "Viridis"
  ) %>%
    layout(
      scene = list(
        xaxis = list(title = x_col, autorange = "reversed"),
        yaxis = list(title = y_col),
        zaxis = list(title = "1 - AUC")
      ),
      title = paste("3D Surface Plot for", file_name, "Data")
    )

  # Save the plot as an interactive HTML file (not self-contained)
  html_file <- file.path(file_output_dir, paste0(file_name, "_3D_plot.html"))
  htmlwidgets::saveWidget(fig, html_file, selfcontained = FALSE)

  # Save the plot as a static PNG
  png_file <- file.path(file_output_dir, paste0(file_name, "_3D_plot.png"))
  webshot2::webshot(html_file, png_file)
}
