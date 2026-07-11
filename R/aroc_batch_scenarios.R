
# Runs the semiparametric AROC.sp/cROC.sp estimators (Rodriguez-Alvarez, ROCnReg) on
# every scenario/real-data CSV in input_real_2/, i.e. the kernel-smoothing benchmark
# that the paper's FNN-based location-scale estimator is compared against. Also
# produces the 1-AUC-vs-age scatter and the BMI x age 3D AUC surface analogous to
# Figures 2-4 (there computed with the FNN and Random Forest estimators instead).
#%%
library('ROCnReg')
library('plotly')
library('htmlwidgets')  # To save interactive HTML files
library('webshot2')     # To save plots as PNG

# Set the directory containing the CSV files
# Run this script from the repository root. input_real_2/ is produced by
# src/simulation/data_generation.py (see hpc/py_gen.sh).
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

