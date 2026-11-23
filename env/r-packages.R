# R packages for 00_data_preparation/, 01_geobia/ and 07_figures/
# Run once:  source("env/r-packages.R")

pkgs <- c(
  # spatial
  "terra", "sf", "stars", "exactextractr", "raster",
  # modelling
  "caret", "randomForest", "e1071", "nnet", "naivebayes", "pROC",
  # texture / segmentation
  "glcm", "landscapemetrics",
  # data handling and figures
  "dplyr", "tidyr", "readr", "purrr", "stringr", "tibble",
  "ggplot2", "patchwork", "scales", "viridis",
  # reporting
  "knitr", "kableExtra"
)

missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}
invisible(lapply(pkgs, library, character.only = TRUE))
cat("R environment ready:", length(pkgs), "packages\n")
