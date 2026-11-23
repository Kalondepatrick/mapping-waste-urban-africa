# --- 1. SETUP & DATA LOADING ---
library(sf)
library(caret)
library(dplyr)
library(purrr)
library(tidyr)

# Paths
model_obj    <- readRDS("01_geobia/modelsRF.rds")
rf_model     <- model_obj$model
folder_path  <- "data/tiled_data/partitioned/Validation/vectors_with_var/"
strata_csv   <- "06_evaluation/04_accuracy_by_landcover/waste_context_summary.csv"

# Load Strata Data
strata_data <- read.csv(strata_csv)

# Identify strata columns (everything starting with "touches_")
strata_cols <- names(strata_data)[grep("^touches_", names(strata_data))]

# Load and Combine all shapefiles
files <- list.files(folder_path, pattern = "\\.shp$", full.names = TRUE)

full_dataset <- files %>%
  map_df(~ {
    ds <- st_read(.x, quiet = TRUE) %>% st_drop_geometry()
    ds$filename <- gsub("\\.shp$", ".tif", basename(.x))
    return(ds)
  })

# Join Strata to Dataset
full_dataset <- full_dataset %>%
  left_join(strata_data, by = "filename")

# Run Predictions
class_levels <- rf_model$levels
full_dataset$predictions <- predict(rf_model, newdata = full_dataset)
full_dataset$predictions <- factor(full_dataset$predictions, levels = class_levels)
full_dataset$class       <- factor(full_dataset$class,       levels = class_levels)

# --- 2. IMPROVED METRICS FUNCTION ---
calc_metrics <- function(data, label) {
  # Handle cases where a strata might have 0 observations or missing classes
  if (nrow(data) == 0) return(NULL)
  
  results <- confusionMatrix(data$predictions, data$class, mode = "everything")
  
  cm  <- as.matrix(results$table)
  tp  <- diag(cm)
  fp  <- rowSums(cm) - tp
  fn  <- colSums(cm) - tp
  iou <- tp / (tp + fp + fn)
  
  # Map index for Waste (Class 1)
  waste_stats <- data.frame(
    Strata = label,
    N_Tiles = length(unique(data$filename)),
    Precision = results$byClass["Class: 1", "Precision"],
    Recall    = results$byClass["Class: 1", "Recall"],
    F1        = results$byClass["Class: 1", "F1"],
    IoU       = iou["1"]
  )
  return(waste_stats)
}

# --- 3. RUN ALL STRATAS AUTOMATICALLY ---

# A. Global Performance
global_perf <- calc_metrics(full_dataset, "Global (All)")

# B. Iterate through every "touches_X" column
stratified_results <- strata_cols %>%
  map_df(~ {
    # Dynamically filter where the current touches_X is TRUE
    # .x is the column name (e.g., "touches_2")
    subset_data <- full_dataset %>% filter(.data[[.x]] == TRUE)
    
    # Create a friendly label based on your class_labels mapping
    class_id <- gsub("touches_", "", .x)
    label_name <- paste("Waste touching", class_labels[class_id])
    
    calc_metrics(subset_data, label_name)
  })

# --- 4. FINAL TABLE ---
final_comparison <- bind_rows(global_perf, stratified_results)

print("--- Complete Stratified Accuracy Assessment ---")
print(final_comparison)

# Save the final results
#write.csv(full_dataset, "06_evaluation/05_accuracy_by_pile_size/waste_stratified_size.csv", row.names = FALSE)
