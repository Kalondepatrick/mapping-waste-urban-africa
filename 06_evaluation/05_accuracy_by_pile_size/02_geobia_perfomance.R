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
# Ensure this path points to your new CSV with the 'size_category' column
strata_csv   <- "06_evaluation/05_accuracy_by_pile_size/waste_size_distribution.csv" 

# Load Strata Data
strata_data <- read.csv(strata_csv)

# Load and Combine all shapefiles
files <- list.files(folder_path, pattern = "\\.shp$", full.names = TRUE)

full_dataset <- files %>%
  map_df(~ {
    ds <- st_read(.x, quiet = TRUE) %>% st_drop_geometry()
    # Ensure filename matching logic is consistent
    ds$filename <- gsub("\\.shp$", ".tif", basename(.x))
    return(ds)
  })

# Join Size Strata to Dataset
# This links every predicted object to the size category of its parent tile
full_dataset <- full_dataset %>%
  left_join(strata_data %>% select(filename, size_category), by = "filename") %>%
  filter(!is.na(size_category)) # Only keep tiles present in your size analysis

# Run Predictions
class_levels <- rf_model$levels
full_dataset$predictions <- predict(rf_model, newdata = full_dataset)
full_dataset$predictions <- factor(full_dataset$predictions, levels = class_levels)
full_dataset$class       <- factor(full_dataset$class,       levels = class_levels)

# --- 2. METRICS FUNCTION ---
calc_metrics <- function(data, label) {
  if (nrow(data) == 0) return(NULL)
  
  # Generate Confusion Matrix
  results <- confusionMatrix(data$predictions, data$class, mode = "everything")
  
  # Manual IoU Calculation for Class 1 (Waste)
  cm  <- as.matrix(results$table)
  # Ensure "1" exists in the matrix to avoid index errors
  if (!"1" %in% rownames(cm)) return(NULL)
  
  tp  <- cm["1", "1"]
  fp  <- sum(cm["1", ]) - tp
  fn  <- sum(cm[, "1"]) - tp
  iou <- tp / (tp + fp + fn)
  
  waste_stats <- data.frame(
    Size_Strata = label,
    N_Objects = nrow(data),
    N_Tiles = length(unique(data$filename)),
    Precision = results$byClass["Class: 1", "Precision"],
    Recall    = results$byClass["Class: 1", "Recall"],
    F1        = results$byClass["Class: 1", "F1"],
    IoU       = iou
  )
  return(waste_stats)
}

# --- 3. RUN STRATIFIED ASSESSMENT ---

# A. Global Performance (Baseline)
global_perf <- calc_metrics(full_dataset, "Global (All Sizes)")

# B. Stratified by Size Category
# We iterate through the unique categories found in your CSV
size_groups <- unique(full_dataset$size_category)

stratified_results <- size_groups %>%
  map_df(~ {
    subset_data <- full_dataset %>% filter(size_category == .x)
    calc_metrics(subset_data, .x)
  })

# --- 4. FINAL TABLE & EXPORT ---
final_comparison <- bind_rows(global_perf, stratified_results) %>%
  arrange(desc(F1))

print("--- Accuracy Assessment Stratified by Pile Size ---")
print(final_comparison)

write.csv(final_comparison, "06_evaluation/05_accuracy_by_pile_size/waste_stratified_size_results.csv", row.names = FALSE)