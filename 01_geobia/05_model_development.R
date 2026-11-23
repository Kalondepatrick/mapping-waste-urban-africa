################################################################################
# MAPPING WASTE PILES: RFE, MULTI-SCENARIO TRAINING, AND IOU METRICS
################################################################################

# --- 1. LIBRARIES & SETUP ---
required_packages <- c("sf", "dplyr", "caret", "tictoc", "rsample", "stringr", "randomForest")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg, dependencies = TRUE)
  library(pkg, character.only = TRUE)
}

vector_dir <- "data/tiled_data/partitioned/modelDevelopment/vectors_with_var"
set.seed(42)

# --- 2. DATA AGGREGATION & CLEANING ---
all_shps <- list.files(vector_dir, pattern = "\\.shp$", full.names = TRUE)
message(paste("Merging", length(all_shps), "tiles..."))
reflectance_sf <- do.call(rbind, lapply(all_shps, st_read, quiet = TRUE))
df_all <- st_drop_geometry(reflectance_sf)

# Clean labels and remove metadata
df_all <- df_all %>%
  mutate(class = as.factor(class)) %>%
  dplyr::select(-any_of(c("ID", "tile_id", "bin", "area_m2")))

# Define Feature Sets
optical_features <- c("Red", "Green", "Blue")
all_features     <- setdiff(names(df_all), "class")

# --- 3. RECURSIVE FEATURE ELIMINATION (RFE) ---
message("\n--- Running Recursive Feature Elimination ---")
# Using a subset for RFE to save time if your dataset is massive
rfe_sample <- df_all %>% group_by(class) %>% sample_n(min(n(), 500)) %>% ungroup()
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)

tic()
rfe_results <- rfe(rfe_sample[, all_features], rfe_sample$class, 
                   sizes = c(1:length(all_features)), 
                   rfeControl = control)
toc()

optimal_features <- predictors(rfe_results)
message(paste("Optimal features selected:", paste(optimal_features, collapse=", ")))

# --- 4. DATA PARTITIONING ---
data_split <- initial_split(df_all, prop = 0.8, strata = class)
train_data <- training(data_split)
test_data  <- testing(data_split)

################################################################################
# 5. TRAINING & EVALUATION FUNCTION (WITH IOU)
################################################################################

train_and_evaluate <- function(train_df, test_df, model_type, feature_set, scenario_name) {
  message(paste("\nScenario:", scenario_name, "| Model:", model_type))
  
  form <- as.formula(paste("class ~", paste(feature_set, collapse = "+")))
  ctrl <- trainControl(method = "cv", number = 5)
  
  tic()
  model <- train(form, data = train_df, method = model_type, 
                 metric = "Accuracy", trControl = ctrl)
  toc()
  
  preds <- predict(model, test_df)
  conf  <- confusionMatrix(preds, test_df$class, mode = "everything")
  
  # --- CALCULATE IOU FOR WASTE (CLASS 1) ---
  # IoU = TP / (TP + FP + FN)
  cm <- conf$table
  waste_id <- "1"
  tp <- cm[waste_id, waste_id]
  fp <- sum(cm[waste_id, ]) - tp
  fn <- sum(cm[, waste_id]) - tp
  iou_waste <- tp / (tp + fp + fn)
  
  # Extract Class 1 specific metrics
  m <- conf$byClass[paste0("Class: ", waste_id), ]
  
  return(data.frame(
    Scenario    = scenario_name,
    Model       = model_type,
    Accuracy    = round(conf$overall['Accuracy'], 4),
    Precision   = round(m['Precision'], 4),
    Sensitivity = round(m['Sensitivity'], 4),
    F1_Score    = round(m['F1'], 4),
    IoU_Waste   = round(iou_waste, 4)
  ))
}

################################################################################
# 6. EXECUTION: OPTICAL VS OPTIMAL
################################################################################

models <- c("rf", "nnet", "nb", "svmLinear")
results_list <- list()

# Run Optical Scenario
for(m in models) {
  results_list[[paste0("Optic_", m)]] <- train_and_evaluate(train_data, test_data, m, optical_features, "Optical_Only")
}

# Run Optimal Scenario (RFE Selected)
for(m in models) {
  results_list[[paste0("Optimal_", m)]] <- train_and_evaluate(train_data, test_data, m, optimal_features, "Optimal_RFE")
}

# --- 7. FINAL COMPARISON ---
comparison_df <- do.call(rbind, results_list)
cat("\n======================================================\n")
cat("FINAL COMPARATIVE ANALYSIS (WASTE CLASS 1)\n")
cat("======================================================\n")
print(comparison_df)

# Export results
#write.csv(comparison_df, "results/01_geobia/model_performance_comparison.csv", row.names = FALSE)
#saveRDS(rfe_results, "data/trained_models/geobia_r/rfe_feature_selection.rds")

message("\n--- Analysis complete. Metrics and IoU for Waste are exported. ---")


library(caret)
library(tictoc)   # Fixes the "tic" error
library(caret)    # Fixes the "trainControl" error
library(randomForest) # Required for the "rf" method

# 1. Create the directory if it doesn't exist
if(!dir.exists("01_geobia/models")) dir.create("01_geobia/models", recursive = TRUE)

# 2. Loop through the list and save each model using its list name
for (model_name in names(model_list)) {
  
  # Define the file path
  file_path <- paste0("01_geobia/models", model_name, ".rds")
  
  # Save the specific model object
  saveRDS(model_list[[model_name]], file = file_path)
  
  message(paste("Saved:", model_name, "to", file_path))
}
