# --- 1. SETUP & DATA LOADING ---
library(sf)
library(caret)
library(dplyr)
library(purrr)
library(terra)

# Load model and define file paths
model_obj   <- readRDS("data/trained_models/geobia_r/RF.rds")
rf_model    <- model_obj$model  # Extract the actual train object
folder_path <- "data/tiled_data/partitioned/Validation/vectors_with_var/"
files       <- list.files(folder_path, pattern = "\\.shp$", full.names = TRUE)

# Get global class levels from the model to keep factor indexing consistent
class_levels <- rf_model$levels

# --- 2. BATCH PROCESSING & GLOBAL METRICS ---

# Read all shapes, drop geometry, and combine for a global validation
full_dataset <- files %>%
  map_df(~ st_read(.x, quiet = TRUE) %>% st_drop_geometry())

# Run Predictions
full_dataset$predictions <- predict(rf_model, newdata = full_dataset)

# Ensure factors match the model levels exactly
full_dataset$predictions <- factor(full_dataset$predictions, levels = class_levels)
full_dataset$class       <- factor(full_dataset$class,       levels = class_levels)

# Calculate Confusion Matrix and Metrics
results <- confusionMatrix(full_dataset$predictions, full_dataset$class, mode = "everything")

# Extract table and calculate IoU (Intersection over Union)
cm  <- as.matrix(results$table)
tp  <- diag(cm)
fp  <- rowSums(cm) - tp
fn  <- colSums(cm) - tp
iou <- tp / (tp + fp + fn)

# Create clean summary table
summary_stats <- data.frame(
  Precision = results$byClass[, "Precision"],
  Recall    = results$byClass[, "Recall"],
  F1        = results$byClass[, "F1"],
  IoU       = iou
)

print("--- Validation Metrics ---")
print(summary_stats)

# Persist the raw confusion matrix. Without the off-diagonal counts the false
# positive rate cannot be recovered, which blocks the prevalence analysis in
# 06_evaluation/. Saving it here keeps GeoBIA comparable with the U-Net results.
dir.create("results/01_geobia", recursive = TRUE, showWarnings = FALSE)
utils::write.csv(as.data.frame.matrix(cm),
                 "results/01_geobia/geobia_rf_confusion_matrix.csv")

waste_level <- if ("waste" %in% class_levels) "waste" else class_levels[1]
tp_w <- cm[waste_level, waste_level]
fp_w <- sum(cm[waste_level, ]) - tp_w
fn_w <- sum(cm[, waste_level]) - tp_w
tn_w <- sum(cm) - tp_w - fp_w - fn_w
counts <- data.frame(
  model = "GeoBIA RF (RFE features)", class = waste_level,
  tp = tp_w, fp = fp_w, fn = fn_w, tn = tn_w,
  precision = tp_w / (tp_w + fp_w),
  recall    = tp_w / (tp_w + fn_w),
  fpr       = fp_w / (fp_w + tn_w),
  prevalence = (tp_w + fn_w) / sum(cm)
)
utils::write.csv(counts, "results/01_geobia/geobia_waste_counts.csv",
                 row.names = FALSE)
print(counts)

# --- 3. SPATIAL EXPORT (SINGLE TILE EXAMPLE) ---

# Load specific tile for rasterization
tile_path <- "data/tiled_data/partitioned/Validation/vectors_with_var/clipped_image_110_tile_512_512.shp"
sample_sf <- st_read(tile_path)

# Predict and convert to fixed numeric indices based on class_levels
sample_sf$predictions <- predict(rf_model, newdata = sample_sf)
sample_sf$pred_num    <- as.numeric(factor(sample_sf$predictions, levels = class_levels))

# Create template and rasterize
# Note: touches = TRUE ensures small polygons aren't lost at high resolutions
target_res <- 0.03
template   <- rast(sample_sf, res = target_res)
pred_rast  <- rasterize(sample_sf, template, field = "pred_num", touches = TRUE)

# Export as GeoTIFF
output_file <- "data/predictions/geobia_map_tiles/predicted_map_tile_110.tif"
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
writeRaster(pred_rast, 
            filename = output_file, 
            overwrite = TRUE, 
            datatype  = "INT1U",
            gdal      = c("COMPRESS=DEFLATE", "TFW=YES"))

message("Success: Raster saved to ", output_file)