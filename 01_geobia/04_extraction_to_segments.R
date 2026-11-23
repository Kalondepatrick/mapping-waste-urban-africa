library(terra)
library(sf)
library(dplyr)
library(stringr)
library(future.apply)

# --- 1. SETUP PATHS ---
segment_dir <- "data/tiled_data/partitioned/modelDevelopment/segments/" 
mask_dir    <- "data/tiled_data/partitioned/modelDevelopment/masks/"
rgb_dir     <- "data/tiled_data/partitioned/modelDevelopment/images/"
multi_dir   <- "data/tiled_data/partitioned/modelDevelopment/multiband/"
out_dir     <- "data/tiled_data/partitioned/modelDevelopment/segments_with_var" 

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# --- 2. CONFIGURE PARALLEL BACKEND ---
# This detects available cores (e.g., on an HPC node)
plan(multisession, workers = parallelly::availableCores() - 1)
message(paste("Running in parallel with", parallelly::availableCores() - 1, "cores."))

# --- 3. PROCESSING FUNCTION ---
process_tile <- function(i, files_df, out_dir, texture_names) {
  row <- files_df[i,]
  
  # Check for NULL or NA paths before calling terra functions
  if (is.null(row$seg_path) || is.na(row$seg_path) || !file.exists(row$seg_path)) {
    return(paste("Skipped: Segment path missing for row", i))
  }
  
  tryCatch({
    # 1. Load Master Reference (RGB)
    r_rgb <- rast(row$rgb_path)[[1:3]]
    
    # 2. Load Vector (vect)
    v <- vect(row$seg_path)
    if (nrow(v) == 0) return(NULL)
    
    # 3. Project Vector
    if(!same.crs(v, r_rgb)) v <- project(v, crs(r_rgb))
    
    # 4. Load others
    r_mask <- rast(row$mask_path)
    r_tex  <- rast(row$multi_path)
    
    # 5. Overlap Gate
    if (!any(is.related(v, ext(r_rgb), "intersects"))) return(NULL)
    
    # 6. Extraction
    names(r_rgb) <- c("Red", "Green", "Blue")
    names(r_tex) <- texture_names[1:nlyr(r_tex)]
    
    class_stats <- terra::extract(r_mask, v, fun = "modal", na.rm = TRUE)
    rgb_stats   <- terra::extract(r_rgb, v, fun = "mean", na.rm = TRUE)
    tex_stats   <- terra::extract(r_tex, v, fun = "mean", na.rm = TRUE)
    
    # 7. Combine
    v_final <- cbind(v, 
                     class = class_stats[, 2], 
                     rgb_stats[, -1, drop=FALSE], 
                     tex_stats[, -1, drop=FALSE])
    
    # 8. Save
    sf_final <- st_as_sf(v_final)
    out_path <- file.path(out_dir, paste0(row$id, ".shp"))
    st_write(sf_final, out_path, delete_dsn = TRUE, quiet = TRUE)
    
    return(paste("Success:", row$id))
    
  }, error = function(e) {
    return(paste("Error in tile", row$id, ":", e$message))
  })
}

# --- 4. EXECUTE IN PARALLEL ---
# Using future_lapply to distribute the 804 tiles
results <- future_lapply(1:nrow(files_df), function(i) {
  process_tile(i, files_df, out_dir, texture_names)
}, future.seed = TRUE)

# Print a summary of the results
message("Processing complete.")