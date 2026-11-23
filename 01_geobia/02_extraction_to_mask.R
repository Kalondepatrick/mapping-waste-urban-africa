library(terra)
library(sf)
library(dplyr)
library(stringr)

# --- 1. SETUP PATHS ---
vector_dir <- "data/tiled_data/vectors/" 
rgb_dir    <- "data/tiled_data/partitioned/modelDevelopment/images/"
multi_dir  <- "data/tiled_data/partitioned/modelDevelopment/multiband/"
out_dir    <- "data/tiled_data/vectors_with_var" # Removed trailing slash

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# --- 3. EXTRACTION AND SAVING ---
for (i in 1:nrow(files_df)) {
  row <- files_df[i,]
  
  # Load Vector
  poly_sf <- st_read(row$vec_path, quiet = TRUE)
  
  # FIX 1: Flexible Renaming
  # Check if the problematic column exists before renaming
  if ("c__1__0" %in% colnames(poly_sf)) {
    poly_sf <- poly_sf %>% rename(class = c__1__0)
  } else if (!"class" %in% colnames(poly_sf)) {
    # If it's not c__1__0 and not already called class, 
    # assume the first attribute column is the class
    colnames(poly_sf)[1] <- "class"
  }
  
  v <- vect(poly_sf)
  
  # Load Rasters
  rgb_r  <- rast(row$rgb_path)[[1:3]]
  tex_r  <- rast(row$multi_path)
  
  # FIX 2: CRS Alignment
  # terra::extract requires matching CRS to be accurate
  if(!same.crs(v, rgb_r)) {
    v <- project(v, crs(rgb_r))
  }
  
  names(rgb_r) <- c("Red", "Green", "Blue")
  names(tex_r) <- texture_names[1:nlyr(tex_r)]
  
  combined_r <- c(rgb_r, tex_r)
  
  # Zonal extraction
  stats <- terra::extract(combined_r, v, fun = "mean", na.rm = TRUE)
  
  # Combine
  v_final <- cbind(v, stats[, -1])
  sf_final <- st_as_sf(v_final)
  
  # FIX 3: Clean Output Path
  # Using file.path without trailing slashes in out_dir prevents // errors
  out_path <- file.path(out_dir, paste0(row$id, ".shp"))
  
  st_write(sf_final, out_path, delete_dsn = TRUE, quiet = TRUE)
  
  if(i %% 10 == 0) message(paste("Processed", i, "of", nrow(files_df)))
}