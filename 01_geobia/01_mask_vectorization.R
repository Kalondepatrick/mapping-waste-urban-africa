library(terra)
library(sf)

mask_dir <- "data/tiled_data/masks/"
output_dir <- "data/tiled_data/vectors/" 

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

tif_files <- list.files(mask_dir, pattern = "\\.tif$", full.names = TRUE)

for (tif_path in tif_files) {
  
  r <- rast(tif_path)
  
  if (global(r, "max", na.rm = TRUE)[1,1] > 0) {
    
    v <- as.polygons(r, round = TRUE, aggregate = TRUE)
    v_sf <- st_as_sf(v)
    
    file_name <- sub("\\.tif$", ".shp", basename(tif_path))
    out_path <- file.path(output_dir, file_name)
    
    st_write(v_sf, out_path, delete_dsn = TRUE, quiet = TRUE)
    
    message(paste("Saved to:", out_path))
  } else {
    message(paste("Skipped empty mask:", basename(tif_path)))
  }
}