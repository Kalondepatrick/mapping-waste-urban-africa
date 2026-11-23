library(fs)
library(dplyr)
library(sf)
library(terra)

# Set Paths
aoi_path <- "data/otherdata/17210-aoi.geojson"
src_img <- "data/tiled_data/images"
src_msk <- "data/tiled_data/masks"
src_mb  <- "data/tiled_data/multiband"
base_target <- "data/tiled_data/partitioned"
output_metadata <- "data/tiled_data/spatial_metadata"

# Create metadata directory if it doesn't exisQt
if (!dir_exists(output_metadata)) dir_create(output_metadata)

# Load AOI
aoi <- st_read(aoi_path)
aoi <- st_transform(aoi, 32736)

#======================================================================================#
#   01. Creation of grids
#======================================================================================#
grid <- st_make_grid(
  aoi,
  cellsize = c(500, 500),
  square = TRUE
)

grid_sf <- st_sf(id = 1:length(grid), geometry = grid)
grid_clip <- st_intersection(grid_sf, aoi)
grid_clip$area <- st_area(grid_clip)
grid_clean <- grid_clip

#======================================================================================#
#   02. Creation of quadrants
#======================================================================================#
aoi_union <- st_union(aoi)
aoi_center <- st_centroid(aoi_union)
center_xy <- st_coordinates(aoi_center)
x_mid <- center_xy[1]
y_mid <- center_xy[2]

grid_cent <- st_centroid(grid_clean)
grid_xy <- st_coordinates(grid_cent)
grid_clean$x <- grid_xy[,1]
grid_clean$y <- grid_xy[,2]

grid_clean$quadrant <- case_when(
  grid_clean$x <= x_mid & grid_clean$y >= y_mid ~ "NW",
  grid_clean$x >  x_mid & grid_clean$y >= y_mid ~ "NE",
  grid_clean$x <= x_mid & grid_clean$y <  y_mid ~ "SW",
  grid_clean$x >  x_mid & grid_clean$y <  y_mid ~ "SE"
)

#======================================================================================#
#   03. Partitioning (REPRODUCIBLE VIA SEED)
#======================================================================================#
set.seed(42)

holdout <- bind_rows(
  grid_clean %>% filter(quadrant == "NW") %>% slice_sample(n = 2),
  grid_clean %>% filter(quadrant == "NE") %>% slice_sample(n = 2),
  grid_clean %>% filter(quadrant == "SW") %>% slice_sample(n = 3),
  grid_clean %>% filter(quadrant == "SE") %>% slice_sample(n = 3)
)

holdout_ids <- holdout$id

grid_clean$set <- ifelse(
  grid_clean$id %in% holdout_ids,
  "holdout",
  "development"
)

#======================================================================================#
#   04. Manifest Creation (Mapping tiles to grids)
#======================================================================================#
all_image_paths <- list.files(src_img, pattern = "\\.tif$", full.names = TRUE)
all_image_paths <- all_image_paths[!grepl("aux.xml", all_image_paths)]

manifest_list <- list()

cat("Reading spatial extents for manifest...\n")
for (i in seq_along(all_image_paths)) {
  path_val <- all_image_paths[i]
  result <- tryCatch({
    r <- rast(path_val)
    ext_val <- ext(r)
    data.frame(
      file_name = basename(path_val),
      xmin = as.numeric(ext_val$xmin),
      ymin = as.numeric(ext_val$ymin),
      xmax = as.numeric(ext_val$xmax),
      ymax = as.numeric(ext_val$ymax)
    )
  }, error = function(e) return(NULL))
  manifest_list[[i]] <- result
}

tile_manifest <- bind_rows(manifest_list)

tile_manifest_sf <- tile_manifest %>%
  mutate(x_cent = (xmin + xmax) / 2, 
         y_cent = (ymin + ymax) / 2) %>%
  st_as_sf(coords = c("x_cent", "y_cent"), crs = st_crs(grid_clean))

tiles_partitioned <- st_join(tile_manifest_sf, grid_clean) %>%
  filter(!is.na(set))

#======================================================================================#
#   05. Save Outputs (The missing part!)
#======================================================================================#

# Save the grid with the attributes (NW/SE/etc and Development/Holdout)
st_write(grid_clean, path(output_metadata, "grid_partitioning.gpkg"), delete_dsn = TRUE)

# Save the joined tile manifest
st_write(tiles_partitioned, path(output_metadata, "tile_assignments.gpkg"), delete_dsn = TRUE)

# Save a CSV for easy non-spatial reference
write.csv(st_drop_geometry(tiles_partitioned), 
          path(output_metadata, "tile_manifest.csv"), row.names = FALSE)

cat("\nSuccess! Spatial data saved to:", output_metadata)

#======================================================================================#
#   06. Data Transfer (DISABLED)
#======================================================================================#
# To re-run copying, uncomment the lines below.

# copy_tiles <- function(df, target_set, target_folder) {
#   to_copy <- df %>% filter(set == target_set)
#   for (fname in to_copy$file_name) {
#     fname_mb <- gsub("\\.tif$", "_combined.tif", fname)
#     if (file_exists(path(src_img, fname))) file_copy(path(src_img, fname), path(base_target, target_folder, "images", fname), overwrite = TRUE)
#     if (file_exists(path(src_msk, fname))) file_copy(path(src_msk, fname), path(base_target, target_folder, "masks", fname), overwrite = TRUE)
#     if (file_exists(path(src_mb, fname_mb))) file_copy(path(src_mb, fname_mb), path(base_target, target_folder, "multiband", fname_mb), overwrite = TRUE)
#   }
# }

# copy_tiles(tiles_partitioned, "development", "modelDevelopment")
# copy_tiles(tiles_partitioned, "holdout", "Validation")