library(fs)
library(dplyr)
library(sf)
library(terra)

output_metadata <- "data/tiled_data/spatial_metadata"
copy_partitioned_files <- FALSE
reuse_existing_partitions <- TRUE
dir_create(output_metadata, recurse = TRUE)

aoi <- st_read("data/otherdata/17210-aoi.geojson")
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

plot(st_geometry(grid_clean), border = "grey")
plot(st_geometry(aoi), add = TRUE, border = "red", lwd = 2)

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


table(grid_clean$quadrant)

plot(
  st_geometry(grid_clean),
  col = c("lightblue","lightgreen","pink","orange")[as.factor(grid_clean$quadrant)],
  border = "grey"
)

plot(st_geometry(aoi), add = TRUE, border = "black", lwd = 2)

plot(aoi_center, add = TRUE, pch = 16, cex = 1.5)

#======================================================================================#
#   03. Reproduce the spatial split used in the study
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
#   04. Locate image tiles within the saved spatial split
#======================================================================================#

src_img <- "data/tiled_data/images"
src_msk <- "data/tiled_data/masks"
src_mb  <- "data/tiled_data/multiband"
base_target <- "data/tiled_data/partitioned"

development_image_paths <- list.files(
  path(base_target, "modelDevelopment", "images"),
  pattern = "\\.tif$",
  full.names = TRUE
)

holdout_image_paths <- list.files(
  path(base_target, "Validation", "images"),
  pattern = "\\.tif$",
  full.names = TRUE
)

if (reuse_existing_partitions &&
    length(development_image_paths) > 0 &&
    length(holdout_image_paths) > 0) {
  all_image_paths <- c(development_image_paths, holdout_image_paths)
  stored_set <- c(
    rep("development", length(development_image_paths)),
    rep("holdout", length(holdout_image_paths))
  )
} else {
  all_image_paths <- list.files(src_img, pattern = "\\.tif$", full.names = TRUE)
  stored_set <- rep(NA_character_, length(all_image_paths))
}

keep_tif <- !grepl("aux.xml", all_image_paths)
all_image_paths <- all_image_paths[keep_tif]
stored_set <- stored_set[keep_tif]

manifest_list <- list()

cat("Reading spatial extents for", length(all_image_paths), "files...\n")

for (i in seq_along(all_image_paths)) {
  path_val <- all_image_paths[i]
  
  result <- tryCatch({
    r <- rast(path_val)
    ext_val <- ext(r)
    
    data.frame(
      file_name = basename(path_val),
      stored_set = stored_set[i],
      xmin = as.numeric(ext_val$xmin),
      ymin = as.numeric(ext_val$ymin),
      xmax = as.numeric(ext_val$xmax),
      ymax = as.numeric(ext_val$ymax)
    )
  }, error = function(e) {
    message(paste("Skipping problematic file:", basename(path_val)))
    return(NULL)
  })
  
  manifest_list[[i]] <- result
}

tile_manifest <- bind_rows(manifest_list)

tile_manifest_sf <- tile_manifest %>%
  mutate(x_cent = (xmin + xmax) / 2, 
         y_cent = (ymin + ymax) / 2) %>%
  st_as_sf(coords = c("x_cent", "y_cent"), crs = st_crs(grid_clean))

tile_grid_lookup <- st_join(tile_manifest_sf, grid_clean)

if (anyDuplicated(tile_grid_lookup$file_name)) {
  stop("Some image tiles intersect more than one grid cell. Check tile centroids on grid boundaries.")
}

unassigned_count <- sum(is.na(tile_grid_lookup$id))

if (unassigned_count > 0) {
  warning(unassigned_count, " image tile(s) fall outside the partition grid and will be excluded.")
}

tiles_partitioned <- tile_grid_lookup %>%
  filter(!is.na(id))

known_stored_set <- !is.na(tiles_partitioned$stored_set)

if (any(known_stored_set & tiles_partitioned$stored_set != tiles_partitioned$set)) {
  stop("The existing development/validation folders do not match the reproducible spatial split.")
}

plot(
  st_geometry(grid_clean),
  col = ifelse(grid_clean$set == "holdout", "red", "lightgrey"),
  border = "black"
)

plot(st_geometry(aoi), add = TRUE, border = "blue", lwd = 2)

#======================================================================================#
#   05. Save the quadrant regions, partition regions, and tile assignments
#======================================================================================#

# Keep the sampled cells as a permanent record of the spatial split. The
# dissolved layer is convenient for display in QGIS, while grid_cells retains
# the cell IDs needed to reproduce and audit the assignment.
partition_regions <- grid_clean %>%
  group_by(set) %>%
  summarise(
    cell_count = n(),
    area_m2 = sum(as.numeric(area)),
    .groups = "drop"
  )

quadrant_regions <- grid_clean %>%
  group_by(quadrant) %>%
  summarise(
    cell_count = n(),
    area_m2 = sum(as.numeric(area)),
    .groups = "drop"
  )

partition_by_quadrant <- grid_clean %>%
  group_by(set, quadrant) %>%
  summarise(
    cell_count = n(),
    area_m2 = sum(as.numeric(area)),
    .groups = "drop"
  )

partition_gpkg <- path(output_metadata, "study_spatial_partitions.gpkg")

if (file_exists(partition_gpkg)) {
  file_delete(partition_gpkg)
}

st_write(
  grid_clean,
  partition_gpkg,
  layer = "grid_cells",
  quiet = TRUE
)

st_write(
  partition_regions,
  partition_gpkg,
  layer = "partition_regions",
  quiet = TRUE
)

st_write(
  quadrant_regions,
  partition_gpkg,
  layer = "quadrant_regions",
  quiet = TRUE
)

st_write(
  partition_by_quadrant,
  partition_gpkg,
  layer = "partition_by_quadrant",
  quiet = TRUE
)

write.csv(
  st_drop_geometry(grid_clean),
  path(output_metadata, "study_grid_partitioning.csv"),
  row.names = FALSE
)

st_write(
  tiles_partitioned,
  path(output_metadata, "study_tile_assignments.gpkg"),
  delete_dsn = TRUE,
  quiet = TRUE
)

write.csv(
  st_drop_geometry(tiles_partitioned),
  path(output_metadata, "study_tile_manifest.csv"),
  row.names = FALSE
)

partition_summary <- tiles_partitioned %>%
  st_drop_geometry() %>%
  count(set, quadrant, name = "tile_count")

write.csv(
  partition_summary,
  path(output_metadata, "study_partition_summary.csv"),
  row.names = FALSE
)

cat("Saved spatial partition metadata to:", output_metadata, "\n")

#======================================================================================#
#   06. Transfer (copy) of images into Development and Validation folders
#======================================================================================#

dirs <- c("modelDevelopment/images", "modelDevelopment/masks", "modelDevelopment/multiband",
          "Validation/images", "Validation/masks", "Validation/multiband")

if (copy_partitioned_files) {
  for (d in dirs) {
    dir_create(path(base_target, d), recurse = TRUE)
  }
}

copy_tiles <- function(df, target_set, target_folder) {
  
  to_copy <- df %>% filter(set == target_set)
  
  f_img <- 0; f_msk <- 0; f_mb <- 0
  
  for (fname in to_copy$file_name) {
    
    fname_img <- fname
    fname_msk <- fname
    fname_mb  <- gsub("\\.tif$", "_combined.tif", fname)
    
    s_img <- path(src_img, fname_img)
    s_msk <- path(src_msk, fname_msk)
    s_mb  <- path(src_mb, fname_mb)
    
    if (file_exists(s_img)) {
      file_copy(s_img, path(base_target, target_folder, "images", fname_img), overwrite = TRUE)
      f_img <- f_img + 1
    }
    
    if (file_exists(s_msk)) {
      file_copy(s_msk, path(base_target, target_folder, "masks", fname_msk), overwrite = TRUE)
      f_msk <- f_msk + 1
    }
    
    if (file_exists(s_mb)) {
      file_copy(s_mb, path(base_target, target_folder, "multiband", fname_mb), overwrite = TRUE)
      f_mb <- f_mb + 1
    }
  }
  
  cat("\n--- Summary for", target_folder, "---\n")
  cat("Images copied:     ", f_img, "\n")
  cat("Masks copied:      ", f_msk, "\n")
  cat("Multibands copied: ", f_mb, "\n")
  cat("------------------------------\n")
}

if (copy_partitioned_files) {
  copy_tiles(tiles_partitioned, "development", "modelDevelopment")
  copy_tiles(tiles_partitioned, "holdout", "Validation")

  cat("\nTransfer complete. Files partitioned into:", base_target, "\n")
} else {
  cat("File copying skipped. Set copy_partitioned_files <- TRUE to enable it.\n")
}
