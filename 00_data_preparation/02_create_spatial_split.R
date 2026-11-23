suppressPackageStartupMessages({
  library(dplyr)
  library(sf)
  library(terra)
})

# This script reconstructs and verifies the spatial split used in the study.
# Machine-specific locations are supplied through environment variables.
project_root <- Sys.getenv("MWUA_PROJECT_ROOT", unset = getwd())
project_root <- normalizePath(project_root, winslash = "/", mustWork = TRUE)

data_root <- Sys.getenv(
  "MWUA_DATA_ROOT",
  unset = file.path(project_root, "data")
)

aoi_path <- Sys.getenv(
  "MWUA_AOI_PATH",
  unset = file.path(data_root, "otherdata", "17210-aoi.geojson")
)

partition_root <- Sys.getenv(
  "MWUA_PARTITION_ROOT",
  unset = file.path(data_root, "tiled_data", "partitioned")
)

output_dir <- file.path(data_root, "splits")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

expected_development <- 804L
expected_holdout <- 222L
expected_total <- expected_development + expected_holdout

development_dir <- file.path(partition_root, "modelDevelopment", "images")
holdout_dir <- file.path(partition_root, "Validation", "images")

required_inputs <- c(aoi_path, development_dir, holdout_dir)
missing_inputs <- required_inputs[!file.exists(required_inputs)]

if (length(missing_inputs) > 0) {
  stop(
    "Missing required Stage 00 input(s):\n",
    paste(missing_inputs, collapse = "\n")
  )
}

list_tifs <- function(directory) {
  paths <- list.files(
    directory,
    pattern = "\\.[tT][iI][fF]$",
    full.names = TRUE
  )
  paths[!grepl("\\.aux\\.xml$", paths, ignore.case = TRUE)]
}

development_paths <- list_tifs(development_dir)
holdout_paths <- list_tifs(holdout_dir)
all_image_paths <- c(development_paths, holdout_paths)
stored_set <- c(
  rep("development", length(development_paths)),
  rep("holdout", length(holdout_paths))
)

observed_counts <- c(
  development = length(development_paths),
  holdout = length(holdout_paths)
)

if (observed_counts[["development"]] != expected_development ||
    observed_counts[["holdout"]] != expected_holdout) {
  stop(
    "Existing folders do not contain the published split. Observed: ",
    observed_counts[["development"]], " development and ",
    observed_counts[["holdout"]], " holdout; expected: ",
    expected_development, " and ", expected_holdout, "."
  )
}

if (anyDuplicated(basename(all_image_paths))) {
  stop("At least one image filename occurs in both partition folders.")
}

aoi <- st_read(aoi_path, quiet = TRUE)
aoi <- st_transform(aoi, 32736)

grid <- st_make_grid(
  aoi,
  cellsize = c(500, 500),
  square = TRUE
)

grid_sf <- st_sf(id = seq_along(grid), geometry = grid)
grid_clean <- st_intersection(grid_sf, aoi)
grid_clean$area_m2 <- as.numeric(st_area(grid_clean))

aoi_center <- st_centroid(st_union(aoi))
center_xy <- st_coordinates(aoi_center)
x_mid <- center_xy[1, 1]
y_mid <- center_xy[1, 2]

grid_centers <- st_coordinates(st_centroid(grid_clean))
grid_clean$x <- grid_centers[, 1]
grid_clean$y <- grid_centers[, 2]
grid_clean$quadrant <- case_when(
  grid_clean$x <= x_mid & grid_clean$y >= y_mid ~ "NW",
  grid_clean$x >  x_mid & grid_clean$y >= y_mid ~ "NE",
  grid_clean$x <= x_mid & grid_clean$y <  y_mid ~ "SW",
  grid_clean$x >  x_mid & grid_clean$y <  y_mid ~ "SE"
)

set.seed(42)

holdout_cells <- bind_rows(
  grid_clean %>% filter(quadrant == "NW") %>% slice_sample(n = 2),
  grid_clean %>% filter(quadrant == "NE") %>% slice_sample(n = 2),
  grid_clean %>% filter(quadrant == "SW") %>% slice_sample(n = 3),
  grid_clean %>% filter(quadrant == "SE") %>% slice_sample(n = 3)
)

holdout_ids <- holdout_cells$id
grid_clean$set <- ifelse(
  grid_clean$id %in% holdout_ids,
  "holdout",
  "development"
)

message("Reading spatial extents for ", length(all_image_paths), " tiles...")

manifest_rows <- lapply(seq_along(all_image_paths), function(i) {
  image_path <- all_image_paths[i]

  tryCatch({
    raster_extent <- ext(rast(image_path))
    data.frame(
      file_name = basename(image_path),
      stored_set = stored_set[i],
      xmin = xmin(raster_extent),
      ymin = ymin(raster_extent),
      xmax = xmax(raster_extent),
      ymax = ymax(raster_extent)
    )
  }, error = function(error) {
    stop("Could not read raster extent for ", image_path, ": ", error$message)
  })
})

tile_manifest <- bind_rows(manifest_rows)
tile_points <- tile_manifest %>%
  mutate(
    x_center = (xmin + xmax) / 2,
    y_center = (ymin + ymax) / 2
  ) %>%
  st_as_sf(
    coords = c("x_center", "y_center"),
    crs = st_crs(grid_clean)
  )

tile_assignments <- st_join(tile_points, grid_clean)

if (anyDuplicated(tile_assignments$file_name)) {
  stop("At least one tile centroid intersects more than one grid cell.")
}

if (any(is.na(tile_assignments$id))) {
  stop(sum(is.na(tile_assignments$id)), " tile centroid(s) fall outside the AOI grid.")
}

set_mismatches <- tile_assignments %>%
  filter(stored_set != set)

if (nrow(set_mismatches) > 0) {
  stop(
    nrow(set_mismatches),
    " stored tile assignment(s) disagree with the seed-42 spatial split."
  )
}

if (nrow(tile_assignments) != expected_total) {
  stop("Expected ", expected_total, " assigned tiles; found ", nrow(tile_assignments), ".")
}

partition_regions <- grid_clean %>%
  group_by(set) %>%
  summarise(
    cell_count = n(),
    area_m2 = sum(area_m2),
    .groups = "drop"
  )

quadrant_regions <- grid_clean %>%
  group_by(quadrant) %>%
  summarise(
    cell_count = n(),
    area_m2 = sum(area_m2),
    .groups = "drop"
  )

partition_by_quadrant <- grid_clean %>%
  group_by(set, quadrant) %>%
  summarise(
    cell_count = n(),
    area_m2 = sum(area_m2),
    .groups = "drop"
  )

partition_path <- file.path(output_dir, "study_spatial_partitions.gpkg")
assignment_path <- file.path(output_dir, "study_tile_assignments.gpkg")

if (file.exists(partition_path)) file.remove(partition_path)
if (file.exists(assignment_path)) file.remove(assignment_path)

st_write(grid_clean, partition_path, layer = "grid_cells", quiet = TRUE)
st_write(partition_regions, partition_path, layer = "partition_regions", quiet = TRUE)
st_write(quadrant_regions, partition_path, layer = "quadrant_regions", quiet = TRUE)
st_write(
  partition_by_quadrant,
  partition_path,
  layer = "partition_by_quadrant",
  quiet = TRUE
)

st_write(tile_assignments, assignment_path, quiet = TRUE)

write.csv(
  st_drop_geometry(grid_clean),
  file.path(output_dir, "study_grid_partitioning.csv"),
  row.names = FALSE
)

write.csv(
  st_drop_geometry(tile_assignments),
  file.path(output_dir, "study_tile_manifest.csv"),
  row.names = FALSE
)

partition_summary <- tile_assignments %>%
  st_drop_geometry() %>%
  count(set, quadrant, name = "tile_count")

write.csv(
  partition_summary,
  file.path(output_dir, "study_partition_summary.csv"),
  row.names = FALSE
)

message(
  "Stage 00 spatial partition verified and saved: ",
  expected_development, " development + ",
  expected_holdout, " holdout = ", expected_total, " tiles."
)

