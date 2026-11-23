# Build dissolved outline polygons of the image tiles.
# Output feeds the study-area figures; see 07_figures/.
# Run from the repository root.

library(terra)
library(sf)

# --- CONFIGURATION ---
input_folder <- "data/tiled_data/images"
output_shapefile <- "data/vectors/merged_outlines/merged_outlines.shp"

# 1. List all GeoTIFF files in the folder (Fixed argument to full.names)
tiff_files <- list.files(input_folder, pattern = "\\.tiff?$", full.names = TRUE, ignore.case = TRUE)

if (length(tiff_files) == 0) {
  stop("No GeoTIFF files found in the specified folder. Check your folder path!")
}

message(paste("Processing", length(tiff_files), "GeoTIFF chips..."))

# 2. Extract bounding boxes as polygons
poly_list <- lapply(tiff_files, function(file) {
  tryCatch({
    r <- rast(file)
    b_box <- as.polygons(ext(r), crs = crs(r))
    st_as_sf(b_box)
  }, error = function(e) {
    warning(paste("Skipping corrupt file:", file))
    return(NULL)
  })
})

# Base R way to drop any NULL entries if a file failed to load
poly_list <- poly_list[!sapply(poly_list, is.null)]

# Combine all individual chip boundaries into one sf dataframe
all_chips <- do.call(rbind, poly_list)

# 3. Dissolve overlapping/neighboring polygons
dissolved_union <- st_union(all_chips)
dissolved_polygons <- st_cast(st_as_sf(dissolved_union), "POLYGON")

# 4. Clean up the attribute table (add an explicit ID column)
dissolved_polygons$id <- 1:nrow(dissolved_polygons)

# Make sure the output directory exists before saving
if(!dir.exists(dirname(output_shapefile))) {
  dir.create(dirname(output_shapefile), recursive = TRUE)
}

# 5. Save to Shapefile
st_write(dissolved_polygons, output_shapefile, delete_dsn = TRUE)

message(paste("Success! Saved", nrow(dissolved_polygons), "distinct polygon outlines to:", output_shapefile))