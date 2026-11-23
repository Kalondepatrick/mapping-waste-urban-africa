library(terra)
library(sf)
library(dplyr)
library(stringr)

# --- 1. SETUP PATHS ---
# trimws() removes any accidental leading/trailing spaces
vector_dir <- trimws("data/tiled_data/partitioned/Validation/vectors/") 
rgb_dir    <- trimws("data/tiled_data/partitioned/Validation/images/")
multi_dir  <- trimws("data/tiled_data/partitioned/Validation/multiband/")
out_dir    <- trimws("data/tiled_data/partitioned/Validation/vectors_with_var")

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# --- 2. FILE MATCHING ---
vec_files   <- list.files(vector_dir, pattern = "\\.shp$", full.names = TRUE)
rgb_files   <- list.files(rgb_dir,    pattern = "\\.tif$", full.names = TRUE)
multi_files <- list.files(multi_dir,  pattern = "\\.tif$", full.names = TRUE)

# Build the matching dataframe
files_df <- data.frame(
  vec_path = vec_files,
  id = tools::file_path_sans_ext(basename(vec_files)),
  stringsAsFactors = FALSE
)

files_df <- files_df %>%
  mutate(
    # RGB matches the ID exactly
    rgb_path   = file.path(rgb_dir,   paste0(id, ".tif")),
    # Multiband has the "_combined" suffix
    multi_path = file.path(multi_dir, paste0(id, "_combined.tif"))
  ) %>%
  # Only keep rows where both raster files actually exist on disk
  filter(file.exists(rgb_path), file.exists(multi_path))

# --- VERIFICATION ---
n_matches <- nrow(files_df)
message(paste("Found", n_matches, "matching sets of files."))

if (n_matches == 0) {
  stop("No matching files found. Double-check your folder paths and suffixes!")
}

# Define texture names (Adjust based on your specific multiband layers)

texture_names <- c("Energy","Entropy","Cor","IDM","Inertia","CS","CP","HC",
                   "Mean","Variance","Dissimilarity","SumAverage","SumVariance",
                   "SumEntropy","Dif_of_Entropies","Dif_of_Variances","IC1","IC2",
                   "SRE","LRE","GLN","RLN","RP","LGRE","HGLRE",
                   "SRLGLE","SRHGLE","LRLGLE")

# --- 3. EXTRACTION AND SAVING ---
for (i in 1:nrow(files_df)) {
  row <- files_df[i, ]
  
  # Load Vector
  poly_sf <- st_read(row$vec_path, quiet = TRUE)
  
  # Skip if the shapefile is empty
  if (nrow(poly_sf) == 0) {
    message(paste("Skipping empty vector:", row$id))
    next
  }
  
  # FIX: Flexible Renaming for the 'class' column
  if ("c__1__0" %in% colnames(poly_sf)) {
    poly_sf <- poly_sf %>% rename(class = c__1__0)
  } else if (!"class" %in% colnames(poly_sf)) {
    # If standard names aren't found, rename the first attribute column to 'class'
    geom_col <- attr(poly_sf, "sf_column")
    attr_cols <- setdiff(colnames(poly_sf), geom_col)
    if(length(attr_cols) > 0) colnames(poly_sf)[which(colnames(poly_sf) == attr_cols[1])] <- "class"
  }
  
  # Convert to terra SpatVector for speed
  v <- vect(poly_sf)
  
  # Load Rasters
  rgb_r  <- rast(row$rgb_path)[[1:3]] # Take only first 3 bands (R, G, B)
  tex_r  <- rast(row$multi_path)
  
  # Ensure CRS alignment
  if(!same.crs(v, rgb_r)) {
    v <- project(v, crs(rgb_r))
  }
  
  # Assign layer names
  names(rgb_r) <- c("Red", "Green", "Blue")
  n_tex <- nlyr(tex_r)
  names(tex_r) <- texture_names[1:min(n_tex, length(texture_names))]
  
  # Combine into a single stack
  combined_r <- c(rgb_r, tex_r)
  
  # Zonal extraction: Mean of every band for every polygon
  stats <- terra::extract(combined_r, v, fun = "mean", na.rm = TRUE)
  
  # Combine stats with vector (dropping the ID column from extract)
  v_final <- cbind(v, stats[, -1])
  sf_final <- st_as_sf(v_final)
  
  # Save Output
  out_path <- file.path(out_dir, paste0(row$id, ".shp"))
  st_write(sf_final, out_path, delete_dsn = TRUE, quiet = TRUE)
  
  if(i %% 10 == 0) message(paste("Processed", i, "of", n_matches))
}

message("Processing complete! Files saved to: ", out_dir)