# ==============================================================================
# WASTE DETECTION PIPELINE
# Multiclass mask extracted from multiband (combined.tif)
# Binary = (waste_id == 1) only
# Pixel + Object feature extraction
# ==============================================================================

library(terra)
library(sf)
library(dplyr)
library(stringr)
library(tidyr)
library(caret)
library(rsample)
library(ggplot2)
library(flextable)
library(officer)
library(tictoc)

#===============================================================================
# 1. DATA LOADING
#===============================================================================

rgb_dir   <- "data/tiled_data/images/"
multi_dir <- "data/tiled_data/multiband/"     # ✅ contains textures + MULTICLASS MASK

get_tile_id <- function(path) {
  fname <- basename(path)
  str_replace_all(fname, "clipped_image_|_combined|\\.tif", "")
}

# RGB files (3-band optical)
rgb_df <- data.frame(
  rgb_path = list.files(rgb_dir, pattern="\\.tif$", full.names=TRUE)
) %>% mutate(id = get_tile_id(rgb_path))

# Multiband files (textures + multiclass mask)
multi_df <- data.frame(
  multi_path = list.files(multi_dir, pattern="combined\\.tif$", full.names=TRUE)
) %>% mutate(id = get_tile_id(multi_path))

# Merge by tile ID
files_df <- inner_join(rgb_df, multi_df, by="id")

#===============================================================================
# Texture names (matching your extraction order)
#===============================================================================
texture_names <- c(
  "Energy","Entropy","Cor","IDM","Inertia","CS","CP","HC",
  "Mean","Variance","Dissimilarity","SumAverage","SumVariance",
  "SumEntropy","Dif_of_Entropies","Dif_of_Variances","IC1","IC2",
  "SRE","LRE","GLN","RLN","RP","LGRE","HGLRE",
  "SRLGLE","SRHGLE","LRLGLE"
)

#===============================================================================
# Load all components into image_list
#===============================================================================

image_list <- lapply(1:nrow(files_df), function(i) {
  
  row <- files_df[i,]
  
  # --- RGB image ---
  rgb_r  <- rast(row$rgb_path)[[1:3]]
  names(rgb_r) <- c("Red","Green","Blue")
  
  # --- Multiband (textures + multiclass mask) ---
  multi_r <- rast(row$multi_path)
  n_bands <- nlyr(multi_r)
  
  # ✅ All texture layers
  tex_r <- multi_r[[1:(n_bands-1)]]
  names(tex_r) <- texture_names[1:(n_bands-1)]
  
  # ✅ LAST BAND = MULTICLASS MASK
  mask_r <- multi_r[[n_bands]]
  names(mask_r) <- "waste_id"
  
  list(mask = mask_r, optical = rgb_r, texture = tex_r, id = row$id)
})





#===============================================================================
# 2. CLASS MAPPING TABLE (MULTICLASS)
#===============================================================================

class_map <- data.frame(
  waste_id = c(1,2,3,4,5,6,7,8,9,99),
  label = c(
    "Waste",
    "Soil",
    "Vegetation",
    "Built-up surface",
    "Built-up surface (paved)",
    "Water Body",
    "Natural Surface - rocks",
    "Non-persistent object",
    "Ancillary class",
    "Ancillary class"
  )
)

#===============================================================================
# 3. PIXEL-LEVEL DATAFRAME
#===============================================================================

pixel_df <- lapply(image_list, function(img) {
  as.data.frame(c(img$optical, img$texture, img$mask), na.rm = TRUE)
}) %>%
  bind_rows() %>%
  left_join(class_map, by="waste_id") %>%   # ✅ add human labels
  mutate(
    class_multi = factor(waste_id),         # ✅ TRUE multiclass (10 classes)
    class_bin   = factor(                   # ✅ OPTION B: Waste = waste_id == 1
      ifelse(waste_id == 1, "Waste", "NonWaste"),
      levels=c("NonWaste","Waste")
    )
  )

#===============================================================================
# 4. OBJECT-LEVEL DATAFRAME
#===============================================================================

object_df <- lapply(image_list, function(img) {
  
  polys <- as.polygons(img$mask, dissolve=TRUE)
  
  stats <- terra::extract(
    c(img$optical, img$texture),
    polys,
    fun = mean,
    na.rm = TRUE
  )
  
  df_out <- data.frame(values(polys), stats) %>%
    select(-ID) %>%
    rename(waste_id = 1) %>%
    left_join(class_map, by="waste_id") %>%
    mutate(
      class_multi = factor(waste_id),       # ✅ TRUE multiclass
      class_bin   = factor(                 # ✅ OPTION B
        ifelse(waste_id == 1, "Waste", "NonWaste"),
        levels=c("NonWaste","Waste")
      )
    )
  
  df_out
  
}) %>%
  bind_rows()

#===============================================================================
# PIPELINE COMPLETE
# Pixel + Object datasets ready for modeling
#===============================================================================