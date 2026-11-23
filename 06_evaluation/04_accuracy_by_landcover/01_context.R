# --- 1. SETUP & LIBRARIES ---
library(terra)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(scales)

# Path to your mask files
mask_folder <- "data/tiled_data/partitioned/Validation/masks/" 
files <- list.files(mask_folder, pattern = "\\.tif$", full.names = TRUE)
files <- files[!grepl("\\.aux\\.xml$", files)]

# Define Class Mappings
class_labels <- c(
  "1" = "Waste",
  "2" = "Soil",
  "3" = "Vegetation",
  "4" = "Built-up surface",
  "5" = "Built-up surface (paved area)",
  "6" = "Water Body",
  "7" = "Natural Surface - rocks",
  "8" = "Non-persistent object",
  "9" = "Ancillary class",
  "99" = "unidentified"
)

WASTE_VAL <- 1
all_classes <- c(1, 2, 3, 4, 5, 6, 7, 8, 9)

# --- 2. GLOBAL PIXEL FREQUENCIES ---
print(paste("Calculating global frequencies for", length(files), "tiles..."))

frequency_data <- files %>%
  map_df(~ {
    r <- rast(.x)
    freq(r)
  }) %>%
  group_by(value) %>%
  summarise(count = sum(count, na.rm = TRUE)) %>%
  mutate(
    Label = class_labels[as.character(value)],
    Percentage = (count / sum(count)) * 100
  ) %>%
  arrange(desc(Percentage))

print("=== Global Landcover Distribution ===")
print(frequency_data)

# --- 3. WASTE-CENTRIC ADJACENCY ANALYSIS ---
# Pre-allocating for efficiency
results_list <- vector("list", length(files))

print("Analyzing spatial context: What does Waste touch?")

for (i in seq_along(files)) {
  if (i %% 100 == 0) cat("Processing Tile:", i, "/", length(files), "\n")
  
  file_path <- files[i]
  r <- rast(file_path)
  
  # Check if Waste (Class 1) exists in this specific tile
  has_w <- any(values(r == WASTE_VAL), na.rm = TRUE)
  
  if (has_w) {
    waste_mask <- r == WASTE_VAL
    # 8-direction buffer (1 pixel) to find immediate neighbors
    waste_edge <- buffer(waste_mask, width = 1)
    
    # Check which other classes are within that 1-pixel border
    neighbors <- sapply(all_classes[-1], function(nbr) {
      any(values(waste_edge & (r == nbr)), na.rm = TRUE)
    })
    
    names(neighbors) <- paste0("touches_", all_classes[-1])
    results_list[[i]] <- as.data.frame(t(neighbors)) %>%
      mutate(filename = basename(file_path), has_waste = TRUE)
  }
}

# Combine only the tiles that actually contained waste
waste_context_df <- bind_rows(results_list)

# --- 4. FINAL WASTE INTERACTION SUMMARY ---
total_waste_tiles <- nrow(waste_context_df)

waste_summary <- waste_context_df %>%
  summarise(across(starts_with("touches_"), sum)) %>%
  pivot_longer(cols = everything(), names_to = "Neighbor_Class", values_to = "Tile_Count") %>%
  mutate(
    Neighbor_Value = gsub("touches_", "", Neighbor_Class),
    Neighbor_Label = class_labels[Neighbor_Value],
    Percentage_of_Waste_Tiles = (Tile_Count / total_waste_tiles) * 100
  ) %>%
  select(Neighbor_Label, Tile_Count, Percentage_of_Waste_Tiles) %>%
  arrange(desc(Tile_Count))

print(paste("=== Waste Interaction Summary (Total Waste Tiles:", total_waste_tiles, ") ==="))
print(waste_summary)

# --- 5. VISUALIZATION ---
ggplot(waste_summary, aes(x = reorder(Neighbor_Label, Percentage_of_Waste_Tiles), y = Percentage_of_Waste_Tiles)) +
  geom_bar(stat = "identity", fill = "firebrick", width = 0.7) +
  coord_flip() +
  scale_y_continuous(labels = unit_format(unit = "%", sep = "")) +
  theme_minimal() +
  labs(
    title = "What does Waste touch most often?",
    subtitle = paste("Based on", total_waste_tiles, "tiles containing Waste"),
    x = "Neighboring Landcover",
    y = "Percentage of Waste Tiles"
  )
write.csv(waste_context_df, "06_evaluation/04_accuracy_by_landcover/waste_context_summary.csv", row.names = FALSE)
