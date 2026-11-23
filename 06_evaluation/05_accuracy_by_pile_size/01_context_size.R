# --- 1. SETUP & LIBRARIES ---
library(terra)
library(dplyr)
library(ggplot2)
library(scales)

# Path setup
mask_folder <- "data/tiled_data/partitioned/Validation/masks/" 
files <- list.files(mask_folder, pattern = "\\.tif$", full.names = TRUE)
files <- files[!grepl("\\.aux\\.xml$", files)]

WASTE_VAL <- 1

# --- 2. CALCULATION LOOP ---
# We only care about the count of Class 1 (Waste) per tile
print("Calculating waste pile sizes per tile...")

waste_sizes <- sapply(files, function(f) {
  r <- rast(f)
  # Efficiently sum pixels where value == 1
  sum(values(r) == WASTE_VAL, na.rm = TRUE)
})

# Create dataframe and filter out tiles with zero waste
waste_df <- data.frame(
  filename = basename(files),
  pixel_count = waste_sizes
) %>% 
  filter(pixel_count > 0)

# --- 3. BINNING BY SIZE ---
# Customize these thresholds to match your specific environmental context
# --- 3. STATISTICALLY SOUND BINNING (QUANTILES) ---
# Calculate the exact breaks based on your 84 tiles
breaks <- quantile(waste_df$pixel_count, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE)

waste_df <- waste_df %>%
  mutate(size_category = cut(
    pixel_count, 
    breaks = breaks, 
    include.lowest = TRUE,
    labels = c("Smallest 25%", "Lower Mid", "Upper Mid", "Largest 25%")
  ))

# Check what those actual pixel values were
print("Quantile thresholds used:")
print(breaks)
# --- 4. SUMMARY STATISTICS ---
size_summary <- waste_df %>%
  group_by(size_category) %>%
  summarise(
    tile_count = n(),
    avg_pixels = mean(pixel_count),
    total_pixels = sum(pixel_count)
  ) %>%
  mutate(percentage = (tile_count / sum(tile_count)) * 100)

print("=== Waste Size Category Distribution ===")
print(size_summary)

# --- 5. VISUALIZATION ---
ggplot(size_summary, aes(x = size_category, y = tile_count, fill = size_category)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), vjust = -0.5) +
  scale_fill_viridis_d(option = "plasma", begin = 0.2, end = 0.8) +
  theme_minimal() +
  labs(
    title = "Distribution of Waste Pile Sizes",
    subtitle = paste("Analyzed across", nrow(waste_df), "tiles containing waste"),
    x = "Waste Size Category (Pixels)",
    y = "Number of Tiles"
  )

# Optional: Save results
write.csv(waste_df, "06_evaluation/05_accuracy_by_pile_size/waste_size_distribution.csv", row.names = FALSE)
