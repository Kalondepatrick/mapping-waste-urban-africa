# Waste probability and contiguous-object threshold analysis
#
# Reference object: one manually digitised waste-pile polygon.
# Predicted object: one 8-connected cluster of pixels at or above a threshold.
#
# Object-level definitions:
#   TP = a predicted object matched one-to-one to a reference pile
#   FP = a predicted object not matched to a reference pile
#   FN = a reference pile not matched to a predicted object
#
# IMPORTANT: Object precision is valid only when waste.shp exhaustively maps
# every waste pile within the evaluated probability raster. If the shapefile
# contains positive examples only, object recall remains useful, but unmatched
# predicted objects cannot automatically be interpreted as false positives.

suppressPackageStartupMessages({
  library(terra)
  library(sf)
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(tidyr)
  library(scales)
})

# =============================================================================
# 1. PATHS AND ANALYSIS SETTINGS
# =============================================================================

base_path <- file.path(
  path.expand("~"),
  "GitHub",
  "mapping-waste-urban-africa"
)

probability_path <- file.path(
  base_path,
  "results",
  "whole_area_prediction",
  "waste_probability.tif"
)

waste_path <- file.path(
  base_path,
  "data",
  "vectors",
  "waste_probability_step0",
  "waste.shp"
)

output_dir <- file.path(
  base_path,
  "results",
  "whole_area_prediction",
  "object_threshold_analysis"
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

output_csv <- file.path(
  output_dir,
  "waste_probability_and_detection_statistics.csv"
)

output_gpkg <- file.path(
  output_dir,
  "waste_probability_and_detection_statistics.gpkg"
)

threshold_metrics_csv <- file.path(
  output_dir,
  "object_metrics_by_threshold.csv"
)

threshold_choices_csv <- file.path(
  output_dir,
  "candidate_thresholds.csv"
)

per_pile_threshold_csv <- file.path(
  output_dir,
  "per_pile_detection_limits.csv"
)

# Connectivity used to define a predicted object: 4 or 8.
connectivity <- 8L

# A cluster smaller than this is excluded as an isolated prediction.
# This should be chosen a priori based on the smallest useful mapped object.
minimum_predicted_object_area_m2 <- 0.25

# Minimum fraction of a reference pile that must overlap a predicted object.
# Use 0 for any overlap, 0.01 for 1%, 0.05 for 5%, and so forth.
minimum_reference_coverage <- 0.01

# Recall constraint for the recall-protected threshold.
target_object_recall <- 0.90

# Start with a coarse search because connected-component labelling can be slow.
coarse_thresholds <- seq(0.05, 0.95, by = 0.05)

# Refine around the most relevant coarse thresholds.
run_refinement <- TRUE
refinement_half_width <- 0.05
refinement_step <- 0.01

# Allow terra to use disk-backed processing for a large raster.
terraOptions(memfrac = 0.70, progress = 1)

# =============================================================================
# 2. LOAD AND VALIDATE INPUT DATA
# =============================================================================

if (!file.exists(probability_path)) {
  stop("Probability raster not found: ", probability_path)
}

if (!file.exists(waste_path)) {
  stop("Waste shapefile not found: ", waste_path)
}

probability <- rast(probability_path)

if (nlyr(probability) != 1L) {
  stop("The probability raster must contain exactly one band.")
}

names(probability) <- "probability"

if (is.na(crs(probability)) || crs(probability) == "") {
  stop("The probability raster has no coordinate reference system.")
}

if (terra::is.lonlat(probability)) {
  stop(
    "The raster must use a projected CRS with metre-based units so that ",
    "object areas can be calculated in square metres."
  )
}

waste <- st_read(waste_path, quiet = TRUE)

if (nrow(waste) == 0L) {
  stop("The waste shapefile contains no features.")
}

waste <- st_make_valid(waste)
waste <- st_transform(waste, crs(probability))
waste <- waste[!st_is_empty(waste), , drop = FALSE]
waste$pile_id <- seq_len(nrow(waste))

waste_vect <- vect(waste)

# The full raster is used as the evaluation area. If the reference polygons
# are exhaustive only within a smaller validation AOI, crop and mask the raster
# to that AOI before running this analysis.
evaluation_probability <- probability

cell_area_m2 <- prod(res(evaluation_probability))

minimum_predicted_cells <- max(
  1L,
  ceiling(minimum_predicted_object_area_m2 / cell_area_m2)
)

reference_objects <- tibble(
  pile_id = waste$pile_id,
  reference_area_m2 = as.numeric(st_area(waste))
)

number_reference_objects <- nrow(reference_objects)

message("Reference piles: ", number_reference_objects)
message("Raster cell area: ", signif(cell_area_m2, 6), " m2")
message(
  "Minimum predicted-object area: ",
  minimum_predicted_object_area_m2,
  " m2 (", minimum_predicted_cells, " cells)"
)
message(
  "Minimum reference coverage for a match: ",
  percent(minimum_reference_coverage, accuracy = 0.1)
)

# =============================================================================
# 3. EXTRACT PROBABILITIES WITHIN EACH REFERENCE PILE
# =============================================================================

extracted <- terra::extract(
  probability,
  waste_vect,
  ID = TRUE,
  touches = FALSE
)

names(extracted)[names(extracted) == "ID"] <- "pile_id"

extracted <- extracted %>%
  filter(
    !is.na(probability),
    is.finite(probability)
  )

if (nrow(extracted) == 0L) {
  stop("No valid probability pixels were found inside the waste polygons.")
}

pile_statistics <- extracted %>%
  group_by(pile_id) %>%
  summarise(
    pixel_count = n(),
    mean_probability = mean(probability),
    variance_probability = var(probability),
    sd_probability = sd(probability),
    minimum_probability = min(probability),
    q25_probability = as.numeric(quantile(probability, 0.25)),
    median_probability = median(probability),
    q75_probability = as.numeric(quantile(probability, 0.75)),
    maximum_probability = max(probability),
    .groups = "drop"
  )

overall_statistics <- extracted %>%
  summarise(
    pixel_count = n(),
    mean_probability = mean(probability),
    variance_probability = var(probability),
    sd_probability = sd(probability),
    minimum_probability = min(probability),
    q25_probability = as.numeric(quantile(probability, 0.25)),
    median_probability = median(probability),
    q75_probability = as.numeric(quantile(probability, 0.75)),
    maximum_probability = max(probability)
  )

# Weakest possible object rule: at least one above-threshold pixel inside every
# pile. This is reported for context but should not be used as the final rule,
# because a single high-probability pixel may be noise.
single_pixel_all_piles_threshold <- min(
  pile_statistics$maximum_probability,
  na.rm = TRUE
)

# =============================================================================
# 4. DESCRIPTIVE PROBABILITY PLOTS
# =============================================================================

p_probability_distribution <- ggplot(
  extracted,
  aes(x = probability)
) +
  geom_histogram(
    bins = 50,
    fill = "#D95F02",
    colour = "white"
  ) +
  geom_density(
    aes(y = after_stat(count)),
    linewidth = 0.9,
    colour = "black"
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.1)
  ) +
  labs(
    x = "Predicted waste probability",
    y = "Number of pixels",
    title = "Predicted probabilities within digitised waste piles"
  ) +
  theme_bw(base_size = 12)

ggsave(
  file.path(output_dir, "01_probability_distribution.png"),
  p_probability_distribution,
  width = 8,
  height = 5,
  dpi = 300
)

pile_order <- pile_statistics %>%
  arrange(mean_probability) %>%
  pull(pile_id)

extracted_for_plot <- extracted %>%
  mutate(
    pile_id_factor = factor(pile_id, levels = pile_order)
  )

p_probability_by_pile <- ggplot(
  extracted_for_plot,
  aes(x = pile_id_factor, y = probability)
) +
  geom_boxplot(
    outlier.shape = NA,
    fill = "#56B4E9",
    width = 0.75
  ) +
  geom_point(
    data = pile_statistics,
    aes(
      x = factor(pile_id, levels = pile_order),
      y = mean_probability
    ),
    colour = "#D55E00",
    size = 1.5,
    inherit.aes = FALSE
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2)
  ) +
  labs(
    x = "Waste pile, ordered by mean probability",
    y = "Predicted waste probability",
    title = "Variation in predicted probability among waste piles",
    subtitle = "Orange points show mean probability"
  ) +
  theme_bw(base_size = 12) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

ggsave(
  file.path(output_dir, "02_probability_by_waste_pile.png"),
  p_probability_by_pile,
  width = 10,
  height = 6,
  dpi = 300
)

p_mean_probability <- pile_statistics %>%
  arrange(mean_probability) %>%
  mutate(pile_rank = row_number()) %>%
  ggplot(aes(x = pile_rank, y = mean_probability)) +
  geom_line(colour = "#0072B2") +
  geom_point(size = 2, colour = "#0072B2") +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.1)
  ) +
  labs(
    x = "Waste pile, ordered by mean probability",
    y = "Mean predicted probability",
    title = "Mean predicted probability of each digitised waste pile"
  ) +
  theme_bw(base_size = 12)

ggsave(
  file.path(output_dir, "03_mean_probability_by_waste_pile.png"),
  p_mean_probability,
  width = 8,
  height = 5,
  dpi = 300
)

# =============================================================================
# 5. MATCHING AND METRIC FUNCTIONS
# =============================================================================

# Match reference and predicted objects one-to-one. Candidate pairs with the
# highest intersection-over-union are assigned first. This prevents one large
# predicted component from receiving credit for several separate reference
# piles.
greedy_object_match <- function(candidate_pairs) {
  if (nrow(candidate_pairs) == 0L) {
    return(candidate_pairs)
  }
  
  candidate_pairs <- candidate_pairs %>%
    arrange(
      desc(iou),
      desc(reference_coverage),
      desc(overlap_area_m2)
    )
  
  used_reference_ids <- integer(0)
  used_prediction_ids <- numeric(0)
  retain <- logical(nrow(candidate_pairs))
  
  for (i in seq_len(nrow(candidate_pairs))) {
    reference_id <- candidate_pairs$pile_id[i]
    prediction_id <- candidate_pairs$patch_id[i]
    
    if (
      !(reference_id %in% used_reference_ids) &&
      !(prediction_id %in% used_prediction_ids)
    ) {
      retain[i] <- TRUE
      used_reference_ids <- c(used_reference_ids, reference_id)
      used_prediction_ids <- c(used_prediction_ids, prediction_id)
    }
  }
  
  candidate_pairs[retain, , drop = FALSE]
}

empty_threshold_result <- function(threshold, predicted_objects = 0L) {
  precision_value <- if (predicted_objects == 0L) NA_real_ else 0
  
  list(
    summary = tibble(
      threshold = threshold,
      reference_objects = number_reference_objects,
      predicted_objects = as.integer(predicted_objects),
      true_positives = 0L,
      false_positives = as.integer(predicted_objects),
      false_negatives = number_reference_objects,
      object_precision = precision_value,
      object_recall = 0,
      object_f1 = 0
    ),
    matches = tibble(
      threshold = numeric(0),
      pile_id = integer(0),
      patch_id = numeric(0),
      overlap_cells = integer(0),
      overlap_area_m2 = numeric(0),
      intersection_area_m2 = numeric(0),
      reference_area_m2 = numeric(0),
      predicted_cells = numeric(0),
      predicted_area_m2 = numeric(0),
      reference_coverage = numeric(0),
      predicted_coverage = numeric(0),
      iou = numeric(0)
    )
  )
}

evaluate_object_threshold <- function(threshold) {
  message("Evaluating object threshold: ", sprintf("%.2f", threshold))
  
  # Cells below the threshold are set to NA so that patches() labels only
  # above-threshold regions.
  binary_prediction <- terra::ifel(
    evaluation_probability >= threshold,
    1,
    NA
  )
  
  predicted_objects_raster <- terra::patches(
    binary_prediction,
    directions = connectivity
  )
  
  names(predicted_objects_raster) <- "patch_id"
  
  patch_frequency <- terra::freq(predicted_objects_raster)
  
  if (is.null(patch_frequency) || nrow(patch_frequency) == 0L) {
    return(empty_threshold_result(threshold))
  }
  
  patch_frequency <- as.data.frame(patch_frequency)
  
  if (!all(c("value", "count") %in% names(patch_frequency))) {
    stop("Unexpected column names returned by terra::freq().")
  }
  
  patch_information <- patch_frequency %>%
    transmute(
      patch_id = as.numeric(value),
      predicted_cells = as.numeric(count),
      predicted_area_m2 = predicted_cells * cell_area_m2
    ) %>%
    filter(predicted_cells >= minimum_predicted_cells)
  
  number_predicted_objects <- nrow(patch_information)
  
  if (number_predicted_objects == 0L) {
    return(empty_threshold_result(threshold))
  }
  
  # Extract component IDs underneath each reference polygon. Repeated patch IDs
  # are counted to estimate the intersection area of each candidate pair.
  overlap_pixels <- terra::extract(
    predicted_objects_raster,
    waste_vect,
    ID = TRUE,
    touches = FALSE
  )
  
  names(overlap_pixels)[1:2] <- c("pile_id", "patch_id")
  
  overlap_pixels <- overlap_pixels %>%
    filter(
      !is.na(patch_id),
      patch_id %in% patch_information$patch_id
    )
  
  if (nrow(overlap_pixels) == 0L) {
    return(empty_threshold_result(threshold, number_predicted_objects))
  }
  
  candidate_pairs <- overlap_pixels %>%
    count(pile_id, patch_id, name = "overlap_cells") %>%
    mutate(overlap_area_m2 = overlap_cells * cell_area_m2) %>%
    left_join(reference_objects, by = "pile_id") %>%
    left_join(patch_information, by = "patch_id") %>%
    mutate(
      # Raster-cell overlap can be slightly larger than a small vector polygon
      # at its boundary. Cap the estimated intersection by both object areas.
      intersection_area_m2 = pmin(
        overlap_area_m2,
        reference_area_m2,
        predicted_area_m2
      ),
      reference_coverage = intersection_area_m2 / reference_area_m2,
      predicted_coverage = intersection_area_m2 / predicted_area_m2,
      iou = intersection_area_m2 /
        (
          reference_area_m2 +
            predicted_area_m2 -
            intersection_area_m2
        )
    ) %>%
    filter(reference_coverage >= minimum_reference_coverage)
  
  matched_objects <- greedy_object_match(candidate_pairs) %>%
    mutate(threshold = threshold, .before = 1)
  
  true_positives <- nrow(matched_objects)
  false_positives <- number_predicted_objects - true_positives
  false_negatives <- number_reference_objects - true_positives
  
  object_precision <- true_positives /
    (true_positives + false_positives)
  
  object_recall <- true_positives /
    (true_positives + false_negatives)
  
  object_f1 <- if (
    object_precision + object_recall > 0
  ) {
    2 * object_precision * object_recall /
      (object_precision + object_recall)
  } else {
    0
  }
  
  list(
    summary = tibble(
      threshold = threshold,
      reference_objects = number_reference_objects,
      predicted_objects = number_predicted_objects,
      true_positives = true_positives,
      false_positives = false_positives,
      false_negatives = false_negatives,
      object_precision = object_precision,
      object_recall = object_recall,
      object_f1 = object_f1
    ),
    matches = matched_objects
  )
}

run_thresholds <- function(threshold_values) {
  threshold_values <- sort(unique(round(threshold_values, 6)))
  lapply(threshold_values, evaluate_object_threshold)
}

summaries_from_results <- function(results) {
  bind_rows(lapply(results, function(x) x$summary))
}

matches_from_results <- function(results) {
  bind_rows(lapply(results, function(x) x$matches))
}

# =============================================================================
# 6. COARSE THRESHOLD SEARCH
# =============================================================================

coarse_results <- run_thresholds(coarse_thresholds)
coarse_metrics <- summaries_from_results(coarse_results)

coarse_balanced <- coarse_metrics %>%
  filter(!is.na(object_f1)) %>%
  slice_max(object_f1, n = 1, with_ties = FALSE)

coarse_recall_protected <- coarse_metrics %>%
  filter(
    object_recall >= target_object_recall,
    !is.na(object_precision)
  ) %>%
  arrange(
    desc(object_precision),
    desc(object_f1),
    desc(threshold)
  ) %>%
  slice(1)

coarse_all_piles <- coarse_metrics %>%
  filter(object_recall >= 1 - 1e-12) %>%
  slice_max(threshold, n = 1, with_ties = FALSE)

# =============================================================================
# 7. REFINED THRESHOLD SEARCH
# =============================================================================

all_results <- coarse_results

if (run_refinement) {
  refinement_centres <- unique(c(
    coarse_balanced$threshold,
    coarse_recall_protected$threshold,
    coarse_all_piles$threshold
  ))
  
  refinement_centres <- refinement_centres[
    is.finite(refinement_centres)
  ]
  
  if (length(refinement_centres) > 0L) {
    refined_thresholds <- sort(unique(unlist(lapply(
      refinement_centres,
      function(centre) {
        seq(
          max(0.01, centre - refinement_half_width),
          min(0.99, centre + refinement_half_width),
          by = refinement_step
        )
      }
    ))))
    
    refined_thresholds <- round(refined_thresholds, 6)
    already_evaluated <- round(coarse_metrics$threshold, 6)
    refined_thresholds <- refined_thresholds[
      !(refined_thresholds %in% already_evaluated)
    ]
    
    if (length(refined_thresholds) > 0L) {
      refined_results <- run_thresholds(refined_thresholds)
      all_results <- c(all_results, refined_results)
    }
  }
}

object_metrics <- summaries_from_results(all_results) %>%
  arrange(threshold) %>%
  distinct(threshold, .keep_all = TRUE)

object_matches <- matches_from_results(all_results) %>%
  arrange(threshold, pile_id)

# =============================================================================
# 8. SELECT CANDIDATE THRESHOLDS
# =============================================================================

# Balanced choice: maximum object-level F1.
balanced_threshold <- object_metrics %>%
  filter(!is.na(object_f1)) %>%
  arrange(
    desc(object_f1),
    desc(object_recall),
    desc(threshold)
  ) %>%
  slice(1)

# Recall-protected choice: maximise precision while meeting the prespecified
# object-recall target.
recall_protected_threshold <- object_metrics %>%
  filter(
    object_recall >= target_object_recall,
    !is.na(object_precision)
  ) %>%
  arrange(
    desc(object_precision),
    desc(object_f1),
    desc(threshold)
  ) %>%
  slice(1)

# Highest evaluated threshold at which every pile remains individually matched.
all_piles_threshold <- object_metrics %>%
  filter(object_recall >= 1 - 1e-12) %>%
  slice_max(threshold, n = 1, with_ties = FALSE)

format_threshold_choice <- function(result, rule) {
  if (nrow(result) == 0L) {
    return(tibble(
      selection_rule = rule,
      threshold = NA_real_,
      predicted_objects = NA_integer_,
      true_positives = NA_integer_,
      false_positives = NA_integer_,
      false_negatives = NA_integer_,
      object_precision = NA_real_,
      object_recall = NA_real_,
      object_f1 = NA_real_
    ))
  }
  
  result %>%
    transmute(
      selection_rule = rule,
      threshold,
      predicted_objects,
      true_positives,
      false_positives,
      false_negatives,
      object_precision,
      object_recall,
      object_f1
    )
}

threshold_choices <- bind_rows(
  format_threshold_choice(
    balanced_threshold,
    "Maximum object-level F1"
  ),
  format_threshold_choice(
    recall_protected_threshold,
    paste0(
      "Maximum precision with object recall >= ",
      percent(target_object_recall, accuracy = 1)
    )
  ),
  format_threshold_choice(
    all_piles_threshold,
    "Highest threshold detecting every reference pile"
  )
)

selected_threshold <- recall_protected_threshold$threshold

if (length(selected_threshold) == 0L || !is.finite(selected_threshold)) {
  selected_threshold <- balanced_threshold$threshold
  selected_rule <- "Maximum object-level F1 because the recall target was not met"
} else {
  selected_rule <- paste0(
    "Maximum precision with object recall >= ",
    percent(target_object_recall, accuracy = 1)
  )
}

# =============================================================================
# 9. PER-PILE DETECTION LIMITS
# =============================================================================

if (nrow(object_matches) > 0L) {
  per_pile_detection_limit <- object_matches %>%
    group_by(pile_id) %>%
    summarise(
      highest_object_threshold = max(threshold),
      .groups = "drop"
    )
} else {
  per_pile_detection_limit <- tibble(
    pile_id = integer(0),
    highest_object_threshold = numeric(0)
  )
}

per_pile_detection_limit <- reference_objects %>%
  left_join(per_pile_detection_limit, by = "pile_id") %>%
  left_join(
    pile_statistics %>%
      select(
        pile_id,
        mean_probability,
        maximum_probability
      ),
    by = "pile_id"
  ) %>%
  mutate(
    detected_at_selected_threshold =
      !is.na(highest_object_threshold) &
      highest_object_threshold >= selected_threshold
  ) %>%
  arrange(highest_object_threshold)

# Add all summary results to the spatial reference layer.
waste_with_statistics <- waste %>%
  left_join(pile_statistics, by = "pile_id") %>%
  left_join(
    per_pile_detection_limit %>%
      select(
        pile_id,
        highest_object_threshold,
        detected_at_selected_threshold
      ),
    by = "pile_id"
  ) %>%
  mutate(pile_area_m2 = as.numeric(st_area(geometry)))

# =============================================================================
# 10. OBJECT-LEVEL PERFORMANCE PLOTS
# =============================================================================

object_metrics_long <- object_metrics %>%
  select(
    threshold,
    object_precision,
    object_recall,
    object_f1
  ) %>%
  pivot_longer(
    cols = -threshold,
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = recode(
      metric,
      object_precision = "Object precision",
      object_recall = "Object recall",
      object_f1 = "Object F1"
    )
  )

p_object_metrics <- ggplot(
  object_metrics_long,
  aes(x = threshold, y = value, colour = metric)
) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_vline(
    xintercept = selected_threshold,
    linetype = "dashed",
    colour = "black"
  ) +
  annotate(
    "text",
    x = selected_threshold,
    y = 0.05,
    label = paste0("Selected = ", round(selected_threshold, 2)),
    angle = 90,
    hjust = 0,
    vjust = -0.5,
    size = 3.5
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.1)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    labels = percent_format()
  ) +
  labs(
    x = "Probability threshold",
    y = "Object-level performance",
    colour = NULL,
    title = "Contiguous waste-object performance by probability threshold",
    subtitle = paste0(
      "Minimum predicted area = ",
      minimum_predicted_object_area_m2,
      " m2; minimum reference coverage = ",
      percent(minimum_reference_coverage, accuracy = 0.1)
    )
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(
  file.path(output_dir, "04_object_precision_recall_f1.png"),
  p_object_metrics,
  width = 9,
  height = 6,
  dpi = 300
)

# Precision-recall curve. The threshold is represented by colour.
p_precision_recall <- object_metrics %>%
  filter(
    !is.na(object_precision),
    !is.na(object_recall)
  ) %>%
  ggplot(
    aes(
      x = object_recall,
      y = object_precision,
      colour = threshold
    )
  ) +
  geom_path(linewidth = 0.8, colour = "grey50") +
  geom_point(size = 3) +
  geom_point(
    data = object_metrics %>%
      filter(abs(threshold - selected_threshold) < 1e-10),
    shape = 21,
    size = 5,
    stroke = 1.2,
    fill = "white"
  ) +
  scale_colour_viridis_c(
    option = "C",
    limits = c(0, 1)
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    labels = percent_format()
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    labels = percent_format()
  ) +
  coord_equal() +
  labs(
    x = "Object recall",
    y = "Object precision",
    colour = "Threshold",
    title = "Object-level precision-recall curve",
    subtitle = "The outlined point is the selected recall-protected threshold"
  ) +
  theme_bw(base_size = 12)

ggsave(
  file.path(output_dir, "05_object_precision_recall_curve.png"),
  p_precision_recall,
  width = 7,
  height = 6,
  dpi = 300
)

p_detection_limit <- ggplot(
  per_pile_detection_limit,
  aes(
    x = reference_area_m2,
    y = highest_object_threshold,
    colour = detected_at_selected_threshold
  )
) +
  geom_hline(
    yintercept = selected_threshold,
    linetype = "dashed",
    colour = "black"
  ) +
  geom_point(size = 2.5, na.rm = TRUE) +
  scale_x_log10() +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.1)
  ) +
  scale_colour_manual(
    values = c(`TRUE` = "#0072B2", `FALSE` = "#D55E00"),
    labels = c(`TRUE` = "Detected", `FALSE` = "Missed"),
    na.value = "grey50"
  ) +
  labs(
    x = "Reference waste-pile area (m2; log scale)",
    y = "Highest threshold at which the pile is detected",
    colour = paste0("At threshold ", round(selected_threshold, 2)),
    title = "Object-detection limit of each waste pile"
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(
  file.path(output_dir, "06_per_pile_object_detection_limits.png"),
  p_detection_limit,
  width = 8,
  height = 6,
  dpi = 300
)

# =============================================================================
# 11. SAVE TABLES AND SPATIAL OUTPUT
# =============================================================================

write_csv(object_metrics, threshold_metrics_csv)
write_csv(threshold_choices, threshold_choices_csv)
write_csv(per_pile_detection_limit, per_pile_threshold_csv)
write_csv(st_drop_geometry(waste_with_statistics), output_csv)

st_write(
  waste_with_statistics,
  output_gpkg,
  layer = "waste_object_statistics",
  delete_dsn = TRUE,
  quiet = TRUE
)

write_csv(
  overall_statistics,
  file.path(output_dir, "overall_probability_statistics.csv")
)

if (nrow(object_matches) > 0L) {
  write_csv(
    object_matches,
    file.path(output_dir, "object_matches_by_threshold.csv")
  )
}

# =============================================================================
# 12. REPORT MAIN RESULTS
# =============================================================================

message("\nAnalysis completed.")
message("Outputs saved in: ", output_dir)
message(
  "Single-pixel threshold that retains every pile: ",
  round(single_pixel_all_piles_threshold, 4),
  " (descriptive only; not recommended as the final threshold)"
)
message(
  "Selected contiguous-object threshold: ",
  round(selected_threshold, 4)
)
message("Selection rule: ", selected_rule)

cat("\nOverall probability statistics:\n")
print(overall_statistics)

cat("\nCandidate thresholds:\n")
print(threshold_choices)

cat("\nPerformance at the selected threshold:\n")
print(
  object_metrics %>%
    filter(abs(threshold - selected_threshold) < 1e-10)
)
