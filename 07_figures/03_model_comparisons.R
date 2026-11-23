# =============================================================================
# WASTE MAPPING: AREA THRESHOLD AND SINGLE-OBSERVER VERIFICATION AUDIT
# =============================================================================
# Run from the top in RStudio (R >= 4.4). All findings are recalculated.
# Outputs: IEEE-style Word tables/report, high-resolution figures, CSV, GPKG.
# Polygon counts are NOT necessarily counts of distinct physical waste piles.
# =============================================================================

# 1. Inputs ------------------------------------------------------------------
required <- c("sf", "ggplot2", "officer", "flextable")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Install these R packages: ", paste(missing, collapse = ", "))

library(sf)
library(ggplot2)

# Paths resolve from the repository root, so this runs on any machine.
root <- local({
  p <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  while (!file.exists(file.path(p, "CITATION.cff")) && dirname(p) != p) p <- dirname(p)
  if (!file.exists(file.path(p, "CITATION.cff")))
    stop("Run this from inside the repository.")
  p
})

reference_file <- file.path(root, "data/waste/waste_outlines.geojson")
raw_file       <- file.path(root, "data/predictions/waste_piles_vector/waste_piles.shp")
# River network supplied for full-study-area river-distance analyses (RQ7 and RQ8).
river_file     <- file.path(root, "data/river/river.shp")

# The visually verified inventory lives in the companion repository. Point
# MWUA_VERIFIED_PILES at it, or drop the shapefile in data/vectors/verified/.
verified_file <- Sys.getenv(
  "MWUA_VERIFIED_PILES",
  unset = file.path(root, "data/vectors/verified/verefied_waste_piles.shp"))

out_dir <- file.path(root, "results", "figures", "manuscript")

crs_m <- 32736
selected_area <- 0.25
thresholds <- c(0, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1, 2, 5, 10, 25)
near_m <- 10                              # Diagnostic only; not a confirmed match
near_river_m <- 20                       # Within 20 m of nearest mapped river
stopifnot(selected_area %in% thresholds)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

load_polygons <- function(path) {
  if (!file.exists(path)) stop("Missing file: ", path)
  x <- st_read(path, quiet = TRUE)
  if (is.na(st_crs(x))) stop("Missing coordinate reference system: ", path)
  x <- st_make_valid(st_transform(x, crs_m))
  x <- suppressWarnings(st_collection_extract(x, "POLYGON"))
  x <- x[!st_is_empty(x), ]
  if (!nrow(x)) stop("No polygons: ", path)
  x
}

reference <- load_polygons(reference_file)
raw <- load_polygons(raw_file)
verified <- load_polygons(verified_file)
reference$outline_no <- seq_len(nrow(reference))
reference_area <- as.numeric(st_area(reference))
raw_area <- as.numeric(st_area(raw))
verified_area <- as.numeric(st_area(verified))
if (any(!is.finite(reference_area) | reference_area <= 0)) stop("Invalid reference areas")

# 2. Polygon-to-reference correspondence ------------------------------------
# A pile counts as overlapping when intersection has positive area. Multiple
# predictions associated with one reference are unioned, to allow splitting.
ref_geom <- st_geometry(reference)
raw_geom <- st_geometry(raw)
verified_geom <- st_geometry(verified)
raw_hits <- st_relate(reference, raw, pattern = "2********", sparse = TRUE)
verified_hits <- st_relate(reference, verified, pattern = "2********", sparse = TRUE)

score_references <- function(pred_geom, hits, keep) {
  out <- matrix(0, nrow(reference), 4,
                dimnames = list(NULL, c("parts", "overlap", "coverage", "iou")))
  for (i in seq_len(nrow(reference))) {
    ids <- hits[[i]]
    ids <- ids[keep[ids]]
    if (!length(ids)) next
    prediction <- st_union(pred_geom[ids])
    pred_area <- as.numeric(st_area(prediction))
    shared <- suppressWarnings(st_intersection(ref_geom[i], prediction))
    common_area <- sum(as.numeric(st_area(shared)))
    out[i, ] <- c(length(ids), as.integer(common_area > 0),
                  common_area / reference_area[i],
                  common_area / (reference_area[i] + pred_area - common_area))
  }
  as.data.frame(out)
}

model_scores <- lapply(thresholds, function(t) {
  keep <- raw_area >= t
  list(threshold = t, count = sum(keep), area = sum(raw_area[keep]),
       score = score_references(raw_geom, raw_hits, keep))
})
base <- model_scores[[1]]$score
at_selected <- model_scores[[which(thresholds == selected_area)]]$score
final <- score_references(verified_geom, verified_hits, rep(TRUE, nrow(verified)))

# 3. Threshold selection and workload ---------------------------------------
# No-loss means no individual reference drops below any of the FOUR criteria.
threshold_results <- do.call(rbind, lapply(model_scores, function(x) {
  s <- x$score
  data.frame(
    threshold_m2 = x$threshold,
    candidates = x$count,
    candidates_removed_pct = 100 * (1 - x$count / nrow(raw)),
    predicted_area_retained_pct = 100 * x$area / sum(raw_area),
    overlap_n = sum(s$overlap > 0),
    coverage50_n = sum(s$coverage >= 0.5),
    iou50_n = sum(s$iou >= 0.5),
    iou75_n = sum(s$iou >= 0.75),
    lost_overlap_n = sum(base$overlap > 0 & s$overlap == 0),
    lost_coverage50_n = sum(base$coverage >= 0.5 & s$coverage < 0.5),
    lost_iou50_n = sum(base$iou >= 0.5 & s$iou < 0.5),
    lost_iou75_n = sum(base$iou >= 0.75 & s$iou < 0.75)
  )
}))
no_loss <- with(threshold_results,
                lost_overlap_n == 0 & lost_coverage50_n == 0 &
                  lost_iou50_n == 0 & lost_iou75_n == 0)
largest_loss_free <- max(threshold_results$threshold_m2[no_loss])
chosen <- threshold_results[threshold_results$threshold_m2 == selected_area, ]

# 4. Verification stage, size and proximity ---------------------------------
stages <- data.frame(
  stage = c("Raw model", "After area threshold", "Final single-observer inventory"),
  polygons = c(nrow(raw), sum(raw_area >= selected_area), nrow(verified)),
  total_area_m2 = c(sum(raw_area), sum(raw_area[raw_area >= selected_area]),
                    sum(verified_area)),
  reference_overlap = c(sum(base$overlap), sum(at_selected$overlap),
                        sum(final$overlap)),
  reference_iou50 = c(sum(base$iou >= 0.5), sum(at_selected$iou >= 0.5),
                      sum(final$iou >= 0.5))
)

# Distances are zero for overlaps and positive otherwise.
distance_final <- rep(0, nrow(reference))
missing_final <- which(final$overlap == 0)
if (length(missing_final)) {
  nearest <- st_nearest_feature(reference[missing_final, ], verified)
  distance_final[missing_final] <- as.numeric(st_distance(
    reference[missing_final, ], verified[nearest, ], by_element = TRUE))
}
group <- ifelse(final$overlap > 0, "Overlapping",
                ifelse(distance_final <= near_m, "Nearby (<=10 m)",
                       "Isolated (>10 m)"))

summarise_size <- function(g) {
  a <- reference_area[group == g]
  data.frame(group = g, n = length(a), median_m2 = median(a),
             mean_m2 = mean(a), min_m2 = min(a), max_m2 = max(a))
}
size_groups <- do.call(rbind, lapply(
  c("Overlapping", "Nearby (<=10 m)", "Isolated (>10 m)"), summarise_size))

# These tests describe associations; they cannot identify observer mechanisms.
size_test <- wilcox.test(reference_area[final$overlap == 0],
                         reference_area[final$overlap > 0],
                         alternative = "less", exact = FALSE)
isolated_test <- wilcox.test(reference_area[group == "Isolated (>10 m)"],
                             reference_area[group == "Overlapping"],
                             alternative = "less", exact = FALSE)
size_fit <- glm(as.integer(final$overlap > 0) ~ log2(reference_area),
                family = binomial())
size_or <- exp(unname(coef(size_fit)[2]))
size_p <- coef(summary(size_fit))[2, 4]

reference_audit <- data.frame(
  outline_no = reference$outline_no,
  reference_area_m2 = reference_area,
  final_group = group,
  distance_to_final_m = distance_final,
  raw_n_at_025 = at_selected$parts,
  raw_coverage_pct = 100 * at_selected$coverage,
  raw_iou_at_025 = at_selected$iou,
  final_n = final$parts,
  final_coverage_pct = 100 * final$coverage,
  final_iou = final$iou
)
priority <- reference_audit[group == "Isolated (>10 m)", ]
priority <- priority[order(-priority$reference_area_m2), ]
priority5 <- head(priority, 5)

# 4b. River-distance dependence, including within-20-m counts ----------------
# Measures geometric distance from each reference POLYGON to nearest mapped
# river line; zero for a polygon touching the river. This is NOT downstream
# river connectivity or transport distance. Reference sample size: 115.
# Compare raw-model delineation and final visual retention separately and
# adjust for reference polygon size to reduce size confounding.
river_analysis <- NULL
if (!file.exists(river_file)) {
  stop("River shapefile not found: ", river_file,
       "\nCheck that .shp, .shx, .dbf, and .prj exist together.")
}
river_available <- TRUE

if (river_available) {
  rivers <- st_read(river_file, quiet = TRUE)
  if (!all(sf::st_geometry_type(rivers) %in% c("LINESTRING", "MULTILINESTRING", "GEOMETRYCOLLECTION"))) {
    stop("River file must contain line geometries. Check: ", river_file)
  }
  if (is.na(st_crs(rivers))) stop("River layer has no defined CRS")
  rivers <- st_make_valid(st_transform(rivers, crs_m))
  rivers <- suppressWarnings(st_collection_extract(rivers, "LINESTRING"))
  rivers <- rivers[!st_is_empty(rivers), ]
  if (!nrow(rivers)) stop("No valid river lines found in river_file")
  message("River layer loaded: ", nrow(rivers), " line features in EPSG:", crs_m,
          ". Confirm this network covers the entire mapping area.")
  
  nearest_river <- st_nearest_feature(reference, rivers)
  river_distance_m <- as.numeric(st_distance(
    reference, rivers[nearest_river, ], by_element = TRUE))
  if (any(!is.finite(river_distance_m))) stop("Missing river distances")
  reference_audit$river_distance_m <- river_distance_m
  reference_audit$within_20m <- river_distance_m <= near_river_m
  
  # Use equal-frequency quartiles to avoid sparse, arbitrarily chosen bins.
  quartile_number <- pmin(4L, ceiling(
    rank(river_distance_m, ties.method = "first") * 4 / nrow(reference)))
  river_quartile <- factor(paste0("Q", quartile_number),
                           levels = paste0("Q", 1:4))
  reference_audit$river_quartile <- as.character(river_quartile)
  
  river_df <- data.frame(
    distance_100m = river_distance_m / 100,
    reference_area_m2 = reference_area,
    quartile = river_quartile,
    raw_iou50 = as.integer(at_selected$iou >= 0.5),
    final_overlap = as.integer(final$overlap > 0),
    final_iou50 = as.integer(final$iou >= 0.5))
  
  wilson <- function(x, n) {
    z <- 1.95996398454005
    a <- x/n; den <- 1 + z*z/n
    center <- (a + z*z/(2*n))/den
    half <- z*sqrt(a*(1-a)/n + z*z/(4*n*n))/den
    c(lower = max(0,center-half), upper = min(1,center+half))
  }
  measures <- c(raw_iou50 = "Raw IoU >=0.50",
                final_overlap = "Final overlap",
                final_iou50 = "Final IoU >=0.50")
  river_cells <- do.call(rbind, lapply(levels(river_quartile), function(q) {
    ii <- which(river_df$quartile == q)
    do.call(rbind, lapply(names(measures), function(m) {
      k <- sum(river_df[[m]][ii]); n <- length(ii); ci <- wilson(k,n)
      data.frame(quartile = q, n = n,
                 min_distance_m = min(river_distance_m[ii]),
                 median_distance_m = median(river_distance_m[ii]),
                 max_distance_m = max(river_distance_m[ii]),
                 criterion = measures[[m]], successes = k,
                 percent = 100*k/n,
                 lower_95 = 100*ci["lower"], upper_95 = 100*ci["upper"],
                 row.names = NULL)
    }))
  }))
  
  # Exploratory, size-adjusted logistic models. The raw overlap outcome is
  # constant (115/115) and thus cannot be modelled; use raw IoU >= 0.50.
  # Retain fitted models for adjusted probability curves (Fig. 7).
  river_fits <- setNames(lapply(names(measures), function(m) {
    y <- river_df[[m]]
    if (length(unique(y)) != 2 || sd(river_df$distance_100m) == 0)
      return(NULL)
    tryCatch(glm(y ~ distance_100m + log2(reference_area_m2),
                 data = river_df, family = binomial()),
             error = function(e) NULL)
  }), names(measures))
  
  river_models <- do.call(rbind, lapply(names(measures), function(m) {
    fit <- river_fits[[m]]
    if (is.null(fit)) return(data.frame(
      criterion = measures[[m]], or_100m = NA_real_,
      ci_low = NA_real_, ci_high = NA_real_, p = NA_real_))
    b <- summary(fit)$coefficients["distance_100m", ]
    data.frame(criterion = measures[[m]],
               or_100m = exp(b["Estimate"]),
               ci_low = exp(b["Estimate"] - 1.96 * b["Std. Error"]),
               ci_high = exp(b["Estimate"] + 1.96 * b["Std. Error"]),
               p = b["Pr(>|z|)"], row.names = NULL)
  }))
  
  # Predict at median reference-pile area, holding size constant. Lines are
  # exploratory associations, not estimates of causal river effects.
  river_range <- range(river_distance_m)
  river_grid <- seq(river_range[1], river_range[2], length.out = 120)
  area_typical <- median(reference_area)
  adjusted_curves <- do.call(rbind, lapply(names(measures), function(m) {
    fit <- river_fits[[m]]
    if (is.null(fit)) return(NULL)
    nd <- data.frame(distance_100m = river_grid / 100,
                     reference_area_m2 = area_typical)
    pr <- predict(fit, newdata = nd, type = "link", se.fit = TRUE)
    data.frame(distance_m = river_grid,
               criterion = measures[[m]],
               percent = 100 * plogis(pr$fit),
               lower_95 = 100 * plogis(pr$fit - 1.96 * pr$se.fit),
               upper_95 = 100 * plogis(pr$fit + 1.96 * pr$se.fit))
  }))
  # RQ8. Distribution of all 735 visually curated polygons relative to river.
  # Use the SAME river network, distance definition, and fixed bins for raw
  # filtered candidates, final verified polygons, and reference polygons.
  # Counts in different groups are not comparable detection sensitivities:
  # candidate polygons can be split/merged/reshaped during subsequent work.
  river_distance <- function(polygons) {
    j <- st_nearest_feature(polygons, rivers)
    as.numeric(st_distance(polygons, rivers[j, ], by_element = TRUE))
  }
  
  candidate_idx <- which(raw_area >= selected_area)
  candidates <- raw[candidate_idx, ]
  candidate_distance_m <- river_distance(candidates)
  verified_distance_m <- river_distance(verified)
  
  # RQ9. Count polygons <=20 m versus >20 m from a mapped river. The raw
  # inventory contains ~466,000 small model-generated POLYGONS, so calculate
  # their proximity in batches to avoid constructing a full distance matrix.
  # This uses polygon-to-river distance, NOT centroid-to-river distance.
  near20_raw <- logical(nrow(raw))
  batch_size <- 50000L
  for (start in seq.int(1L, nrow(raw), by = batch_size)) {
    ids <- start:min(nrow(raw), start + batch_size - 1L)
    near20_raw[ids] <- lengths(st_is_within_distance(
      raw[ids, ], rivers, dist = near_river_m)) > 0L
  }
  
  within20 <- list(
    `Manual reference` = river_distance_m <= near_river_m,
    `Raw model (unfiltered)` = near20_raw,
    `Raw model (>=0.25 m2)` = near20_raw[candidate_idx],
    `Visually verified` = verified_distance_m <= near_river_m
  )
  river20 <- do.call(rbind, lapply(names(within20), function(nm) {
    v <- within20[[nm]]
    total <- length(v)
    near <- sum(v)
    data.frame(
      inventory = nm,
      total_n = total,
      within_20m_n = near,
      within_20m_pct = 100 * near / total,
      beyond_20m_n = total - near,
      beyond_20m_pct = 100 * (total - near) / total,
      row.names = NULL
    )
  }))
  stopifnot(all(river20$within_20m_n + river20$beyond_20m_n == river20$total_n))
  cat("\nRIVER PROXIMITY <=20 M VERSUS >20 M (TABLE VIII)\n")
  print(river20, row.names = FALSE, digits = 4)
  write.csv(river20, file.path(out_dir, "river_within_20m_counts.csv"),
            row.names = FALSE)
  
  # Positive-area spatial intersection, NOT a verified decision log.
  candidate_overlap_final <- lengths(st_relate(
    candidates, verified, pattern = "2********", sparse = TRUE)) > 0
  
  river_breaks <- c(-Inf, 25, 50, 100, 200, Inf)
  river_labels <- c("0-25", "25-50", "50-100", "100-200", ">200")
  river_band <- function(x) cut(x, breaks = river_breaks,
                                labels = river_labels, right = TRUE)
  candidate_band <- river_band(candidate_distance_m)
  verified_band <- river_band(verified_distance_m)
  reference_band <- river_band(river_distance_m)
  
  count_by_band <- function(z) as.integer(table(factor(z, levels = river_labels)))
  reference_by_band <- count_by_band(reference_band)
  candidate_by_band <- count_by_band(candidate_band)
  verified_by_band <- count_by_band(verified_band)
  candidate_overlap_by_band <- vapply(river_labels, function(b) {
    sum(candidate_overlap_final[candidate_band == b])
  }, integer(1))
  
  river_inventory <- data.frame(
    band_m = river_labels,
    reference_n = reference_by_band,
    candidate_n = candidate_by_band,
    verified_n = verified_by_band,
    reference_share_pct = 100 * reference_by_band / nrow(reference),
    candidate_share_pct = 100 * candidate_by_band / length(candidate_idx),
    verified_share_pct = 100 * verified_by_band / nrow(verified),
    candidates_intersect_verified_n = candidate_overlap_by_band,
    candidate_intersection_pct = ifelse(candidate_by_band > 0,
                                        100 * candidate_overlap_by_band / pmax(candidate_by_band, 1), NA_real_),
    relative_verified_share = ifelse(candidate_by_band > 0,
                                     (verified_by_band / nrow(verified)) / (candidate_by_band / length(candidate_idx)),
                                     NA_real_)
  )
  
  verified_river_audit <- data.frame(
    verified_row = seq_len(nrow(verified)),
    waste_id = if ("waste_id" %in% names(verified))
      as.character(verified$waste_id) else as.character(seq_len(nrow(verified))),
    verified_area_m2 = verified_area,
    river_distance_m = verified_distance_m,
    river_band = as.character(verified_band)
  )
  candidate_river_audit <- data.frame(
    raw_row = candidate_idx,
    raw_area_m2 = raw_area[candidate_idx],
    river_distance_m = candidate_distance_m,
    river_band = as.character(candidate_band),
    intersects_final_inventory = candidate_overlap_final
  )
  
  river_analysis <- list(cells = river_cells, models = river_models,
                         adjusted = adjusted_curves,
                         distance_m = river_distance_m,
                         inventory = river_inventory,
                         verified_distances_m = verified_distance_m,
                         candidate_distances_m = candidate_distance_m,
                         verified_audit = verified_river_audit,
                         candidate_audit = candidate_river_audit,
                         within20 = river20)
  cat("\nRIVER DISTANCE OF AREA-FILTERED CANDIDATES AND 735 VERIFIED POLYGONS (TABLE VII)\n")
  print(river_inventory, row.names = FALSE, digits = 3)
  cat("\nRIVER DISTANCE: MEDIAN (M)\n")
  print(c(reference = median(river_distance_m),
          candidates = median(candidate_distance_m),
          verified = median(verified_distance_m)))
  write.csv(river_inventory, file.path(out_dir, "river_inventory_distance_bands.csv"),
            row.names = FALSE)
  write.csv(verified_river_audit, file.path(out_dir, "verified_polygon_river_distances.csv"),
            row.names = FALSE)
  write.csv(candidate_river_audit, file.path(out_dir, "filtered_candidate_river_distances.csv"),
            row.names = FALSE)
  
  cat("\nRIVER-DISTANCE PERFORMANCE BY REFERENCE QUARTILE (TABLE V)\n")
  print(river_cells, row.names = FALSE)
  cat("\nEXPLORATORY AREA-ADJUSTED ASSOCIATIONS (TABLE VI)\n")
  print(river_models, row.names = FALSE)
  write.csv(river_cells, file.path(out_dir, "river_distance_quartiles.csv"),
            row.names = FALSE)
  write.csv(river_models, file.path(out_dir, "river_distance_adjusted_models.csv"),
            row.names = FALSE)
  if (!is.null(adjusted_curves)) write.csv(
    adjusted_curves, file.path(out_dir, "river_distance_adjusted_predictions.csv"),
    row.names = FALSE)
} else {
  message("RQ7-RQ8 pending: set river_file to a river LINE layer covering the full study area.")
}


# 4c. Cumulative river-distance distributions: TWO inventories only ---------
# Research question: How many mapped waste polygons lie at or within any
# distance d from a river? N(d) = sum_i I(distance_i <= d).
# The curves use EVERY observed polygon distance (continuous x-axis), not
# distance bands. They are empirical cumulative distributions (step curves).
# Polygon-to-river distance is zero when any part of a polygon touches a river.
# A cumulative count always rises with d, by definition; it is NOT a density
# or waste-risk gradient and is not corrected for buffer area / land availability.

if (!is.null(river_analysis)) {
  ref_dist <- river_analysis$distance_m
  ver_dist <- river_analysis$verified_distances_m
  
  make_cumulative <- function(x, inventory) {
    stopifnot(all(is.finite(x)), all(x >= 0))
    x <- sort(as.numeric(x))
    data.frame(
      inventory = inventory,
      distance_m = c(0, x),
      cumulative_n = c(0L, seq_along(x)),
      cumulative_pct = 100 * c(0L, seq_along(x)) / length(x)
    )
  }
  
  manual_label <- paste0("Manual reference (n=", length(ref_dist), ")")
  verified_label <- paste0("Visually verified (n=", length(ver_dist), ")")
  cumulative_curve <- rbind(
    make_cumulative(ref_dist, manual_label),
    make_cumulative(ver_dist, verified_label)
  )
  
  # Values at reporting distances are calculated from the original distances,
  # not read from the figures or approximated by interpolation.
  report_distances_m <- c(0, 10, 20, 25, 50, 100, 200, 500)
  cumulative_table <- do.call(rbind, lapply(report_distances_m, function(d) {
    ref_n <- sum(ref_dist <= d)
    ver_n <- sum(ver_dist <= d)
    data.frame(
      distance_m = d,
      reference_n = ref_n,
      reference_pct = 100 * ref_n / length(ref_dist),
      verified_n = ver_n,
      verified_pct = 100 * ver_n / length(ver_dist)
    )
  }))
  
  cumulative_quantiles <- data.frame(
    inventory = c("Manual reference", "Visually verified"),
    total_n = c(length(ref_dist), length(ver_dist)),
    median_distance_m = c(median(ref_dist), median(ver_dist)),
    q75_distance_m = c(unname(quantile(ref_dist, .75)),
                       unname(quantile(ver_dist, .75))),
    q90_distance_m = c(unname(quantile(ref_dist, .90)),
                       unname(quantile(ver_dist, .90)))
  )
  
  cat("\nCUMULATIVE WASTE POLYGONS BY RIVER DISTANCE (TABLE IX)\n")
  print(cumulative_table, row.names = FALSE, digits = 4)
  cat("\nRIVER DISTANCE QUANTILES (REFERENCE AND VERIFIED)\n")
  print(cumulative_quantiles, row.names = FALSE, digits = 4)
  write.csv(cumulative_table,
            file.path(out_dir, "river_cumulative_counts_table_IX.csv"),
            row.names = FALSE)
  write.csv(cumulative_curve,
            file.path(out_dir, "river_cumulative_each_polygon.csv"),
            row.names = FALSE)
  write.csv(cumulative_quantiles,
            file.path(out_dir, "river_cumulative_distance_quantiles.csv"),
            row.names = FALSE)
}

# 5. Professional tables: console, CSV, Word --------------------------------
fmt_n <- function(x) format(x, big.mark = ",", scientific = FALSE, trim = TRUE)
fmt_t <- function(x) {
  if (x == 0) return("0")
  z <- sprintf("%.3f", x)
  sub("\\.$", "", sub("0+$", "", z))
}

T1 <- data.frame(
  `Min. area (m2)` = vapply(thresholds, fmt_t, character(1)),
  `Candidates (n)` = fmt_n(threshold_results$candidates),
  `Removed (%)` = sprintf("%.2f", threshold_results$candidates_removed_pct),
  `Area retained (%)` = sprintf("%.1f", threshold_results$predicted_area_retained_pct),
  `Overlap (n)` = threshold_results$overlap_n,
  `Coverage >=50% (n)` = threshold_results$coverage50_n,
  `IoU >=0.50 (n)` = threshold_results$iou50_n,
  `IoU >=0.75 (n)` = threshold_results$iou75_n,
  check.names = FALSE
)
T2 <- data.frame(
  `Inventory stage` = stages$stage,
  `Polygons (n)` = fmt_n(stages$polygons),
  `Mapped area (m2)` = fmt_n(round(stages$total_area_m2)),
  `Reference overlap (n)` = stages$reference_overlap,
  `Reference IoU >=0.50 (n)` = stages$reference_iou50,
  check.names = FALSE
)
T3 <- data.frame(
  `Reference group` = size_groups$group,
  `n` = size_groups$n,
  `Median (m2)` = sprintf("%.2f", size_groups$median_m2),
  `Mean (m2)` = sprintf("%.2f", size_groups$mean_m2),
  `Minimum (m2)` = sprintf("%.2f", size_groups$min_m2),
  `Maximum (m2)` = sprintf("%.2f", size_groups$max_m2),
  check.names = FALSE
)
T4 <- data.frame(
  `Outline` = priority5$outline_no,
  `Reference area (m2)` = sprintf("%.2f", priority5$reference_area_m2),
  `Raw parts >=0.25` = priority5$raw_n_at_025,
  `Raw coverage (%)` = sprintf("%.1f", priority5$raw_coverage_pct),
  `Raw IoU` = sprintf("%.3f", priority5$raw_iou_at_025),
  `Nearest verified (m)` = sprintf("%.1f", priority5$distance_to_final_m),
  check.names = FALSE
)

tables <- list(
  list(id = "I", title = "Minimum-area threshold sensitivity",
       data = T1, note = "Overlap, coverage and IoU are measured against 115 manually delineated reference polygons."),
  list(id = "II", title = "Model filtering and final inventory",
       data = T2, note = "The final stage includes single-observer visual curation; intervening processing is not independently reconstructed."),
  list(id = "III", title = "Reference-pile area by final inventory correspondence",
       data = T3, note = "Nearby means no positive-area overlap but a verified polygon is within 10 m; proximity is not a confirmed match."),
  list(id = "IV", title = "Five largest isolated reference piles",
       data = T4, note = "Raw-model metrics are computed after 0.25-m2 polygon filtering. These piles have no verified polygon within 10 m.")
)

if (!is.null(river_analysis)) {
  cell <- river_analysis$cells
  compact <- data.frame(
    Quartile = paste0("Q", 1:4),
    `Distance range (m)` = vapply(paste0("Q", 1:4), function(q) {
      a <- cell[cell$quartile == q, ][1, ]
      sprintf("%.1f–%.1f", a$min_distance_m, a$max_distance_m)
    }, character(1)),
    `Reference n` = vapply(paste0("Q", 1:4), function(q) {
      cell$n[cell$quartile == q][1]
    }, integer(1)),
    `Raw IoU >=0.50` = vapply(paste0("Q",1:4), function(q) {
      z <- cell[cell$quartile==q & cell$criterion=="Raw IoU >=0.50", ]
      sprintf("%d (%.1f%%)", z$successes, z$percent)
    }, character(1)),
    `Final overlap` = vapply(paste0("Q",1:4), function(q) {
      z <- cell[cell$quartile==q & cell$criterion=="Final overlap", ]
      sprintf("%d (%.1f%%)", z$successes, z$percent)
    }, character(1)),
    `Final IoU >=0.50` = vapply(paste0("Q",1:4), function(q) {
      z <- cell[cell$quartile==q & cell$criterion=="Final IoU >=0.50", ]
      sprintf("%d (%.1f%%)", z$successes, z$percent)
    }, character(1)), check.names = FALSE)
  mm <- river_analysis$models
  fitted <- data.frame(
    `Criterion` = mm$criterion,
    `Adjusted OR per 100 m` = sprintf("%.2f", mm$or_100m),
    `95% CI` = sprintf("%.2f–%.2f",mm$ci_low,mm$ci_high),
    `p` = sprintf("%.3f",mm$p), check.names = FALSE)
  tables <- c(tables, list(
    list(id = "V", title = "Reference correspondence by distance from river",
         data = compact, note = "Distance is minimum Euclidean polygon-to-river-line distance; quartiles are based on 115 reference polygons; counts are not sensitivity estimates for the full mapping area."),
    list(id = "VI", title = "Area-adjusted association with river distance",
         data = fitted, note = "Exploratory logistic models adjust for log2 reference-pile area. OR is per 100 m farther from the mapped river. These are associations, not causal effects.")))
  v <- river_analysis$inventory
  verified_table <- data.frame(
    `Distance (m)` = v$band_m,
    `Reference (n)` = v$reference_n,
    `Filtered candidates (n)` = fmt_n(v$candidate_n),
    `Verified (n)` = v$verified_n,
    `Candidate share (%)` = sprintf("%.1f", v$candidate_share_pct),
    `Verified share (%)` = sprintf("%.1f", v$verified_share_pct),
    `Candidates intersecting final (n)` = fmt_n(v$candidates_intersect_verified_n),
    `Intersecting (%)` = sprintf("%.1f", v$candidate_intersection_pct),
    check.names = FALSE
  )
  tables[[length(tables) + 1L]] <- list(
    id = "VII", title = "River distance of visually curated and candidate polygons",
    data = verified_table,
    note = paste0("Fixed Euclidean polygon-to-river bands across the full mapping extent. ",
                  "Shares are within inventory; candidate intersections are spatial only, ",
                  "not documented reviewer decisions or true detection sensitivity."))
  
  w <- river_analysis$within20
  T8 <- data.frame(
    `Inventory` = w$inventory,
    `Total (n)` = fmt_n(w$total_n),
    `Within 20 m (n)` = fmt_n(w$within_20m_n),
    `Within 20 m (%)` = sprintf("%.1f", w$within_20m_pct),
    `Beyond 20 m (n)` = fmt_n(w$beyond_20m_n),
    `Beyond 20 m (%)` = sprintf("%.1f", w$beyond_20m_pct),
    check.names = FALSE
  )
  tables[[length(tables) + 1L]] <- list(
    id = "VIII", title = "Waste polygon proximity to mapped rivers (20-m criterion)",
    data = T8,
    note = paste0("Within means minimum distance from any part of a polygon ",
                  "to the mapped river line is <=20 m; beyond means >20 m. ",
                  "Totals are polygon counts, not distinct physical piles. ",
                  "Percentages use each inventory's own total as denominator; ",
                  "river mapping must cover the full study extent."))
}


# Table IX. Cumulative counts at selected distances; the underlying curves
# retain every individual polygon's measured river distance.
if (!is.null(river_analysis)) {
  x <- cumulative_table
  T9 <- data.frame(
    `Maximum river distance (m)` = fmt_n(x$distance_m),
    `Manual (n)` = fmt_n(x$reference_n),
    `Manual (%)` = sprintf("%.1f", x$reference_pct),
    `Verified (n)` = fmt_n(x$verified_n),
    `Verified (%)` = sprintf("%.1f", x$verified_pct),
    check.names = FALSE
  )
  tables[[length(tables) + 1L]] <- list(
    id = "IX",
    title = "Cumulative mapped waste polygons by distance from rivers",
    data = T9,
    note = paste0(
      "Each row counts polygons whose minimum polygon-to-river distance is <= the stated distance; ",
      "percentages are relative to 115 manual-reference or 735 visually verified polygons. ",
      "These are empirical cumulative distributions, not waste density, ",
      "sampling-adjusted prevalence, model accuracy or independent physical pile counts."
    )
  )
}

for (tb in tables) {
  cat("\nTABLE ", tb$id, ". ", toupper(tb$title), "\n", sep = "")
  print(tb$data, row.names = FALSE, right = TRUE)
  cat("Note: ", tb$note, "\n", sep = "")
  write.csv(tb$data, file.path(out_dir, paste0("table_", tb$id, ".csv")),
            row.names = FALSE)
}

# 6. IEEE-style figures: vector PDF and 600-dpi PNG --------------------------
axis_labels <- vapply(thresholds, fmt_t, character(1))
x_levels <- factor(axis_labels, levels = axis_labels)
selected_x <- which(thresholds == selected_area)

paper_theme <- theme_classic(base_size = 10, base_family = "serif") + theme(
  axis.text.x = element_text(angle = 45, hjust = 1, size = 8.5),
  axis.title = element_text(size = 10),
  legend.position = "bottom", legend.title = element_blank(),
  legend.text = element_text(size = 9),
  plot.margin = margin(12, 18, 10, 12)
)

fig1 <- ggplot(data.frame(x = x_levels, n = threshold_results$candidates),
               aes(x, n, group = 1)) +
  geom_vline(xintercept = selected_x, colour = "grey45", linetype = "dashed") +
  geom_line(colour = "black", linewidth = 0.7) +
  geom_point(shape = 21, fill = "white", colour = "black", size = 2.6) +
  geom_point(data = data.frame(x = x_levels[selected_x],
                               n = threshold_results$candidates[selected_x]),
             shape = 21, fill = "black", colour = "black", size = 3.2) +
  scale_y_log10(labels = fmt_n) +
  labs(x = "Minimum polygon area (m²)",
       y = "Model-generated polygons remaining (log scale)") + paper_theme

reference_lines <- rbind(
  data.frame(x = x_levels, criterion = "Any overlap",
             n = threshold_results$overlap_n),
  data.frame(x = x_levels, criterion = "Coverage >=50%",
             n = threshold_results$coverage50_n),
  data.frame(x = x_levels, criterion = "IoU >=0.50",
             n = threshold_results$iou50_n)
)
reference_lines$criterion <- factor(reference_lines$criterion,
                                    levels = c("Any overlap", "Coverage >=50%", "IoU >=0.50"))

fig2 <- ggplot(reference_lines,
               aes(x, n, group = criterion, shape = criterion,
                   linetype = criterion)) +
  geom_vline(xintercept = selected_x, colour = "grey45", linetype = "dashed") +
  geom_line(colour = "black", linewidth = 0.75) +
  geom_point(colour = "black", size = 2.4) +
  scale_linetype_manual(values = c("solid", "dashed", "dotdash")) +
  scale_shape_manual(values = c(16, 17, 15)) +
  scale_y_continuous(limits = c(0, nrow(reference) + 5),
                     breaks = c(0, 25, 50, 75, 100, nrow(reference))) +
  labs(x = "Minimum polygon area (m²)",
       y = paste0("Reference waste piles (n = ", nrow(reference), ")")) + paper_theme

fig3 <- ggplot(size_groups, aes(x = group, y = n)) +
  geom_col(width = 0.55, fill = "grey75", colour = "black", linewidth = 0.5) +
  geom_text(aes(label = n), vjust = -0.6, size = 3.7, family = "serif") +
  scale_y_continuous(expand = expansion(mult = c(0, .14))) +
  labs(x = NULL, y = "Number of reference waste piles") +
  paper_theme + theme(axis.text.x = element_text(angle = 0, hjust = .5))

figures <- list(fig1, fig2, fig3)
figure_names <- c("candidate_count_by_threshold", "reference_detection_by_threshold",
                  "verified_correspondence_groups")
figure_captions <- c(
  "Fig. 1. Model-generated polygon candidates remaining across minimum-area thresholds. The dashed line marks 0.25 m².",
  "Fig. 2. Reference-pile correspondence across area thresholds. The dashed line marks 0.25 m²; positive-area overlap is the most permissive criterion.",
  "Fig. 3. Correspondence of 115 reference piles with the final curated inventory. Nearby piles are within 10 m without polygon overlap."
)
if (!is.null(river_analysis)) {
  df <- river_analysis$cells
  df$criterion <- factor(df$criterion,
                         levels = c("Raw IoU >=0.50", "Final overlap", "Final IoU >=0.50"))
  # FIG. 4: connected lines (rather than unrelated quartile points),
  # showing reference correspondence against increasing river distance.
  # Wilson intervals account for binomial count variability per quartile.
  q_medians <- vapply(paste0("Q", 1:4), function(q) {
    df$median_distance_m[df$quartile == q][1]
  }, numeric(1))
  q_labels <- paste0("Q", 1:4, "\n", round(q_medians), " m")
  df$quartile <- factor(df$quartile, levels = paste0("Q", 1:4))
  fig4 <- ggplot(df, aes(x = quartile, y = percent, group = criterion,
                         linetype = criterion, shape = criterion)) +
    geom_line(colour = "black", linewidth = 0.85, position = position_dodge(width = 0.12)) +
    geom_errorbar(aes(ymin = lower_95, ymax = upper_95),
                  colour = "grey35", linewidth = 0.45,
                  width = 0.12, position = position_dodge(width = 0.15)) +
    geom_point(colour = "black", fill = "white", size = 2.6,
               position = position_dodge(width = 0.15)) +
    scale_linetype_manual(values = c("solid", "dashed", "dotdash")) +
    scale_shape_manual(values = c(16, 17, 15)) +
    scale_x_discrete(labels = q_labels) +
    scale_y_continuous(limits = c(0, 100), breaks = c(0,25,50,75,100),
                       labels = function(x) paste0(x, "%")) +
    labs(x = "Reference-pile river-distance quartile (median distance)",
         y = "Reference piles meeting criterion (%)") + paper_theme +
    theme(axis.text.x = element_text(angle = 0, size = 9),
          legend.position = "bottom")
  figures[[4]] <- fig4
  figure_names <- c(figure_names, "reference_detection_river_distance_lines")
  figure_captions <- c(figure_captions,
                       "Fig. 4. Spatial correspondence of reference waste piles by increasing distance from the nearest mapped river (115 manually delineated outlines). Lines represent raw-model IoU >=0.50, final-inventory overlap, and final IoU >=0.50; vertical bars are 95% Wilson intervals. Quartiles show median distance. A positive-area intersection alone does not establish accurate detection.")
}
if (!is.null(river_analysis)) {
  v <- river_analysis$inventory
  plot_inventory <- rbind(
    data.frame(band = factor(v$band_m, levels = v$band_m),
               inventory = "Area-filtered model", percent = v$candidate_share_pct),
    data.frame(band = factor(v$band_m, levels = v$band_m),
               inventory = "Visually verified", percent = v$verified_share_pct),
    data.frame(band = factor(v$band_m, levels = v$band_m),
               inventory = "Manual reference", percent = v$reference_share_pct)
  )
  plot_inventory$inventory <- factor(plot_inventory$inventory,
                                     levels = c("Area-filtered model", "Visually verified", "Manual reference"))
  fig5 <- ggplot(plot_inventory,
                 aes(band, percent, group = inventory, linetype = inventory, shape = inventory)) +
    geom_line(colour = "black", linewidth = .75) +
    geom_point(colour = "black", size = 2.4) +
    scale_linetype_manual(values = c("solid", "dashed", "dotdash")) +
    scale_shape_manual(values = c(16,17,15)) +
    labs(x = "Minimum polygon-to-river distance (m)",
         y = "Share within each inventory (%)") + paper_theme +
    theme(axis.text.x = element_text(angle = 0, hjust = .5))
  figures[[5]] <- fig5
  figure_names <- c(figure_names, "candidate_verified_reference_river_distances")
  figure_captions <- c(figure_captions,
                       "Fig. 5. Distance-to-river distributions of manually delineated references, area-filtered model candidates and the final single-observer inventory. Within-inventory percentages describe spatial representation, not sensitivity, review effort or reviewer bias.")
  
  fig6 <- ggplot(v, aes(x = factor(band_m, levels = band_m),
                        y = candidate_intersection_pct)) +
    geom_col(fill = "grey75", colour = "black", width = .65) +
    geom_text(aes(label = sprintf("%.1f%%", candidate_intersection_pct)),
              vjust = -.35, size = 3.2, family = "serif", na.rm = TRUE) +
    scale_y_continuous(limits = c(0, 105), breaks = c(0,25,50,75,100)) +
    labs(x = "Minimum polygon-to-river distance (m)",
         y = "Filtered candidates intersecting final inventory (%)") + paper_theme +
    theme(axis.text.x = element_text(angle = 0, hjust = .5))
  figures[[6]] <- fig6
  figure_names <- c(figure_names, "candidate_final_spatial_overlap_by_river")
  figure_captions <- c(figure_captions,
                       "Fig. 6. Percentage of filtered model polygons whose footprint intersects any verified polygon, by river-distance band. Intersections do not constitute exact one-to-one retention or human verification decisions.")
  
  # FIG. 7: Continuous size-adjusted probability curves, not bin-wise lines.
  # The reference area is fixed at its observed median for all predictions.
  if (!is.null(river_analysis$adjusted)) {
    fit_curves <- river_analysis$adjusted
    fit_curves$criterion <- factor(fit_curves$criterion,
                                   levels = c("Raw IoU >=0.50", "Final overlap", "Final IoU >=0.50"))
    fig7 <- ggplot(fit_curves, aes(x = distance_m, y = percent,
                                   linetype = criterion, group = criterion)) +
      geom_ribbon(aes(ymin = lower_95, ymax = upper_95, group = criterion),
                  fill = "grey65", alpha = 0.12, colour = NA) +
      geom_line(colour = "black", linewidth = 0.85) +
      scale_linetype_manual(values = c("solid", "dashed", "dotdash")) +
      scale_y_continuous(limits = c(0,100), breaks = c(0,25,50,75,100),
                         labels = function(x) paste0(x, "%")) +
      labs(x = "Minimum distance to mapped river (m)",
           y = "Adjusted probability of reference correspondence") +
      paper_theme + theme(axis.text.x = element_text(angle = 0))
    figures[[7]] <- fig7
    figure_names <- c(figure_names, "river_distance_area_adjusted_probability")
    figure_captions <- c(figure_captions, sprintf(paste0(
      "Fig. 7. Exploratory logistic-model predictions of reference correspondence ",
      "versus river distance, adjusted to the median reference-pile area ",
      "(%.2f m²). Shaded regions indicate pointwise 95%% confidence intervals. ",
      "The raw-model outcome is IoU >=0.50 because positive-area overlap ",
      "occurred for all 115 reference piles. Associations are not causal."),
      median(reference_area)))
  }
}
# FIG. 8. Within-inventory share within versus beyond 20 m of a river.
if (!is.null(river_analysis)) {
  w <- river_analysis$within20
  proximity_plot <- rbind(
    data.frame(inventory = w$inventory, category = "Within 20 m",
               percent = w$within_20m_pct),
    data.frame(inventory = w$inventory, category = "Beyond 20 m",
               percent = w$beyond_20m_pct)
  )
  proximity_plot$inventory <- factor(proximity_plot$inventory,
                                     levels = w$inventory)
  proximity_plot$category <- factor(proximity_plot$category,
                                    levels = c("Within 20 m", "Beyond 20 m"))
  fig8 <- ggplot(proximity_plot, aes(inventory, percent, fill = category)) +
    geom_col(width = 0.62, colour = "black", linewidth = 0.35) +
    geom_text(aes(label = sprintf("%.1f%%", percent)),
              position = position_stack(vjust = 0.5), size = 3.3,
              family = "serif") +
    scale_fill_manual(values = c("grey45", "grey88")) +
    scale_y_continuous(limits = c(0, 100), breaks = c(0, 25, 50, 75, 100),
                       labels = function(x) paste0(x, "%"),
                       expand = expansion(mult = c(0, 0))) +
    labs(x = NULL, y = "Share of polygons within inventory (%)") +
    paper_theme + theme(axis.text.x = element_text(angle = 15, hjust = 1))
  figures[[length(figures) + 1L]] <- fig8
  figure_names <- c(figure_names, "river_proximity_20m_inventory_shares")
  figure_captions <- c(figure_captions,
                       "Fig. 8. Proportion of mapped polygon candidates within 20 m versus more than 20 m from a river line in each inventory. Counts are provided in Table VIII. Raw model polygons include small segmentation fragments; polygon percentages should not be interpreted as probabilities of detecting physical waste accumulations.")
}

# FIGS. 9-10. Exact empirical cumulative distributions of polygon distance.
# Lines are step curves because N(d) changes at observed polygon distances.
# Display both count and percentage to separate inventory size (115 vs 735)
# from the relative spatial distribution.
if (!is.null(river_analysis)) {
  cumulative_curve$inventory <- factor(
    cumulative_curve$inventory,
    levels = c(manual_label, verified_label)
  )
  cumulative_styles <- c("solid", "dashed")
  fig9 <- ggplot(cumulative_curve,
                 aes(distance_m, cumulative_n, group = inventory,
                     linetype = inventory)) +
    geom_vline(xintercept = 20, colour = "grey65", linetype = "dotted") +
    geom_step(colour = "black", linewidth = .9, direction = "hv") +
    scale_linetype_manual(values = cumulative_styles) +
    scale_y_continuous(labels = fmt_n,
                       expand = expansion(mult = c(0, .05))) +
    labs(x = "Minimum polygon-to-river distance (m)",
         y = "Cumulative number of mapped polygons") +
    paper_theme + theme(axis.text.x = element_text(angle = 0))
  
  fig10 <- ggplot(cumulative_curve,
                  aes(distance_m, cumulative_pct, group = inventory,
                      linetype = inventory)) +
    geom_vline(xintercept = 20, colour = "grey65", linetype = "dotted") +
    geom_step(colour = "black", linewidth = .9, direction = "hv") +
    scale_linetype_manual(values = cumulative_styles) +
    scale_y_continuous(limits = c(0, 100),
                       breaks = c(0, 25, 50, 75, 100),
                       labels = function(x) paste0(x, "%")) +
    labs(x = "Minimum polygon-to-river distance (m)",
         y = "Cumulative percentage of mapped polygons") +
    paper_theme + theme(axis.text.x = element_text(angle = 0))
  
  figures <- c(figures, list(fig9, fig10))
  figure_names <- c(figure_names,
                    "river_distance_cumulative_counts",
                    "river_distance_cumulative_percent")
  figure_captions <- c(figure_captions,
                       "Fig. 9. Exact cumulative numbers of manually delineated reference and visually verified polygons within distance d of the nearest mapped river. The dotted vertical line indicates 20 m. Inventory sizes differ (115 and 735).",
                       "Fig. 10. Empirical cumulative distributions of minimum polygon-to-river distance in the manually delineated reference set (n = 115) and visually verified inventory (n = 735). The dotted line marks 20 m. The vertical separation describes relative spatial concentration, not observer bias or pile density.")
}

for (i in seq_along(figures)) {
  print(figures[[i]])
  stem <- file.path(out_dir, paste0("figure_", i, "_", figure_names[i]))
  ggsave(paste0(stem, ".pdf"), figures[[i]], width = 7.5, height = 4.3)
  ggsave(paste0(stem, ".png"), figures[[i]], width = 7.5, height = 4.3,
         dpi = 600, bg = "white")
}

# 7. Research questions and answers: derived from the calculations -----------
n_ref <- nrow(reference)
n_unmatched <- length(missing_final)
n_near <- sum(group == "Nearby (<=10 m)")
n_isolated <- sum(group == "Isolated (>10 m)")
n_split <- sum(final$parts > 1)
n_raw_missed <- sum(at_selected$overlap[missing_final] > 0)
n_missed_iou50 <- sum(at_selected$iou[missing_final] >= .50)
n_missed_iou75 <- sum(at_selected$iou[missing_final] >= .75)
original_manual_min <- min(reference_area)

research <- list(
  list(q = "RQ1. Which tested minimum-area threshold removes the most small predicted polygons without losing reference-pile correspondence?",
       a = sprintf("%.3g m² is the largest tested loss-free threshold: all %d/%d references retain overlap, %d retain >=50%% coverage, and %d retain IoU >=0.50. These values equal the unfiltered results. This does not establish performance for piles below %.2f m² (Table I; Fig. 2).",
                   largest_loss_free, chosen$overlap_n, n_ref, chosen$coverage50_n,
                   chosen$iou50_n, original_manual_min)),
  list(q = "RQ2. How much candidate-review workload does this threshold remove?",
       a = sprintf("At %.2f m², %s of %s model-generated polygons remain (%.2f%% fewer), retaining %.2f%% of their total mapped area. Candidate reduction is a proxy, NOT a measured reduction in review time (Table I; Fig. 1).",
                   selected_area, fmt_n(chosen$candidates), fmt_n(nrow(raw)),
                   chosen$candidates_removed_pct, chosen$predicted_area_retained_pct)),
  list(q = "RQ3. What remains after single-observer visual curation, and were reference detections lost?",
       a = sprintf("The final inventory contains %s polygons compared with %s after filtering, and %d/%d reference outlines retain overlap (Table II). Of the %d reference outlines absent from the final layer, %d still overlap filtered raw predictions; %d have raw IoU >=0.50 and %d have raw IoU >=0.75 (Table IV). The analysis does not establish which intermediate step excluded them.",
                   fmt_n(nrow(verified)), fmt_n(chosen$candidates),
                   sum(final$overlap), n_ref, n_unmatched, n_raw_missed,
                   n_missed_iou50, n_missed_iou75)),
  list(q = "RQ4. Are reference piles absent from the final inventory mainly smaller?",
       a = sprintf("The %d unmatched reference piles have median area %.2f m² versus %.2f m² for the %d overlapping piles (one-sided Wilcoxon p = %.4f); the odds ratio per doubling of area is %.2f (logistic p = %.4f). The association does not determine why a pile was excluded (Table III).",
                   n_unmatched, median(reference_area[final$overlap == 0]),
                   median(reference_area[final$overlap > 0]), sum(final$overlap),
                   size_test$p.value, size_or, size_p)),
  list(q = "RQ5. Could nearby piles, splitting, or position explain the disagreement?",
       a = sprintf("Of the %d unmatched references, %d are within 10 m of a verified polygon and %d are farther than 10 m; %d original outlines overlap multiple verified polygons (Table III; Fig. 3). The isolated group has median area %.2f m² and does not show a significant one-sided area shift relative to overlapping references (p = %.4f). Nearness and splitting are candidate explanations, not confirmed physical matches.",
                   n_unmatched, n_near, n_isolated, n_split,
                   median(reference_area[group == "Isolated (>10 m)"]),
                   isolated_test$p.value)),
  list(q = "RQ6. What does this comparison tell us about visual inspection, including MapSwipe-style screening?",
       a = paste0("Single-observer curation yields a much smaller inventory, but the evidence cannot separate justified false-positive removal from missed valid waste, differences in imagery or intervening processing (Tables II-IV). Fatigue, attention, observer bias and screening-interface limitations are hypotheses, not measured causes. No independent second observer, decision log, review order/time or adjudication data were available; inter-observer reliability or MapSwipe performance cannot be estimated. A blinded re-review of sampled retained and excluded candidates would be needed."))
)

if (!is.null(river_analysis)) {
  mm <- river_analysis$models
  raw_row <- mm[mm$criterion == "Raw IoU >=0.50", ]
  final_row <- mm[mm$criterion == "Final overlap", ]
  effect_direction <- function(or, p) {
    if (!is.finite(or) || !is.finite(p)) return("could not be estimated reliably")
    if (p >= 0.05) return("did not provide clear statistical evidence of an association")
    if (or < 1) return("indicated higher correspondence nearer the river")
    return("indicated higher correspondence farther from the river")
  }
  river_answer <- sprintf(paste0(
    "Figure 4 connects the observed proportions of the 115 reference piles ",
    "meeting each detection criterion across river-distance quartiles, ",
    "with 95%% Wilson intervals (Table V). Adjusting for pile area, the odds ",
    "ratio per 100 m farther from the river is %.2f (95%% CI %.2f to %.2f; ",
    "p = %.3f) for raw-model IoU >=0.50, which %s; and %.2f ",
    "(95%% CI %.2f to %.2f; p = %.3f) for final-inventory overlap, which %s ",
    "(Table VI; Fig. 7). Raw-model overlap itself is constant at 115/115; ",
    "these findings concern geometric agreement and final correspondence, ",
    "not causal effects or independently confirmed waste detection."),
    raw_row$or_100m,raw_row$ci_low,raw_row$ci_high,raw_row$p,
    effect_direction(raw_row$or_100m,raw_row$p),
    final_row$or_100m,final_row$ci_low,final_row$ci_high,final_row$p,
    effect_direction(final_row$or_100m,final_row$p))
} else {
  river_answer <- paste0(
    "Not yet tested: a river-line layer covering the entire waste-mapping ",
    "study area has not been supplied. Set river_file at the start of the script ",
    "and rerun to obtain Tables V–VI and Fig. 4. Analysis will distinguish ",
    "raw IoU >=0.50 from final-inventory overlap, show detection-rate ",
    "lines by river-distance quartile and adjusted probability curves, ",
    "and fit exploratory models adjusted for reference-pile area. Do not ",
    "interpret a Nasolo-only river layer as covering the full area.")
}
research[[length(research) + 1L]] <- list(
  q = "RQ7. Are reference waste piles nearer a river more likely to be accurately delineated by the model or represented in the final inventory?",
  a = river_answer)

if (!is.null(river_analysis)) {
  v <- river_analysis$inventory
  p_near <- sum(river_analysis$verified_distances_m <= 50) / nrow(verified) * 100
  p_candidate_near <- sum(river_analysis$candidate_distances_m <= 50) /
    length(river_analysis$candidate_distances_m) * 100
  river_verified_answer <- sprintf(paste0(
    "Of %d visually verified polygons, %.1f%% lie within 50 m of a mapped river ",
    "(median %.1f m). Among %s model candidates retained after the %.2f-m² ",
    "area threshold, %.1f%% lie within 50 m (median %.1f m). Table VII and ",
    "Fig. 5 compare each inventory's river-distance distribution. Fig. 6 ",
    "shows the percentage of candidate polygons spatially intersecting final ",
    "polygons by river-distance band. These comparisons are descriptive: ",
    "they cannot establish observer preference, fatigue, physical pile prevalence ",
    "or exact individual candidate retention without matching and a review log."),
    nrow(verified), p_near, median(river_analysis$verified_distances_m),
    fmt_n(sum(raw_area >= selected_area)), selected_area, p_candidate_near,
    median(river_analysis$candidate_distances_m))
} else {
  river_verified_answer <- paste0(
    "Not yet tested: provide a river-line layer covering the entire mapping ",
    "area using river_file. When supplied, Table VII and Figs. 5-6 compare ",
    "all 735 visually verified polygons with the area-filtered model candidates ",
    "and manually delineated references, using fixed distance bands. They ",
    "describe spatial representation rather than verified-candidate retention ",
    "or observer performance. No claim about river-related verification ",
    "bias or fatigue is possible without an independent review log.")
}
research[[length(research) + 1L]] <- list(
  q = paste0("RQ8. Does the river-distance distribution of visually verified waste ",
             "polygons differ from that of area-filtered model candidates?"),
  a = river_verified_answer)

if (!is.null(river_analysis)) {
  w <- river_analysis$within20
  describe_group <- function(nm) {
    i <- match(nm, w$inventory)
    sprintf("%s: %s/%s (%.1f%%) within 20 m and %s (%.1f%%) beyond 20 m",
            nm, fmt_n(w$within_20m_n[i]), fmt_n(w$total_n[i]),
            w$within_20m_pct[i], fmt_n(w$beyond_20m_n[i]), w$beyond_20m_pct[i])
  }
  twenty_answer <- paste0(
    paste(vapply(w$inventory, describe_group, character(1)), collapse = "; "),
    " (Table VIII; Fig. 8). These are within-inventory polygon proportions, ",
    "not detection sensitivity or observer retention probabilities."
  )
  research[[length(research) + 1L]] <- list(
    q = "RQ9. How many polygons in each inventory occur within 20 m of mapped rivers, and how many occur farther away?",
    a = twenty_answer)
}


if (!is.null(river_analysis)) {
  c20 <- cumulative_table[cumulative_table$distance_m == 20, ]
  research[[length(research) + 1L]] <- list(
    q = paste0(
      "RQ10. How does the cumulative number and share of mapped waste ",
      "polygons change continuously with distance from a river, for manual ",
      "reference and visually verified inventories separately?"
    ),
    a = sprintf(paste0(
      "At 20 m, %d/%d (%.1f%%) manual reference polygons and %d/%d ",
      "(%.1f%%) verified polygons are included within the river buffer. ",
      "The median polygon-to-river distance is %.1f m for the reference ",
      "sample and %.1f m for the visually verified inventory. ",
      "Table IX provides cumulative counts and shares at selected distances; ",
      "Figs. 9-10 trace the exact empirical cumulative distributions ",
      "at every polygon distance. The curves cannot establish that ",
      "waste density or physical pile prevalence decreases with distance, ",
      "because land area available at each distance, sample selection, ",
      "and fragmentation have not been accounted for."),
      c20$reference_n, nrow(reference), c20$reference_pct,
      c20$verified_n, nrow(verified), c20$verified_pct,
      median(river_analysis$distance_m),
      median(river_analysis$verified_distances_m))
  )
}

cat("\nRESEARCH QUESTIONS AND ANSWERS\n", strrep("=", 60), "\n", sep = "")
answers <- unlist(lapply(research, function(x) c(x$q, x$a, "")))
cat(paste(answers, collapse = "\n"), "\n")
writeLines(answers, file.path(out_dir, "research_questions_answers.txt"))

# 8. Export polished Word report: actual editable tables ---------------------
# Tables have booktabs-style horizontal rules, no vertical grid lines, Times
# New Roman text, and separate numbered captions and notes.
make_ft <- function(data) {
  ft <- flextable::flextable(data)
  ft <- flextable::theme_booktabs(ft)
  ft <- flextable::font(ft, fontname = "Times New Roman", part = "all")
  ft <- flextable::fontsize(ft, size = 9, part = "all")
  ft <- flextable::bold(ft, part = "header")
  ft <- flextable::padding(ft, padding = 4, part = "all")
  ft <- flextable::align(ft, j = 1, align = "left", part = "all")
  if (ncol(data) > 1) ft <- flextable::align(
    ft, j = 2:ncol(data), align = "right", part = "all")
  ft <- flextable::autofit(ft)
  flextable::set_table_properties(ft, layout = "autofit")
}

report <- officer::read_docx()
report <- officer::body_add_par(report,
                                "Waste mapping: minimum-area filtering and verification audit", style = "Title")
report <- officer::body_add_par(report,
                                "Reference-level spatial correspondence and river-distance distributions; polygon counts do not necessarily equal distinct physical waste accumulations.")

for (tb in tables) {
  report <- officer::body_add_par(report,
                                  paste0("TABLE ", tb$id, ". ", toupper(tb$title)), style = "heading 2")
  report <- flextable::body_add_flextable(report, make_ft(tb$data))
  report <- officer::body_add_par(report, paste0("Note. ", tb$note))
}

for (i in seq_along(figures)) {
  report <- officer::body_add_par(report, paste0("Figure ", i), style = "heading 2")
  pic <- file.path(out_dir, paste0("figure_", i, "_", figure_names[i], ".png"))
  report <- officer::body_add_img(report, src = pic, width = 7.2, height = 4.13)
  report <- officer::body_add_par(report, figure_captions[i])
}

report <- officer::body_add_break(report)
report <- officer::body_add_par(report, "Research questions and answers", style = "heading 1")
for (item in research) {
  report <- officer::body_add_par(report, item$q, style = "heading 2")
  report <- officer::body_add_par(report, item$a)
}
report <- officer::body_add_par(report, "Interpretation and limitations", style = "heading 2")
report <- officer::body_add_par(report,
                                "The reference set is not necessarily exhaustive or an independent accuracy standard. Positive-area overlap is permissive, while IoU evaluates geometry. The 0.25 m² filter was applied to individual polygonised detections, not to physical waste accumulations. The single-observer verification step has no independent reliability estimate; observer fatigue cannot be inferred from cross-sectional inventory comparisons. River-distance associations are exploratory and require a river layer covering the full mapping extent; they cannot establish causal river effects. Differences between the river-distance distributions of candidates and curated polygons do not establish individual candidate retention, observer bias or fatigue. Cumulative counts increase mechanically with distance and should not be confused with polygon density or risk unless the land area at each river distance is quantified. Confirm processing provenance and independently adjudicate a sample of exclusions before assigning causes.")

report <- officer::body_set_default_section(report,
                                            officer::prop_section(
                                              page_size = officer::page_size(orient = "landscape"),
                                              page_margins = officer::page_mar(top = 0.7, bottom = 0.7,
                                                                               left = 0.75, right = 0.75)))
report_path <- file.path(out_dir, "IEEE_tables_figures_research_answers.docx")
print(report, target = report_path)

# 9. Save detailed audits ----------------------------------------------------
write.csv(threshold_results, file.path(out_dir, "threshold_sensitivity_full.csv"),
          row.names = FALSE)
write.csv(reference_audit, file.path(out_dir, "reference_level_audit.csv"),
          row.names = FALSE)
reference$raw_iou_025 <- at_selected$iou
reference$raw_coverage_025 <- 100 * at_selected$coverage
reference$final_iou <- final$iou
reference$verification_group <- group
reference$nearest_verified_m <- distance_final
if (!is.null(river_analysis)) {
  reference$nearest_river_m <- river_analysis$distance_m
  reference$within_20m <- reference$nearest_river_m <= near_river_m
  reference$river_quartile <- as.character(river_quartile)
}
st_write(reference, file.path(out_dir, "reference_audit.gpkg"),
         layer = "references", delete_dsn = TRUE, quiet = TRUE)
if (!is.null(river_analysis)) {
  verified$river_dist_m <- river_analysis$verified_distances_m
  verified$within_20m <- verified$river_dist_m <= near_river_m
  verified$river_band <- river_analysis$verified_audit$river_band
  st_write(verified, file.path(out_dir, "verified_river_audit.gpkg"),
           layer = "verified", delete_dsn = TRUE, quiet = TRUE)
}
cat("\nSaved professional Word tables/report, figures, and audits to:\n", out_dir,
    "\n", sep = "")
