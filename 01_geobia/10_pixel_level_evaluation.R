################################################################################
# GeoBIA evaluated on the same basis as the deep-learning models
################################################################################
# Why this script exists
# ----------------------
# The GeoBIA numbers in the manuscript are not comparable with the U-Net and
# pretrained-backbone numbers, for two reasons:
#
#  1. BASIS. GeoBIA accuracy was a caret confusion matrix over polygon FEATURES.
#     The deep models are evaluated at PIXEL level, pooled over a spatially
#     clean held-out partition. A segment-level F1 is a different quantity.
#
#  2. SEGMENTS. Everything downstream of 01_mask_vectorization.R used polygons
#     obtained by vectorising the reference MASK (.../vectors_with_var). Those
#     polygon boundaries ARE the reference boundaries, so classifying them and
#     rasterising back reproduces the mask wherever the class is right. That is
#     an oracle-segmentation upper bound, not an automated method, and it is why
#     precision comes out at or near 1.000.
#
# This script evaluates both conditions on the deep-learning basis:
#
#     --segments otb     OTB meanshift segmentation (the real automated method)
#     --segments oracle  mask-derived polygons (the upper bound, for contrast)
#
# Held constant with the deep models:
#   partition        data/tiled_data/partitioned/{modelDevelopment,Validation}
#   train/val split  read verbatim from the deep models' saved split.csv
#   test set         the 216 spatially clean tiles in test_tiles_clean.csv
#   labels           raw 1-9, waste = 1, label 99 ignored everywhere
#   seeds            1, 2, 3
#   decision rules   argmax, and a threshold chosen on validation only
#   metrics          identical formulas, pooled over pixels
#
# Output uses the SAME SCHEMA as the PyTorch trainer, so load_all_results() in
# 07_figures/00_figure_theme.R picks it up and GeoBIA appears in every figure.
#
#   Rscript 01_geobia/10_pixel_level_evaluation.R --segments otb
#   Rscript 01_geobia/10_pixel_level_evaluation.R --segments oracle --seeds 1
#
# Writes results/02_geobia_pixel/geobia_<segments>_<classifier>/
#          metrics_all_seeds.csv, config.json, per_tile_counts_test_<rule>.csv
################################################################################

suppressPackageStartupMessages({
  library(terra); library(randomForest)
})

root <- local({
  p <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  while (!file.exists(file.path(p, "CITATION.cff")) && dirname(p) != p) p <- dirname(p)
  if (!file.exists(file.path(p, "CITATION.cff"))) stop("Run from inside the repository.")
  p
})
setwd(root)

args   <- commandArgs(trailingOnly = TRUE)
getarg <- function(f, d) { i <- match(f, args)
                           if (is.na(i) || i == length(args)) d else args[i + 1] }
SEGMENTS    <- match.arg(getarg("--segments", "otb"), c("otb", "oracle"))
CLASSIFIER  <- getarg("--classifier", "rf")
SEEDS       <- as.integer(strsplit(getarg("--seeds", "1,2,3"), ",")[[1]])
WASTE       <- as.integer(getarg("--waste-label", "1"))
IGNORE      <- as.integer(getarg("--ignore-label", "99"))
NTREE       <- as.integer(getarg("--ntree", "300"))
MAX_TEST    <- as.integer(getarg("--max-test-tiles", "0"))
LABELS      <- 1:9
P_LANDSCAPE <- 0.00045
THRESH_GRID <- seq(0.05, 0.95, by = 0.05)

DEV <- "data/tiled_data/partitioned/modelDevelopment"
VAL <- "data/tiled_data/partitioned/Validation"
SPLIT_CSV <- "results/03_unet_pytorch/unet_f16_clean/seed_1/split.csv"
CLEAN_CSV <- "results/03_unet_pytorch/test_tiles_clean.csv"

EXP     <- paste0("geobia_", SEGMENTS, "_", CLASSIFIER)
OUT_DIR <- file.path("results", "02_geobia_pixel", EXP)
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

message("GeoBIA pixel-level evaluation")
message("  segments  : ", SEGMENTS,
        if (SEGMENTS == "oracle") "  (mask-derived: upper bound, NOT a method)"
        else "  (OTB meanshift, spatialr 50, ranger 30)")
message("  classifier: ", CLASSIFIER, " | seeds: ", paste(SEEDS, collapse = ","))

# ---- helpers -----------------------------------------------------------------
pixel_metrics <- function(tp, fp, fn, tn) {
  eps <- 1e-12
  rec <- tp / (tp + fn + eps); fpr <- fp / (fp + tn + eps)
  list(tp = tp, fp = fp, fn = fn, tn = tn,
       prevalence  = (tp + fn) / (tp + fp + fn + tn + eps),
       precision   = tp / (tp + fp + eps), recall = rec,
       f1          = 2 * tp / (2 * tp + fp + fn + eps),
       iou         = tp / (tp + fp + fn + eps),
       fpr         = fpr, specificity = tn / (tn + fp + eps),
       implied      = rec * P_LANDSCAPE /
         (rec * P_LANDSCAPE + fpr * (1 - P_LANDSCAPE) + eps),
       required_fpr = rec * P_LANDSCAPE / (1 - P_LANDSCAPE + eps))
}

to_json <- function(x, indent = 2) {
  fmt <- function(v) {
    if (is.null(v)) return("null")
    if (is.character(v)) {
      v <- gsub('"', "'", v)
      if (length(v) == 1) paste0('"', v, '"')
      else paste0("[", paste0('"', v, '"', collapse = ", "), "]")
    } else if (length(v) == 1) as.character(v)
    else paste0("[", paste(v, collapse = ", "), "]")
  }
  paste0("{\n", paste0(strrep(" ", indent), '"', names(x), '": ',
                       vapply(x, fmt, character(1)), collapse = ",\n"), "\n}")
}

norm_id <- function(x) sub("^clipped_image_", "", x)

first_hit <- function(cands) {
  hit <- cands[file.exists(cands)]
  if (length(hit)) hit[1] else NA_character_
}

seg_path <- function(dir, id) {
  b <- norm_id(id)
  if (SEGMENTS == "otb")
    first_hit(file.path(dir, "segments",
                        c(paste0(id, "_seg.shp"), paste0(b, "_seg.shp"),
                          paste0("clipped_image_", b, "_seg.shp"))))
  else
    first_hit(file.path(dir, "vectors_with_var",
                        c(paste0(id, ".shp"), paste0(b, ".shp"),
                          paste0("clipped_image_", b, ".shp"))))
}

mask_path <- function(dir, id) {
  b <- norm_id(id)
  first_hit(file.path(dir, "masks",
                      c(paste0(id, ".tif"), paste0(b, ".tif"),
                        paste0("clipped_image_", b, ".tif"))))
}

# ---- 1. partitions -----------------------------------------------------------
if (!file.exists(SPLIT_CSV))
  stop("Missing ", SPLIT_CSV, "\nRun the PyTorch baseline first so the exact ",
       "train/val split is on disk.")
sp <- utils::read.csv(SPLIT_CSV, stringsAsFactors = FALSE)
sp$id <- tools::file_path_sans_ext(sp$tile)
train_ids <- sp$id[sp$split == "train"]
val_ids   <- sp$id[sp$split == "val"]
test_ids  <- tools::file_path_sans_ext(
  utils::read.csv(CLEAN_CSV, stringsAsFactors = FALSE)$tile)
if (MAX_TEST > 0) test_ids <- head(test_ids, MAX_TEST)
message(sprintf("  %d train, %d val, %d clean test tiles",
                length(train_ids), length(val_ids), length(test_ids)))

# ---- 2. segment feature tables ----------------------------------------------
load_features <- function(dir) {
  if (SEGMENTS == "otb") {
    f <- file.path(dir, "segment_features.csv")
    if (!file.exists(f))
      stop("Missing ", f, "\nRun: Rscript 01_geobia/04b_extract_to_segments.R ",
           "--partition ", basename(dir))
    d <- utils::read.csv(f, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    shps <- list.files(file.path(dir, "vectors_with_var"),
                       pattern = "[.]shp$", full.names = TRUE)
    if (!length(shps)) stop("No oracle polygons under ", dir)
    d <- do.call(rbind, lapply(shps, function(s) {
      v <- tryCatch(suppressWarnings(as.data.frame(vect(s))), error = function(e) NULL)
      if (is.null(v) || !nrow(v)) return(NULL)
      v$tile <- tools::file_path_sans_ext(basename(s))
      v$sid  <- seq_len(nrow(v))
      v
    }))
  }
  d$.key <- norm_id(d$tile)
  d
}

message("  loading segment features ...")
dev_feat  <- load_features(DEV)
test_feat <- load_features(VAL)

drop_cols <- c("ID", "tile_id", "bin", "area_m2", "tile", "sid", "n_pixels", ".key")
feature_names <- setdiff(names(dev_feat), c("class", drop_cols))
feature_names <- feature_names[vapply(dev_feat[feature_names], is.numeric, logical(1))]
feature_names <- intersect(feature_names, names(test_feat))

prep <- function(d) {
  d <- d[!is.na(d$class) & d$class != IGNORE & d$class %in% LABELS, , drop = FALSE]
  d <- d[stats::complete.cases(d[, feature_names, drop = FALSE]), , drop = FALSE]
  d$class <- factor(as.integer(d$class), levels = LABELS)
  droplevels(d)
}
train_tbl <- prep(dev_feat[dev_feat$.key %in% norm_id(train_ids), , drop = FALSE])
n_tiles <- length(unique(train_tbl$.key))
message(sprintf("  %d training segments over %d tiles (%.1f per tile), %d features",
                nrow(train_tbl), n_tiles, nrow(train_tbl) / max(1, n_tiles),
                length(feature_names)))
if (!as.character(WASTE) %in% levels(train_tbl$class))
  stop("Waste class ", WASTE, " absent from the training segments.")

# ---- 3. per-tile probability and argmax rasters ------------------------------
rasterise_tile <- function(model, dir, feat, id) {
  shp <- seg_path(dir, id); msk <- mask_path(dir, id)
  if (is.na(shp) || is.na(msk)) return(NULL)
  rows <- feat[feat$.key == norm_id(id), , drop = FALSE]
  if (!nrow(rows)) return(NULL)
  ok <- stats::complete.cases(rows[, feature_names, drop = FALSE])
  if (!any(ok)) return(NULL)

  newd <- rows[ok, feature_names, drop = FALSE]
  pr   <- predict(model, newdata = newd, type = "prob")
  cls  <- as.character(predict(model, newdata = newd))
  wc   <- as.character(WASTE)

  v <- tryCatch(suppressWarnings(vect(shp)), error = function(e) NULL)
  if (is.null(v) || nrow(v) == 0) return(NULL)

  pw <- rep(0, nrow(v)); am <- rep(0L, nrow(v))
  idx <- rows$sid[ok]
  good <- idx >= 1 & idx <= nrow(v)
  if (any(good)) {
    pw[idx[good]] <- if (wc %in% colnames(pr)) pr[good, wc] else 0
    am[idx[good]] <- as.integer(cls[good] == wc)
  }
  v$.pw <- pw; v$.am <- am

  r_mask <- rast(msk)
  if (!same.crs(v, r_mask)) v <- project(v, crs(r_mask))
  pr_r <- tryCatch(rasterize(v, r_mask, field = ".pw", background = 0, touches = TRUE),
                   error = function(e) NULL)
  am_r <- tryCatch(rasterize(v, r_mask, field = ".am", background = 0, touches = TRUE),
                   error = function(e) NULL)
  if (is.null(pr_r) || is.null(am_r)) return(NULL)

  ref <- terra::values(r_mask)[, 1]
  p   <- terra::values(pr_r)[, 1]; p[is.na(p)] <- 0
  a   <- terra::values(am_r)[, 1]; a[is.na(a)] <- 0
  keep <- !is.na(ref) & ref != IGNORE
  list(ref = ref[keep], p = p[keep], am = a[keep] == 1)
}

score_set <- function(model, dir, feat, ids) {
  out <- lapply(ids, function(i) rasterise_tile(model, dir, feat, i))
  names(out) <- ids
  Filter(Negate(is.null), out)
}

count_at <- function(cache, thr) {
  tp <- fp <- fn <- tn <- 0
  for (z in cache) {
    w <- z$ref == WASTE; pw <- z$p >= thr
    tp <- tp + sum(w & pw); fp <- fp + sum(!w & pw)
    fn <- fn + sum(w & !pw); tn <- tn + sum(!w & !pw)
  }
  c(tp = tp, fp = fp, fn = fn, tn = tn)
}
count_argmax <- function(cache) {
  tp <- fp <- fn <- tn <- 0
  for (z in cache) {
    w <- z$ref == WASTE; pw <- z$am
    tp <- tp + sum(w & pw); fp <- fp + sum(!w & pw)
    fn <- fn + sum(w & !pw); tn <- tn + sum(!w & !pw)
  }
  c(tp = tp, fp = fp, fn = fn, tn = tn)
}
per_tile_counts <- function(cache, thr, use_argmax = FALSE) {
  do.call(rbind, lapply(names(cache), function(id) {
    z <- cache[[id]]; w <- z$ref == WASTE
    pw <- if (use_argmax) z$am else z$p >= thr
    data.frame(tile = id, tp = sum(w & pw), fp = sum(!w & pw),
               fn = sum(w & !pw), tn = sum(!w & !pw))
  }))
}
average_precision <- function(cache) {
  p <- unlist(lapply(cache, function(z) z$p), use.names = FALSE)
  y <- unlist(lapply(cache, function(z) z$ref == WASTE), use.names = FALSE)
  if (!any(y)) return(NA_real_)
  o <- order(p, decreasing = TRUE); y <- y[o]
  tp <- cumsum(y); prec <- tp / seq_along(y); rec <- tp / sum(y)
  sum(diff(c(0, rec)) * prec)
}

# ---- 4. seeds ----------------------------------------------------------------
rows <- list()
for (sd in SEEDS) {
  message("\n  --- seed ", sd, " ---")
  set.seed(sd)
  t0 <- Sys.time()
  model <- randomForest::randomForest(
    x = train_tbl[, feature_names, drop = FALSE], y = train_tbl$class, ntree = NTREE)
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  message(sprintf("    trained in %.2f min", mins))

  message("    scoring validation tiles ...")
  cache_val  <- score_set(model, DEV, dev_feat, val_ids)
  message("    scoring clean test tiles ...")
  cache_test <- score_set(model, VAL, test_feat, test_ids)
  message(sprintf("    scored %d val, %d test tiles",
                  length(cache_val), length(cache_test)))
  if (!length(cache_test)) stop("No test tiles could be scored.")

  f1s <- vapply(THRESH_GRID, function(t) {
    cc <- count_at(cache_val, t)
    unname(pixel_metrics(cc[["tp"]], cc[["fp"]], cc[["fn"]], cc[["tn"]])$f1)
  }, numeric(1))
  thr <- THRESH_GRID[which.max(f1s)]
  message(sprintf("    threshold from validation: %.2f (val F1 %.4f)", thr, max(f1s)))

  ap <- c(val = average_precision(cache_val), test = average_precision(cache_test))

  for (part in c("val", "test")) {
    cache <- if (part == "val") cache_val else cache_test
    for (rule in c("argmax", "thresh")) {
      cnt <- if (rule == "argmax") count_argmax(cache) else count_at(cache, thr)
      m <- pixel_metrics(cnt[["tp"]], cnt[["fp"]], cnt[["fn"]], cnt[["tn"]])
      r <- data.frame(
        seed = sd, arch = paste0("geobia-", SEGMENTS, "-", CLASSIFIER),
        split = part, rule = rule,
        threshold = if (rule == "thresh") thr else NA_real_,
        average_precision = unname(ap[[part]]),
        epochs_run = NA_integer_, minutes = mins,
        tp = m$tp, fp = m$fp, fn = m$fn, tn = m$tn,
        prevalence = m$prevalence, precision = m$precision, recall = m$recall,
        f1 = m$f1, iou = m$iou, fpr = m$fpr, specificity = m$specificity,
        check.names = FALSE)
      r[["implied_precision_at_p=0.00045"]]      <- m$implied
      r[["required_fpr_for_P=0.5_at_p=0.00045"]] <- m$required_fpr
      rows[[length(rows) + 1]] <- r
      message(sprintf("    %-4s %-6s  F1 %.4f  P %.4f  R %.4f  FPR %.5f",
                      part, rule, m$f1, m$precision, m$recall, m$fpr))
      if (part == "test")
        utils::write.csv(per_tile_counts(cache, thr, rule == "argmax"),
                         file.path(OUT_DIR,
                                   sprintf("per_tile_counts_test_%s.csv", rule)),
                         row.names = FALSE)
    }
  }
}

out <- do.call(rbind, rows)
utils::write.csv(out, file.path(OUT_DIR, "metrics_all_seeds.csv"), row.names = FALSE)
writeLines(to_json(list(
  arch = paste0("geobia-", SEGMENTS, "-", CLASSIFIER),
  segments = SEGMENTS, classifier = CLASSIFIER, seeds = SEEDS,
  waste_label = WASTE, ignore_label = IGNORE,
  n_features = length(feature_names), ntree = NTREE,
  split_source = SPLIT_CSV, clean_test = CLEAN_CSV,
  basis = "pixel level, pooled over tiles, label 99 excluded",
  segmentation = if (SEGMENTS == "otb")
    "OTB meanshift, spatialr 50, ranger 30"
  else "polygons vectorised from the reference mask: ORACLE upper bound, not a method",
  note = "Comparable with results/0{3,4,5,6}_*: same partition, split and metrics"
)), file.path(OUT_DIR, "config.json"))

message("\nwrote ", file.path(OUT_DIR, "metrics_all_seeds.csv"))
if (SEGMENTS == "oracle")
  message("NOTE: oracle segments are an upper bound, not an automated method.")
