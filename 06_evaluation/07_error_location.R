# =============================================================================
# Spatial distribution of false positives
# =============================================================================
#   Rscript 06_evaluation/07_error_location.R
#
# A false positive in a tile that also contains waste is a delineation error; a
# false positive in a tile with no waste is a detection in unfamiliar
# background. The two call for different remedies, so the split matters more
# than the total. Computed from the per-tile confusion counts the trainer
# writes, so no inference is re-run.
#
# Writes results/tables/T7_error_location.{csv,md} and appends to manifest.csv
# =============================================================================

local({
  p <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  while (!file.exists(file.path(p, "CITATION.cff")) && dirname(p) != p) p <- dirname(p)
  if (!file.exists(file.path(p, "CITATION.cff")))
    stop("Run this from inside the repository.")
  setwd(p)
})
suppressMessages(source("07_figures/00_figure_theme.R"))

TAB_DIR <- file.path(RESULTS_DIR, "tables")

files <- list.files(RESULTS_DIR, pattern = "^per_tile_counts_test_argmax\\.csv$",
                    recursive = TRUE, full.names = TRUE)
files <- files[!grepl("geobia_oracle_rf", files)]
if (!length(files)) stop("No per-tile count files found.")

# Restrict to the same 216 spatially clean tiles that every other table uses.
# The per-tile files cover all 222 validation tiles, six of which overlap a
# development tile; including them here would measure a different population.
CLEAN <- utils::read.csv(file.path(RESULTS_DIR, "03_unet_pytorch",
                                   "test_tiles_clean.csv"))$tile
CLEAN_STEM <- sub("[.]tif$", "", CLEAN)

# config name: the directory above seed_N, or the directory itself
cfg_of <- function(f) {
  parts <- strsplit(gsub("\\\\", "/", dirname(f)), "/")[[1]]
  if (grepl("^seed_[0-9]+$", parts[length(parts)])) parts[length(parts) - 1]
  else parts[length(parts)]
}

rows <- do.call(rbind, lapply(split(files, vapply(files, cfg_of, character(1))),
  function(fs) {
    per_seed <- lapply(fs, function(f) {
      x <- utils::read.csv(f)
      x <- x[sub("[.]tif$", "", x$tile) %in% CLEAN_STEM, , drop = FALSE]
      # Actual waste pixels in the tile = true positives + false negatives.
      # Derived rather than read from n_waste, which the GeoBIA evaluator does
      # not write, so that every configuration is measured the same way.
      has_waste <- (x$tp + x$fn) > 0
      c(fp_total   = sum(x$fp),
        fp_in      = sum(x$fp[has_waste]),
        tiles      = nrow(x),
        tiles_with = sum(has_waste))
    })
    m <- do.call(rbind, per_seed)
    share <- m[, "fp_in"] / m[, "fp_total"]
    data.frame(
      config     = cfg_of(fs[1]),
      seeds      = length(fs),
      tiles      = m[1, "tiles"],
      tiles_with_waste = m[1, "tiles_with"],
      fp_total   = mean(m[, "fp_total"]),
      share_in_waste_tiles = mean(share),
      share_sd   = if (length(share) > 1) stats::sd(share) else 0,
      stringsAsFactors = FALSE)
  }))
rownames(rows) <- NULL
rows <- rows[order(-rows$share_in_waste_tiles), ]

out <- data.frame(
  Configuration = config_label(rows$config),
  Seeds = rows$seeds,
  `False-positive pixels` = format(round(rows$fp_total), big.mark = ","),
  `In tiles containing waste` = sprintf("%.1f%%", 100 * rows$share_in_waste_tiles),
  `In tiles with no waste` = sprintf("%.1f%%", 100 * (1 - rows$share_in_waste_tiles)),
  check.names = FALSE)

utils::write.csv(out, file.path(TAB_DIR, "T7_error_location.csv"), row.names = FALSE)

title <- "Spatial Distribution of False Positives on the Held-Out Partition"
note <- sprintf(paste(
  "Of the %d spatially clean held-out tiles, %d contain at least one waste",
  "pixel. A false",
  "positive inside such a tile is a delineation error; one in a tile with no",
  "waste is a false detection in unfamiliar background. Counts are means over",
  "seeds. Reported for every configuration that stored per-tile confusion",
  "counts; the backbone comparison of Table IV did not store them, so the",
  "selected model is not listed and the pattern should be read as a property",
  "of this imagery rather than of one architecture."),
  rows$tiles[1], rows$tiles_with_waste[1])

align <- c(":---", "---:", "---:", "---:", "---:")
md <- c("**TABLE VII**", paste0("**", title, "**"), "",
        paste0("| ", paste(names(out), collapse = " | "), " |"),
        paste0("| ", paste(align, collapse = " | "), " |"),
        apply(out, 1, function(r) paste0("| ", paste(trimws(r), collapse = " | "), " |")),
        "", paste0("*", note, "*"))
writeLines(md, file.path(TAB_DIR, "T7_error_location.md"))

mf <- file.path(TAB_DIR, "manifest.csv")
if (file.exists(mf)) {
  m <- utils::read.csv(mf, stringsAsFactors = FALSE)
  m <- m[m$file != "T7_error_location.csv", ]
  m[nrow(m) + 1, ] <- list("T7_error_location.csv", "VII", title, note, "body")
  utils::write.csv(m, mf, row.names = FALSE)
}

print(out, row.names = FALSE)
cat(sprintf("\nrange across configurations: %.1f%% to %.1f%% of false positives\n",
            100 * min(rows$share_in_waste_tiles),
            100 * max(rows$share_in_waste_tiles)))
cat(sprintf("unweighted mean: %.1f%%\n", 100 * mean(rows$share_in_waste_tiles)))
