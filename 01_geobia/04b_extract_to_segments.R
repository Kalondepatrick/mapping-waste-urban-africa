################################################################################
# Extract features and the reference class to OTB segments
################################################################################
# Replaces 04_extraction_to_segments.R, which cannot run as saved: it calls
# future_lapply over `files_df` and `texture_names`, neither of which it ever
# defines, and it only ever pointed at modelDevelopment.
#
# For every tile in a partition this reads the existing OTB segments and
# attaches, per segment:
#   class            modal value of the reference mask under the segment
#   Red/Green/Blue   mean of the RGB orthomosaic tile
#   28 texture bands mean of the stacked colour-index + Haralick tile
#
# The 31 feature names match those already in vectors_with_var, so the feature
# space is identical to the earlier oracle-segment experiments and the ONLY
# thing that changes is the segmentation.
#
# Output is one CSV per partition rather than 1,026 shapefiles: the segment
# geometry is already on disk and is re-read at evaluation time, so duplicating
# it would cost time and disk for nothing.
#
#   Rscript 01_geobia/04b_extract_to_segments.R --partition modelDevelopment
#   Rscript 01_geobia/04b_extract_to_segments.R --partition Validation
#
# Writes data/tiled_data/partitioned/<partition>/segment_features.csv
################################################################################

suppressPackageStartupMessages({
  library(terra)
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
PARTITION <- getarg("--partition", "modelDevelopment")
LIMIT     <- as.integer(getarg("--limit", "0"))

base      <- file.path("data/tiled_data/partitioned", PARTITION)
seg_dir   <- file.path(base, "segments")
mask_dir  <- file.path(base, "masks")
rgb_dir   <- file.path(base, "images")
multi_dir <- file.path(base, "multiband")
out_csv   <- file.path(base, "segment_features.csv")

texture_names <- c("Energy", "Entropy", "Cor", "IDM", "Inertia", "CS", "CP",
                   "HC", "Mean", "Varianc", "Dssmlrt", "SumAvrg", "SumVrnc",
                   "SmEntrp", "Df_f_En", "Df_f_Vr", "IC1", "IC2", "SRE", "LRE",
                   "GLN", "RLN", "RP", "LGRE", "HGLRE", "SRLGLE", "SRHGLE",
                   "LRLGLE")

seg_files <- list.files(seg_dir, pattern = "_seg[.]shp$", full.names = TRUE)
if (!length(seg_files)) stop("No OTB segments in ", seg_dir,
                             "\nRun 01_geobia/03b_segment_partition.R first.")
if (LIMIT > 0) seg_files <- head(seg_files, LIMIT)

tile_of <- function(p) sub("_seg$", "", tools::file_path_sans_ext(basename(p)))

first_existing <- function(dir, id) {
  cand <- c(file.path(dir, paste0(id, ".tif")),
            file.path(dir, paste0(sub("^clipped_image_", "", id), ".tif")),
            file.path(dir, paste0("clipped_image_", id, ".tif")),
            file.path(dir, paste0(id, "_combined.tif")),
            file.path(dir, paste0(sub("^clipped_image_", "", id), "_combined.tif")))
  hit <- cand[file.exists(cand)]
  if (length(hit)) hit[1] else NA_character_
}

# Resume: keep tiles already present in the CSV.
done_tiles <- character(0)
if (file.exists(out_csv)) {
  prev <- utils::read.csv(out_csv, stringsAsFactors = FALSE)
  done_tiles <- unique(prev$tile)
  message("resuming: ", length(done_tiles), " tiles already extracted")
}

message("partition: ", PARTITION)
message("segments : ", length(seg_files), " tiles")

acc <- list(); ok <- 0; skipped <- 0; failed <- character(0)
t0 <- Sys.time()

for (k in seq_along(seg_files)) {
  id <- tile_of(seg_files[k])
  if (id %in% done_tiles) { skipped <- skipped + 1; next }

  res <- tryCatch({
    v <- suppressWarnings(vect(seg_files[k]))
    if (nrow(v) == 0) stop("no segments")

    rgb_p  <- first_existing(rgb_dir, id)
    mask_p <- first_existing(mask_dir, id)
    mul_p  <- first_existing(multi_dir, id)
    if (is.na(rgb_p) || is.na(mask_p)) stop("missing image or mask")

    r_rgb <- rast(rgb_p)[[1:3]]
    names(r_rgb) <- c("Red", "Green", "Blue")
    if (!same.crs(v, r_rgb)) v <- project(v, crs(r_rgb))

    r_mask <- rast(mask_p)
    stk <- c(r_rgb, r_mask)
    nm  <- c("Red", "Green", "Blue", ".mask")

    if (!is.na(mul_p)) {
      r_tex <- rast(mul_p)
      # The multiband tiles carry a compound CRS (UTM + vertical datum) while
      # the RGB tiles carry plain UTM. The grids are identical, so align the
      # CRS label rather than resampling.
      if (!same.crs(r_tex, r_rgb) &&
          isTRUE(all.equal(as.vector(ext(r_tex)), as.vector(ext(r_rgb)))) &&
          all(dim(r_tex)[1:2] == dim(r_rgb)[1:2])) {
        crs(r_tex) <- crs(r_rgb)
      }
      nb <- min(nlyr(r_tex), length(texture_names))
      r_tex <- r_tex[[seq_len(nb)]]
      names(r_tex) <- texture_names[seq_len(nb)]
      stk <- c(stk, r_tex)
      nm  <- c(nm, texture_names[seq_len(nb)])
    }

    # Rasterise the segment index once, then aggregate every band by it.
    v$.sid <- seq_len(nrow(v))
    sid <- terra::values(terra::rasterize(v, r_rgb, field = ".sid"))[, 1]

    M <- terra::values(stk); colnames(M) <- nm
    keep <- !is.na(sid)
    if (!any(keep)) stop("no pixels under segments")
    sidk <- sid[keep]; Mk <- M[keep, , drop = FALSE]

    n_per <- as.vector(rowsum(rep(1, length(sidk)), sidk))
    means <- rowsum(Mk, sidk, na.rm = TRUE) / n_per
    ids   <- as.integer(rownames(means))

    modal_class <- vapply(split(Mk[, ".mask"], sidk), function(z) {
      z <- z[!is.na(z)]
      if (!length(z)) return(NA_real_)
      tb <- table(z); as.numeric(names(tb)[which.max(tb)])
    }, numeric(1))

    feat <- setdiff(nm, ".mask")
    acc[[length(acc) + 1]] <- data.frame(
      tile = id, sid = ids, n_pixels = n_per,
      class = as.numeric(modal_class),
      means[, feat, drop = FALSE],
      row.names = NULL, check.names = FALSE)
    "ok"
  }, error = function(e) paste("FAIL", id, ":", conditionMessage(e)))

  if (identical(res, "ok")) ok <- ok + 1 else failed <- c(failed, res)

  if (k %% 50 == 0) {
    el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    message(sprintf("  %d/%d  ok %d  failed %d  %.1f min  (%.1f tiles/min)",
                    k, length(seg_files), ok, length(failed), el,
                    if (el > 0) ok / el else 0))
  }
}

new_rows <- if (length(acc)) do.call(rbind, acc) else NULL
if (!is.null(new_rows)) {
  if (file.exists(out_csv)) {
    prev <- utils::read.csv(out_csv, stringsAsFactors = FALSE, check.names = FALSE)
    new_rows <- rbind(prev, new_rows)
  }
  utils::write.csv(new_rows, out_csv, row.names = FALSE)
}

message(sprintf("\ndone: %d tiles extracted, %d already present, %d failed, %.1f min",
                ok, skipped, length(failed),
                as.numeric(difftime(Sys.time(), t0, units = "mins"))))
if (length(failed)) writeLines(paste("  ", head(failed, 20)))
if (!is.null(new_rows))
  message("wrote ", out_csv, "  (", nrow(new_rows), " segments)")
