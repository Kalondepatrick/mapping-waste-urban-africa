################################################################################
# OTB meanshift segmentation for a named partition
################################################################################
# 03_creation_of_segments.R only ever ran on modelDevelopment, so the Validation
# partition had no automated segments and the OTB branch could not be evaluated
# on held-out data. This script does either partition.
#
#   Rscript 01_geobia/03b_segment_partition.R --partition Validation
#   Rscript 01_geobia/03b_segment_partition.R --partition modelDevelopment
#
# Parameters match 03_creation_of_segments.R exactly (spatialr 50, ranger 30) so
# the two partitions are segmented identically.
################################################################################

suppressPackageStartupMessages({
  library(parallel)
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

PARTITION <- getarg("--partition", "Validation")
SPATIAL_R <- as.integer(getarg("--spatialr", "50"))
RANGE_R   <- as.integer(getarg("--ranger", "30"))
OTB_PATH  <- getarg("--otb",
  "C:/Users/Peter/Documents/Tools/OTB-9.1.0-Win64/bin/otbcli_Segmentation.bat")

image_folder <- file.path("data/tiled_data/partitioned", PARTITION, "images")
output_base  <- file.path("data/tiled_data/partitioned", PARTITION, "segments")

if (!file.exists(OTB_PATH))
  stop("OTB not found at ", OTB_PATH, "\nPass --otb <path to otbcli_Segmentation.bat>")
if (!dir.exists(image_folder)) stop("No images at ", image_folder)
dir.create(output_base, recursive = TRUE, showWarnings = FALSE)

tif_files <- list.files(image_folder, pattern = "[.]tif$", full.names = TRUE)
todo <- tif_files[!file.exists(file.path(
  output_base, paste0(tools::file_path_sans_ext(basename(tif_files)), "_seg.shp")))]

message("partition : ", PARTITION)
message("images    : ", length(tif_files), " (", length(todo), " still to segment)")
message("meanshift : spatialr ", SPATIAL_R, ", ranger ", RANGE_R)
if (!length(todo)) { message("nothing to do"); quit(save = "no") }

process_image <- function(image_path, otb_exe, out_dir, spat_r, ran_r) {
  nm  <- tools::file_path_sans_ext(basename(image_path))
  out <- file.path(out_dir, paste0(nm, "_seg.shp"))
  if (file.exists(out)) return(paste("skip", nm))
  cmd <- paste(shQuote(otb_exe),
               "-in", shQuote(image_path),
               "-filter meanshift",
               "-filter.meanshift.spatialr", spat_r,
               "-filter.meanshift.ranger", ran_r,
               "-mode vector",
               "-mode.vector.out", shQuote(out))
  system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE)
  if (file.exists(out)) paste("ok", nm) else paste("FAILED", nm)
}

used <- max(1, parallel::detectCores() - 5)
message("cores     : ", used)
t0 <- Sys.time()

cl <- makeCluster(used)
on.exit(stopCluster(cl), add = TRUE)
clusterExport(cl, c("process_image", "OTB_PATH", "output_base",
                    "SPATIAL_R", "RANGE_R"), envir = environment())
res <- parLapply(cl, todo, function(f)
  process_image(f, OTB_PATH, output_base, SPATIAL_R, RANGE_R))

ok <- sum(grepl("^ok", unlist(res)))
message(sprintf("done: %d/%d segmented in %.1f min", ok, length(todo),
                as.numeric(difftime(Sys.time(), t0, units = "mins"))))
failed <- unlist(res)[grepl("^FAILED", unlist(res))]
if (length(failed)) {
  message("failures (", length(failed), "):")
  writeLines(paste("  ", head(failed, 20)))
}
