suppressPackageStartupMessages(library(terra))

project_root <- Sys.getenv("MWUA_PROJECT_ROOT", unset = getwd())
project_root <- normalizePath(project_root, winslash = "/", mustWork = TRUE)

data_root <- Sys.getenv(
  "MWUA_DATA_ROOT",
  unset = file.path(project_root, "data")
)

output_root <- Sys.getenv(
  "MWUA_OUTPUT_ROOT",
  unset = file.path(project_root, "outputs")
)

partition_root <- Sys.getenv(
  "MWUA_PARTITION_ROOT",
  unset = file.path(data_root, "tiled_data", "partitioned")
)

manifest_path <- file.path(data_root, "splits", "study_tile_manifest.csv")
audit_dir <- file.path(output_root, "metrics", "00_data_preparation")
dir.create(audit_dir, recursive = TRUE, showWarnings = FALSE)

expected_development <- 804L
expected_holdout <- 222L
expected_total <- 1026L
expected_waste_positive_validation <- 84L
waste_class <- 1

if (!file.exists(manifest_path)) {
  stop("Missing Stage 00 manifest: ", manifest_path)
}

manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
required_columns <- c("file_name", "set", "id", "quadrant")
missing_columns <- setdiff(required_columns, names(manifest))

if (length(missing_columns) > 0) {
  stop("Manifest is missing column(s): ", paste(missing_columns, collapse = ", "))
}

list_tif_names <- function(directory) {
  if (!dir.exists(directory)) return(character())
  basename(list.files(directory, pattern = "\\.[tT][iI][fF]$", full.names = TRUE))
}

partition_paths <- list(
  development = file.path(partition_root, "modelDevelopment"),
  holdout = file.path(partition_root, "Validation")
)

expected_names <- split(manifest$file_name, manifest$set)

file_audit <- lapply(names(partition_paths), function(set_name) {
  set_root <- partition_paths[[set_name]]
  expected_images <- sort(expected_names[[set_name]])
  expected_multiband <- sub("\\.tif$", "_combined.tif", expected_images, ignore.case = TRUE)

  actual_images <- sort(list_tif_names(file.path(set_root, "images")))
  actual_masks <- sort(list_tif_names(file.path(set_root, "masks")))
  actual_multiband <- sort(list_tif_names(file.path(set_root, "multiband")))

  list(
    set = set_name,
    missing_images = setdiff(expected_images, actual_images),
    extra_images = setdiff(actual_images, expected_images),
    missing_masks = setdiff(expected_images, actual_masks),
    extra_masks = setdiff(actual_masks, expected_images),
    missing_multiband = setdiff(expected_multiband, actual_multiband),
    extra_multiband = setdiff(actual_multiband, expected_multiband)
  )
})

count_issue <- function(field) {
  sum(vapply(file_audit, function(item) length(item[[field]]), integer(1)))
}

set_counts <- table(manifest$set)
development_count <- unname(set_counts["development"])
holdout_count <- unname(set_counts["holdout"])
if (is.na(development_count)) development_count <- 0L
if (is.na(holdout_count)) holdout_count <- 0L

duplicate_count <- sum(duplicated(manifest$file_name))

sets_per_cell <- aggregate(set ~ id, manifest, function(values) length(unique(values)))
shared_cell_count <- sum(sets_per_cell$set > 1)

validation_mask_dir <- file.path(partition_paths$holdout, "masks")
validation_names <- expected_names$holdout

waste_positive <- vapply(validation_names, function(file_name) {
  mask_path <- file.path(validation_mask_dir, file_name)
  if (!file.exists(mask_path)) return(FALSE)
  mask <- rast(mask_path)
  waste_pixels <- global(mask == waste_class, "sum", na.rm = TRUE)[1, 1]
  is.finite(waste_pixels) && waste_pixels > 0
}, logical(1))

waste_positive_count <- sum(waste_positive)

audit_rows <- data.frame(
  check = c(
    "total_tiles",
    "development_tiles",
    "holdout_tiles",
    "duplicate_filenames",
    "grid_cells_shared_between_sets",
    "missing_images",
    "extra_images",
    "missing_masks",
    "extra_masks",
    "missing_multiband",
    "extra_multiband",
    "waste_positive_validation_tiles"
  ),
  observed = c(
    nrow(manifest),
    development_count,
    holdout_count,
    duplicate_count,
    shared_cell_count,
    count_issue("missing_images"),
    count_issue("extra_images"),
    count_issue("missing_masks"),
    count_issue("extra_masks"),
    count_issue("missing_multiband"),
    count_issue("extra_multiband"),
    waste_positive_count
  ),
  expected = c(
    expected_total,
    expected_development,
    expected_holdout,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    expected_waste_positive_validation
  ),
  stringsAsFactors = FALSE
)

audit_rows$status <- ifelse(
  audit_rows$observed == audit_rows$expected,
  "PASS",
  "FAIL"
)

audit_path <- file.path(audit_dir, "stage00_dataset_audit.csv")
write.csv(audit_rows, audit_path, row.names = FALSE)
print(audit_rows, row.names = FALSE)

if (any(audit_rows$status == "FAIL")) {
  stop("Stage 00 dataset audit failed. See: ", audit_path)
}

message("Stage 00 dataset audit passed: ", audit_path)

