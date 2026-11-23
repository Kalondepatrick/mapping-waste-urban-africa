# ============================================================
# WASTE PREDICTION REVIEW APP
# SQLite-backed, image-only Shiny reviewer
#
# FEATURES
# --------
# - Uses pre-generated PNG review chips only
# - No terra/raster processing during normal review
# - SQLite-backed review log with WAL mode
# - Multi-user chip claiming
# - Browser preloading of likely next images
# - Review classes:
#       waste
#       not_waste
#       mixed
# - Visual correction mode:
#       * Edit my last review
#       * Browse my reviewed images
#       * Previous / next reviewed image
#       * No need to know chip IDs
# - Full audit history retained in annotation_history
# - Spatial export to GeoPackage + CSV
#
# KEYBOARD
# --------
# NORMAL REVIEW MODE
#   1 = Waste
#   2 = Not waste
#   3 = Waste + false positives
#   S = Skip
#
# CORRECTION MODE
#   1 = Change/current classification to Waste
#   2 = Change/current classification to Not waste
#   3 = Change/current classification to Waste + false positives
#   S = Move to next reviewed image without changing it
#
# EXPECTED PROJECT STRUCTURE
# --------------------------
#
# mapping-waste-urban-africa/
# └── 05_finetuning/
#     └── waste_review_app/
#         ├── app.R
#         ├── review_log.sqlite          # created automatically
#         ├── data/
#         │   ├── chips.csv
#         │   ├── object_chip_map.csv
#         │   └── review_grid.gpkg
#         ├── exports/
#         │   └── ...                    # created automatically
#         └── www/
#             └── chips/
#                 ├── C00001.png
#                 ├── C00002.png
#                 └── ...
#
# PROJECT OUTPUT ALSO REQUIRED
# ----------------------------
# mapping-waste-urban-africa/
# └── outputs/
#     └── waste_piles.shp
#
# chips.csv requires:
#   chip_id, image_file
#
# Recommended:
#   chip_id, image_file, n_objects, centre_x, centre_y
#
# object_chip_map.csv requires:
#   candidate_id, chip_id
# ============================================================


# ============================================================
# 0. PACKAGES
# ============================================================

required_packages <- c(
  "shiny",
  "DBI",
  "RSQLite",
  "sf"
)

missing_packages <- required_packages[
  !required_packages %in% rownames(installed.packages())
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(shiny)
library(DBI)
library(RSQLite)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_PATH <- file.path(
  Sys.getenv("USERPROFILE"),
  "Documents",
  "GitHub",
  "mapping-waste-urban-africa"
)

APP_DIR <- file.path(
  BASE_PATH,
  "05_finetuning",
  "waste_review_app"
)

DATA_DIR <- file.path(
  APP_DIR,
  "data"
)

CHIPS_DIR <- file.path(
  APP_DIR,
  "www",
  "chips"
)

CHIP_METADATA_PATH <- file.path(
  DATA_DIR,
  "chips.csv"
)

OBJECT_CHIP_MAP_PATH <- file.path(
  DATA_DIR,
  "object_chip_map.csv"
)

REVIEW_GRID_PATH <- file.path(
  DATA_DIR,
  "review_grid.gpkg"
)

PREDICTION_PATH <- file.path(
  BASE_PATH,
  "outputs",
  "waste_piles.shp"
)

DB_PATH <- file.path(
  APP_DIR,
  "review_log.sqlite"
)

EXPORT_DIR <- file.path(
  APP_DIR,
  "exports"
)

EXPORT_GPKG <- file.path(
  EXPORT_DIR,
  "waste_review_export.gpkg"
)

EXPORT_CSV <- file.path(
  EXPORT_DIR,
  "chip_review_results.csv"
)

EXPORT_SUMMARY_CSV <- file.path(
  EXPORT_DIR,
  "review_export_summary.csv"
)

dir.create(
  EXPORT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# 2. SETTINGS
# ============================================================

CLAIM_TIMEOUT_MINUTES <- 30

CLAIM_HEARTBEAT_SECONDS <- 120

N_PRELOAD <- 3

STATS_REFRESH_MS <- 5000


# ============================================================
# 3. CHECK INPUTS
# ============================================================

if (!dir.exists(APP_DIR)) {
  stop(
    "Could not find app folder:\n",
    APP_DIR
  )
}

if (!dir.exists(DATA_DIR)) {
  stop(
    "Could not find data folder:\n",
    DATA_DIR
  )
}

if (!dir.exists(CHIPS_DIR)) {
  stop(
    "Could not find chip image folder:\n",
    CHIPS_DIR
  )
}

if (!file.exists(CHIP_METADATA_PATH)) {
  stop(
    "Could not find chips.csv:\n",
    CHIP_METADATA_PATH
  )
}

if (!file.exists(OBJECT_CHIP_MAP_PATH)) {
  stop(
    "Could not find object_chip_map.csv:\n",
    OBJECT_CHIP_MAP_PATH
  )
}

if (!file.exists(REVIEW_GRID_PATH)) {
  stop(
    "Could not find review_grid.gpkg:\n",
    REVIEW_GRID_PATH
  )
}

if (!file.exists(PREDICTION_PATH)) {
  stop(
    "Could not find predicted waste polygons:\n",
    PREDICTION_PATH
  )
}


# ============================================================
# 4. SERVE CHIP IMAGES TO THE BROWSER
# ============================================================

CHIPS_DIR <- normalizePath(
  CHIPS_DIR,
  winslash = "/",
  mustWork = TRUE
)

addResourcePath(
  prefix = "review_chips",
  directoryPath = CHIPS_DIR
)


# ============================================================
# 5. READ PREPARED METADATA
# ============================================================

chips <- read.csv(
  CHIP_METADATA_PATH,
  stringsAsFactors = FALSE
)

object_chip_map <- read.csv(
  OBJECT_CHIP_MAP_PATH,
  stringsAsFactors = FALSE
)


required_chip_columns <- c(
  "chip_id",
  "image_file"
)

missing_chip_columns <- setdiff(
  required_chip_columns,
  names(chips)
)

if (length(missing_chip_columns) > 0) {
  stop(
    "chips.csv is missing column(s): ",
    paste(
      missing_chip_columns,
      collapse = ", "
    )
  )
}


required_map_columns <- c(
  "candidate_id",
  "chip_id"
)

missing_map_columns <- setdiff(
  required_map_columns,
  names(object_chip_map)
)

if (length(missing_map_columns) > 0) {
  stop(
    "object_chip_map.csv is missing column(s): ",
    paste(
      missing_map_columns,
      collapse = ", "
    )
  )
}


# ------------------------------------------------------------
# Confirm PNGs exist
# ------------------------------------------------------------

chip_paths <- file.path(
  CHIPS_DIR,
  chips$image_file
)

missing_images <- chips$image_file[
  !file.exists(chip_paths)
]

if (length(missing_images) > 0) {
  stop(
    "Some chip images listed in chips.csv do not exist.\n",
    "Examples:\n",
    paste(
      head(
        missing_images,
        10
      ),
      collapse = "\n"
    )
  )
}


# ------------------------------------------------------------
# Add per-chip object counts if needed
# ------------------------------------------------------------

if (!"n_objects" %in% names(chips)) {

  object_counts <- table(
    object_chip_map$chip_id
  )

  chips$n_objects <- as.integer(
    object_counts[
      match(
        chips$chip_id,
        names(object_counts)
      )
    ]
  )

  chips$n_objects[
    is.na(chips$n_objects)
  ] <- 0L
}


TOTAL_CHIPS <- nrow(
  chips
)

TOTAL_OBJECTS <- length(
  unique(
    object_chip_map$candidate_id
  )
)

chip_order <- setNames(
  seq_len(TOTAL_CHIPS),
  chips$chip_id
)


cat(
  "\n============================================\n",
  "WASTE REVIEW APP\n",
  "============================================\n",
  "App folder: ", APP_DIR, "\n",
  "Chip folder: ", CHIPS_DIR, "\n",
  "Review chips: ", format(TOTAL_CHIPS, big.mark = ","), "\n",
  "Candidate objects: ", format(TOTAL_OBJECTS, big.mark = ","), "\n",
  "Database: ", DB_PATH, "\n",
  "============================================\n\n",
  sep = ""
)


# ============================================================
# 6. SQLITE CONNECTION
# ============================================================

db_connect <- function() {

  con <- DBI::dbConnect(
    RSQLite::SQLite(),
    DB_PATH
  )

  DBI::dbExecute(
    con,
    "PRAGMA journal_mode=WAL;"
  )

  DBI::dbExecute(
    con,
    "PRAGMA busy_timeout=5000;"
  )

  con
}


# ============================================================
# 7. INITIALISE DATABASE
# ============================================================

db_init <- function() {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  DBI::dbExecute(
    con,
    "
    CREATE TABLE IF NOT EXISTS annotations (
      chip_id      TEXT PRIMARY KEY,
      label        TEXT NOT NULL,
      annotator    TEXT NOT NULL,
      notes        TEXT,
      timestamp    TEXT NOT NULL
    );
    "
  )


  DBI::dbExecute(
    con,
    "
    CREATE TABLE IF NOT EXISTS annotation_history (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      chip_id      TEXT NOT NULL,
      label        TEXT NOT NULL,
      annotator    TEXT NOT NULL,
      notes        TEXT,
      timestamp    TEXT NOT NULL
    );
    "
  )


  DBI::dbExecute(
    con,
    "
    CREATE TABLE IF NOT EXISTS claims (
      chip_id      TEXT PRIMARY KEY,
      annotator    TEXT NOT NULL,
      claimed_at   TEXT NOT NULL
    );
    "
  )


  DBI::dbExecute(
    con,
    "
    CREATE INDEX IF NOT EXISTS idx_annotations_label
    ON annotations(label);
    "
  )


  DBI::dbExecute(
    con,
    "
    CREATE INDEX IF NOT EXISTS idx_annotations_annotator
    ON annotations(annotator);
    "
  )


  DBI::dbExecute(
    con,
    "
    CREATE INDEX IF NOT EXISTS idx_history_chip
    ON annotation_history(chip_id);
    "
  )


  DBI::dbExecute(
    con,
    "
    CREATE INDEX IF NOT EXISTS idx_history_annotator
    ON annotation_history(annotator);
    "
  )
}


db_init()


# ============================================================
# 8. GENERAL DATABASE HELPERS
# ============================================================

release_expired_claims <- function(
  con
) {

  cutoff <- format(
    Sys.time() -
      CLAIM_TIMEOUT_MINUTES * 60,
    "%Y-%m-%d %H:%M:%S"
  )


  DBI::dbExecute(
    con,
    "
    DELETE FROM claims
    WHERE claimed_at < ?;
    ",
    params = list(
      cutoff
    )
  )
}


touch_claim <- function(
  chip_id,
  annotator
) {

  if (
    is.null(chip_id) ||
    !nzchar(chip_id) ||
    is.null(annotator) ||
    !nzchar(annotator)
  ) {
    return(
      invisible(NULL)
    )
  }


  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  DBI::dbExecute(
    con,
    "
    UPDATE claims
    SET claimed_at = ?
    WHERE chip_id = ?
      AND annotator = ?;
    ",
    params = list(
      format(
        Sys.time(),
        "%Y-%m-%d %H:%M:%S"
      ),
      chip_id,
      annotator
    )
  )


  invisible(NULL)
}


release_claim <- function(
  chip_id,
  annotator = NULL
) {

  if (
    is.null(chip_id) ||
    !nzchar(chip_id)
  ) {
    return(
      invisible(NULL)
    )
  }


  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  if (is.null(annotator)) {

    DBI::dbExecute(
      con,
      "
      DELETE FROM claims
      WHERE chip_id = ?;
      ",
      params = list(
        chip_id
      )
    )

  } else {

    DBI::dbExecute(
      con,
      "
      DELETE FROM claims
      WHERE chip_id = ?
        AND annotator = ?;
      ",
      params = list(
        chip_id,
        annotator
      )
    )
  }


  invisible(NULL)
}


get_unavailable_chip_ids <- function(
  con
) {

  release_expired_claims(
    con
  )


  annotated <- DBI::dbGetQuery(
    con,
    "
    SELECT chip_id
    FROM annotations;
    "
  )$chip_id


  claimed <- DBI::dbGetQuery(
    con,
    "
    SELECT chip_id
    FROM claims;
    "
  )$chip_id


  union(
    annotated,
    claimed
  )
}


get_current_annotation <- function(
  chip_id
) {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  result <- DBI::dbGetQuery(
    con,
    "
    SELECT
      chip_id,
      label,
      annotator,
      notes,
      timestamp
    FROM annotations
    WHERE chip_id = ?;
    ",
    params = list(
      chip_id
    )
  )


  if (nrow(result) == 0) {
    return(NULL)
  }


  result
}


get_annotations <- function() {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  DBI::dbGetQuery(
    con,
    "
    SELECT
      chip_id,
      label,
      annotator,
      notes,
      timestamp
    FROM annotations;
    "
  )
}


# ============================================================
# 9. CLAIM NEXT UNREVIEWED CHIP
# ============================================================

claim_next_chip <- function(
  annotator,
  exclude_ids = character(0)
) {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  unavailable <- get_unavailable_chip_ids(
    con
  )


  unavailable <- union(
    unavailable,
    exclude_ids
  )


  available <- chips$chip_id[
    !chips$chip_id %in%
      unavailable
  ]


  if (length(available) == 0) {
    return(NULL)
  }


  available <- available[
    order(
      chip_order[
        available
      ]
    )
  ]


  for (chip_id in available) {

    inserted <- DBI::dbExecute(
      con,
      "
      INSERT OR IGNORE INTO claims (
        chip_id,
        annotator,
        claimed_at
      )
      VALUES (?, ?, ?);
      ",
      params = list(
        chip_id,
        annotator,
        format(
          Sys.time(),
          "%Y-%m-%d %H:%M:%S"
        )
      )
    )


    if (inserted == 1) {
      return(
        chip_id
      )
    }
  }


  NULL
}


# ============================================================
# 10. REVIEW HISTORY FOR CURRENT REVIEWER
# ============================================================

get_reviewer_history <- function(
  annotator
) {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  DBI::dbGetQuery(
    con,
    "
    SELECT
      chip_id,
      label,
      annotator,
      notes,
      timestamp
    FROM annotations
    WHERE annotator = ?
    ORDER BY timestamp DESC;
    ",
    params = list(
      annotator
    )
  )
}


# ============================================================
# 11. CLAIM A REVIEWED CHIP FOR CORRECTION
# ============================================================

claim_reviewed_chip <- function(
  chip_id,
  annotator
) {

  annotation <- get_current_annotation(
    chip_id
  )


  if (is.null(annotation)) {

    return(
      list(
        ok = FALSE,
        message = "This chip is not currently reviewed."
      )
    )
  }


  if (
    annotation$annotator[1] !=
      annotator
  ) {

    return(
      list(
        ok = FALSE,
        message = paste0(
          "This chip is currently assigned to ",
          annotation$annotator[1],
          ". You can only correct your own current reviews."
        )
      )
    )
  }


  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  release_expired_claims(
    con
  )


  existing_claim <- DBI::dbGetQuery(
    con,
    "
    SELECT
      chip_id,
      annotator
    FROM claims
    WHERE chip_id = ?;
    ",
    params = list(
      chip_id
    )
  )


  if (nrow(existing_claim) > 0) {

    if (
      existing_claim$annotator[1] !=
        annotator
    ) {

      return(
        list(
          ok = FALSE,
          message = paste0(
            "This chip is currently open in another review session."
          )
        )
      )
    }


    DBI::dbExecute(
      con,
      "
      UPDATE claims
      SET claimed_at = ?
      WHERE chip_id = ?
        AND annotator = ?;
      ",
      params = list(
        format(
          Sys.time(),
          "%Y-%m-%d %H:%M:%S"
        ),
        chip_id,
        annotator
      )
    )

  } else {

    inserted <- DBI::dbExecute(
      con,
      "
      INSERT OR IGNORE INTO claims (
        chip_id,
        annotator,
        claimed_at
      )
      VALUES (?, ?, ?);
      ",
      params = list(
        chip_id,
        annotator,
        format(
          Sys.time(),
          "%Y-%m-%d %H:%M:%S"
        )
      )
    )


    if (inserted != 1) {

      return(
        list(
          ok = FALSE,
          message = "Could not open this reviewed chip for correction."
        )
      )
    }
  }


  list(
    ok = TRUE,
    annotation = annotation
  )
}


# ============================================================
# 12. PRELOAD LIKELY NEXT UNREVIEWED CHIPS
# ============================================================

peek_next_chip_ids <- function(
  n = 3,
  exclude_ids = character(0)
) {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  unavailable <- get_unavailable_chip_ids(
    con
  )


  unavailable <- union(
    unavailable,
    exclude_ids
  )


  available <- chips$chip_id[
    !chips$chip_id %in%
      unavailable
  ]


  if (length(available) == 0) {
    return(
      character(0)
    )
  }


  available <- available[
    order(
      chip_order[
        available
      ]
    )
  ]


  head(
    available,
    n
  )
}


# ============================================================
# 13. SAVE / CORRECT CLASSIFICATION
# ============================================================

save_annotation <- function(
  chip_id,
  label,
  annotator,
  notes = ""
) {

  stopifnot(
    label %in%
      c(
        "waste",
        "not_waste",
        "mixed"
      )
  )


  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  now <- format(
    Sys.time(),
    "%Y-%m-%d %H:%M:%S"
  )


  DBI::dbBegin(
    con
  )


  tryCatch(
    {

      # Every original decision and correction is permanently kept.
      DBI::dbExecute(
        con,
        "
        INSERT INTO annotation_history (
          chip_id,
          label,
          annotator,
          notes,
          timestamp
        )
        VALUES (?, ?, ?, ?, ?);
        ",
        params = list(
          chip_id,
          label,
          annotator,
          notes,
          now
        )
      )


      # annotations contains only the current/latest state.
      DBI::dbExecute(
        con,
        "
        INSERT INTO annotations (
          chip_id,
          label,
          annotator,
          notes,
          timestamp
        )
        VALUES (?, ?, ?, ?, ?)

        ON CONFLICT(chip_id)
        DO UPDATE SET
          label = excluded.label,
          annotator = excluded.annotator,
          notes = excluded.notes,
          timestamp = excluded.timestamp;
        ",
        params = list(
          chip_id,
          label,
          annotator,
          notes,
          now
        )
      )


      DBI::dbExecute(
        con,
        "
        DELETE FROM claims
        WHERE chip_id = ?;
        ",
        params = list(
          chip_id
        )
      )


      DBI::dbCommit(
        con
      )

    },

    error = function(e) {

      DBI::dbRollback(
        con
      )

      stop(e)
    }
  )


  TRUE
}


# ============================================================
# 14. LIVE STATISTICS
# ============================================================

get_stats <- function() {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  counts <- DBI::dbGetQuery(
    con,
    "
    SELECT
      COUNT(*) AS reviewed,

      SUM(
        CASE
        WHEN label = 'waste'
        THEN 1
        ELSE 0
        END
      ) AS waste,

      SUM(
        CASE
        WHEN label = 'not_waste'
        THEN 1
        ELSE 0
        END
      ) AS not_waste,

      SUM(
        CASE
        WHEN label = 'mixed'
        THEN 1
        ELSE 0
        END
      ) AS mixed

    FROM annotations;
    "
  )


  board <- DBI::dbGetQuery(
    con,
    "
    SELECT
      annotator,
      COUNT(*) AS n_reviewed
    FROM annotations
    GROUP BY annotator
    ORDER BY
      n_reviewed DESC,
      annotator ASC;
    "
  )


  for (
    nm in c(
      "reviewed",
      "waste",
      "not_waste",
      "mixed"
    )
  ) {

    if (
      is.na(
        counts[[nm]][1]
      )
    ) {
      counts[[nm]][1] <- 0
    }
  }


  list(

    reviewed =
      as.integer(
        counts$reviewed[1]
      ),

    waste =
      as.integer(
        counts$waste[1]
      ),

    not_waste =
      as.integer(
        counts$not_waste[1]
      ),

    mixed =
      as.integer(
        counts$mixed[1]
      ),

    board =
      board
  )
}


# ============================================================
# 15. PROVISIONAL REMAINING OBJECT COUNT
# ============================================================

calculate_remaining_objects <- function() {

  annotations <- get_annotations()


  if (nrow(annotations) == 0) {
    return(
      TOTAL_OBJECTS
    )
  }


  status_lookup <- setNames(
    annotations$label,
    annotations$chip_id
  )


  map_status <- status_lookup[
    object_chip_map$chip_id
  ]


  map_status[
    is.na(map_status)
  ] <- "unreviewed"


  status_by_object <- split(
    map_status,
    object_chip_map$candidate_id
  )


  keep <- vapply(
    status_by_object,
    function(x) {

      !all(
        x == "not_waste"
      )
    },
    logical(1)
  )


  sum(
    keep
  )
}


# ============================================================
# 16. EXPORT HELPERS
# ============================================================

write_gpkg_layer <- function(
  x,
  layer
) {

  if (nrow(x) == 0) {
    return(
      invisible(FALSE)
    )
  }


  sf::st_write(
    x,
    dsn = EXPORT_GPKG,
    layer = layer,
    delete_layer = TRUE,
    quiet = TRUE
  )


  invisible(TRUE)
}


export_review_results <- function() {

  annotations <- get_annotations()


  # ----------------------------------------------------------
  # Review grid + current classifications
  # ----------------------------------------------------------

  review_grid <- sf::st_read(
    REVIEW_GRID_PATH,
    quiet = TRUE
  )


  if (!"chip_id" %in% names(review_grid)) {
    stop(
      "review_grid.gpkg does not contain chip_id."
    )
  }


  m <- match(
    review_grid$chip_id,
    annotations$chip_id
  )


  review_grid$review_class <- ifelse(
    is.na(m),
    "unreviewed",
    annotations$label[m]
  )


  review_grid$annotator <- ifelse(
    is.na(m),
    NA_character_,
    annotations$annotator[m]
  )


  review_grid$notes <- ifelse(
    is.na(m),
    NA_character_,
    annotations$notes[m]
  )


  review_grid$reviewed_at <- ifelse(
    is.na(m),
    NA_character_,
    annotations$timestamp[m]
  )


  # ----------------------------------------------------------
  # CSV
  # ----------------------------------------------------------

  chip_review_table <- sf::st_drop_geometry(
    review_grid
  )


  write.csv(
    chip_review_table,
    EXPORT_CSV,
    row.names = FALSE
  )


  # ----------------------------------------------------------
  # GeoPackage
  # ----------------------------------------------------------

  if (file.exists(EXPORT_GPKG)) {
    file.remove(
      EXPORT_GPKG
    )
  }


  write_gpkg_layer(
    review_grid,
    "reviewed_grid"
  )


  retained_locations <- review_grid[
    review_grid$review_class == "waste",
    ,
    drop = FALSE
  ]


  second_review_locations <- review_grid[
    review_grid$review_class == "mixed",
    ,
    drop = FALSE
  ]


  rejected_locations <- review_grid[
    review_grid$review_class == "not_waste",
    ,
    drop = FALSE
  ]


  pending_locations <- review_grid[
    review_grid$review_class == "unreviewed",
    ,
    drop = FALSE
  ]


  write_gpkg_layer(
    retained_locations,
    "retained_locations"
  )


  write_gpkg_layer(
    second_review_locations,
    "second_review_locations"
  )


  write_gpkg_layer(
    rejected_locations,
    "rejected_locations"
  )


  write_gpkg_layer(
    pending_locations,
    "pending_locations"
  )


  # ----------------------------------------------------------
  # Object-level review class
  # ----------------------------------------------------------

  status_lookup <- setNames(
    review_grid$review_class,
    review_grid$chip_id
  )


  map_status <- status_lookup[
    object_chip_map$chip_id
  ]


  map_status[
    is.na(map_status)
  ] <- "unreviewed"


  status_by_object <- split(
    map_status,
    object_chip_map$candidate_id
  )


  classify_object <- function(x) {

    x <- unique(x)


    if ("unreviewed" %in% x) {
      return(
        "pending"
      )
    }


    if ("mixed" %in% x) {
      return(
        "second_review"
      )
    }


    if (all(x == "waste")) {
      return(
        "retained"
      )
    }


    if (all(x == "not_waste")) {
      return(
        "rejected"
      )
    }


    # Waste + not_waste across different chips is unresolved.
    "second_review"
  }


  object_review <- vapply(
    status_by_object,
    classify_object,
    character(1)
  )


  object_class_table <- data.frame(

    candidate_id =
      names(object_review),

    review_class =
      unname(object_review),

    stringsAsFactors =
      FALSE
  )


  object_class_table$candidate_id <-
    as.character(
      object_class_table$candidate_id
    )


  # ----------------------------------------------------------
  # Original prediction polygons
  # ----------------------------------------------------------

  predictions <- sf::st_read(
    PREDICTION_PATH,
    quiet = TRUE
  )


  if (!"candidate_id" %in% names(predictions)) {

    predictions$candidate_id <- seq_len(
      nrow(predictions)
    )
  }


  predictions$candidate_id <-
    as.character(
      predictions$candidate_id
    )


  p_match <- match(
    predictions$candidate_id,
    object_class_table$candidate_id
  )


  predictions$review_class <- ifelse(
    is.na(p_match),
    "pending",
    object_class_table$review_class[p_match]
  )


  retained_predictions <- predictions[
    predictions$review_class == "retained",
    ,
    drop = FALSE
  ]


  second_review_predictions <- predictions[
    predictions$review_class == "second_review",
    ,
    drop = FALSE
  ]


  rejected_predictions <- predictions[
    predictions$review_class == "rejected",
    ,
    drop = FALSE
  ]


  pending_predictions <- predictions[
    predictions$review_class == "pending",
    ,
    drop = FALSE
  ]


  write_gpkg_layer(
    retained_predictions,
    "retained_predictions"
  )


  write_gpkg_layer(
    second_review_predictions,
    "second_review_predictions"
  )


  write_gpkg_layer(
    rejected_predictions,
    "rejected_predictions"
  )


  write_gpkg_layer(
    pending_predictions,
    "pending_predictions"
  )


  # ----------------------------------------------------------
  # Summary
  # ----------------------------------------------------------

  summary_table <- data.frame(

    measure = c(
      "total_review_chips",
      "retained_locations",
      "second_review_locations",
      "rejected_locations",
      "pending_locations",
      "total_candidate_objects",
      "retained_predictions",
      "second_review_predictions",
      "rejected_predictions",
      "pending_predictions"
    ),

    value = c(
      nrow(review_grid),
      nrow(retained_locations),
      nrow(second_review_locations),
      nrow(rejected_locations),
      nrow(pending_locations),
      nrow(predictions),
      nrow(retained_predictions),
      nrow(second_review_predictions),
      nrow(rejected_predictions),
      nrow(pending_predictions)
    ),

    stringsAsFactors =
      FALSE
  )


  write.csv(
    summary_table,
    EXPORT_SUMMARY_CSV,
    row.names = FALSE
  )


  list(
    gpkg = EXPORT_GPKG,
    chip_csv = EXPORT_CSV,
    summary_csv = EXPORT_SUMMARY_CSV,
    summary = summary_table
  )
}


# ============================================================
# 17. USER INTERFACE
# ============================================================

ui <- fluidPage(


  tags$head(


    # --------------------------------------------------------
    # Keyboard shortcuts
    # --------------------------------------------------------

    tags$script(
      HTML(
        "
        document.addEventListener('keydown', function(e) {

          var tag = document.activeElement.tagName.toLowerCase();

          if (tag === 'input' || tag === 'textarea') {
            return;
          }

          if (e.key === '1') {
            document.getElementById('mark_waste').click();
          }

          if (e.key === '2') {
            document.getElementById('mark_not_waste').click();
          }

          if (e.key === '3') {
            document.getElementById('mark_mixed').click();
          }

          if (e.key.toLowerCase() === 's') {
            document.getElementById('skip_btn').click();
          }
        });
        "
      )
    ),


    tags$style(
      HTML(
        "
        html,
        body {
          width: 100%;
          height: 100%;
          margin: 0;
          padding: 0;
          overflow: hidden;
          background-color: #f3f3f3;
        }

        .container-fluid {
          width: 100%;
          height: 100vh;
          padding: 6px 10px;
          overflow: hidden;
        }

        .app-title {
          height: 5vh;
          min-height: 40px;
          display: flex;
          align-items: center;
          font-size: 28px;
          font-weight: 500;
          padding-left: 3px;
        }

        .main-row {
          height: calc(95vh - 12px);
          margin: 0;
        }

        .sidebar-column {
          height: 100%;
          overflow-y: auto;
          overflow-x: hidden;
          padding-right: 8px;
        }

        .image-column {
          height: 100%;
          overflow: hidden;
          padding-left: 8px;
        }

        .box {
          background: white;
          border-radius: 6px;
          border: 1px solid #dddddd;
          padding: 8px 9px;
          margin-bottom: 6px;
        }

        .box h4 {
          font-size: 16px;
          margin: 0 0 6px 0;
        }

        .chip-title {
          font-size: 19px;
          font-weight: 700;
          margin-bottom: 3px;
        }

        .small-line {
          font-size: 13px;
          margin: 2px 0;
        }

        .correction-banner {
          background: #fff1bf;
          border: 1px solid #d6b75e;
          border-radius: 5px;
          padding: 6px 7px;
          margin-top: 5px;
          font-size: 13px;
          font-weight: 700;
        }

        .current-class {
          margin-top: 5px;
          font-size: 14px;
          font-weight: 700;
        }

        .classification-button {
          width: 100%;
          margin-bottom: 5px;
          padding-top: 7px;
          padding-bottom: 7px;
          font-size: 15px;
        }

        .btn-waste {
          background-color: #e9f6e9;
          border-color: #aacdaa;
        }

        .btn-nowaste {
          background-color: #f8e8e8;
          border-color: #d7aaaa;
        }

        .btn-mixed {
          background-color: #fff5dd;
          border-color: #d9c38d;
        }

        .history-nav-row {
          display: flex;
          gap: 5px;
          width: 100%;
          margin-top: 5px;
        }

        .history-nav-row .btn {
          flex: 1;
        }

        .image-box {
          height: 100%;
          background: white;
          border-radius: 6px;
          border: 1px solid #dddddd;
          padding: 7px;
          overflow: hidden;
        }

        .image-header {
          height: 34px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          font-size: 18px;
          font-weight: 500;
          padding: 0 4px;
        }

        .image-container {
          width: 100%;
          height: calc(100% - 34px);
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
          background-color: #111111;
          position: relative;
        }

        .review-image {
          width: 100%;
          height: 100%;
          object-fit: contain;
        }

        .preload-image {
          position: absolute;
          width: 1px;
          height: 1px;
          left: -9999px;
          top: -9999px;
          opacity: 0;
          pointer-events: none;
        }

        .metric {
          font-size: 13px;
          line-height: 1.25;
          margin-bottom: 2px;
        }

        textarea.form-control {
          height: 45px !important;
          min-height: 45px !important;
          resize: none;
        }

        .form-group {
          margin-bottom: 5px;
        }

        hr {
          margin-top: 5px;
          margin-bottom: 5px;
        }

        table {
          font-size: 12px;
        }

        pre {
          font-size: 11px;
          padding: 4px;
          margin: 4px 0 0 0;
          white-space: pre-wrap;
        }
        "
      )
    )
  ),


  div(
    class = "app-title",
    "Waste Prediction Review"
  ),


  fluidRow(

    class = "main-row",


    # ========================================================
    # SIDEBAR
    # ========================================================

    column(

      width = 2,

      class = "sidebar-column",


      div(
        class = "box",
        uiOutput(
          "welcome_text"
        )
      ),


      div(
        class = "box",
        uiOutput(
          "chip_info"
        )
      ),


      div(

        class = "box",

        h4(
          "Classification"
        ),


        actionButton(
          "mark_waste",
          "1  Waste",
          class = "classification-button btn-waste"
        ),


        actionButton(
          "mark_not_waste",
          "2  Not waste",
          class = "classification-button btn-nowaste"
        ),


        actionButton(
          "mark_mixed",
          "3  Waste + false positives",
          class = "classification-button btn-mixed"
        ),


        actionButton(
          "skip_btn",
          "S  Skip",
          width = "100%"
        )
      ),


      div(

        class = "box",

        textAreaInput(
          "notes",
          "Notes",
          value = "",
          rows = 2,
          width = "100%"
        )
      ),


      # ------------------------------------------------------
      # VISUAL CORRECTION
      # ------------------------------------------------------

      div(

        class = "box",

        h4(
          "Correct earlier reviews"
        ),


        actionButton(
          "edit_last",
          "Edit my last review",
          width = "100%"
        ),


        div(
          style = "height:5px;"
        ),


        actionButton(
          "browse_reviews",
          "Browse my reviewed images",
          width = "100%"
        ),


        uiOutput(
          "history_controls"
        )
      ),


      div(

        class = "box",

        h4(
          "Progress"
        ),

        uiOutput(
          "progress_ui"
        ),


        actionButton(
          "update_object_count",
          "Update pile count",
          width = "100%"
        ),


        div(
          style = "height:4px;"
        ),


        textOutput(
          "object_count_text"
        )
      ),


      div(

        class = "box",

        h4(
          "Export"
        ),

        actionButton(
          "export_results",
          "Export spatial results",
          width = "100%"
        ),


        verbatimTextOutput(
          "export_status"
        )
      ),


      div(

        class = "box",

        h4(
          "Reviewers"
        ),

        tableOutput(
          "leaderboard_table"
        )
      )

    ),


    # ========================================================
    # IMAGE
    # ========================================================

    column(

      width = 10,

      class = "image-column",


      div(

        class = "image-box",


        div(

          class = "image-header",

          span(
            "Orthomosaic with predicted waste outlines"
          ),

          span(
            textOutput(
              "image_counter",
              inline = TRUE
            )
          )
        ),


        div(

          class = "image-container",

          uiOutput(
            "current_image"
          )
        )
      )

    )

  )

)


# ============================================================
# 18. SERVER
# ============================================================

server <- function(
  input,
  output,
  session
) {


  # ----------------------------------------------------------
  # Reviewer identity
  # ----------------------------------------------------------

  annotator_name <- reactiveVal(
    NULL
  )


  showModal(
    modalDialog(

      title = "Waste prediction review",

      textInput(
        "name_input",
        "Enter your name:"
      ),

      footer = actionButton(
        "name_submit",
        "Start reviewing",
        class = "btn-primary"
      ),

      easyClose = FALSE
    )
  )


  observeEvent(
    input$name_submit,
    {

      nm <- trimws(
        input$name_input
      )


      if (
        nzchar(nm)
      ) {

        annotator_name(
          nm
        )

        removeModal()

      } else {

        showNotification(
          "Please enter your name.",
          type = "error",
          duration = 4
        )
      }
    }
  )


  output$welcome_text <- renderUI({

    req(
      annotator_name()
    )


    tagList(

      div(
        class = "small-line",

        strong(
          paste0(
            "Reviewer: ",
            annotator_name()
          )
        )
      ),


      div(
        class = "small-line",
        "Red outlines = predicted waste."
      ),


      div(
        class = "small-line",
        "1 = waste, 2 = not waste, 3 = mixed."
      )
    )
  })


  # ----------------------------------------------------------
  # Session state
  # ----------------------------------------------------------

  state <- reactiveValues(

    current_chip_id = NULL,

    mode = "review",

    previous_label = NULL,

    history_ids = character(0),

    history_index = NA_integer_

  )


  session_skipped <- reactiveVal(
    character(0)
  )


  provisional_object_count <- reactiveVal(
    NA_integer_
  )


  object_count_time <- reactiveVal(
    NULL
  )


  export_message <- reactiveVal(
    ""
  )


  # ----------------------------------------------------------
  # Release whatever is currently open
  # ----------------------------------------------------------

  release_current_claim <- function() {

    current_id <- state$current_chip_id


    if (
      !is.null(current_id) &&
      !is.null(
        annotator_name()
      )
    ) {

      release_claim(
        current_id,
        annotator_name()
      )
    }
  }


  # ----------------------------------------------------------
  # Load next unreviewed chip
  # ----------------------------------------------------------

  load_next_chip <- function() {

    req(
      annotator_name()
    )


    next_id <- claim_next_chip(
      annotator = annotator_name(),
      exclude_ids = session_skipped()
    )


    if (
      is.null(next_id) &&
      length(session_skipped()) > 0
    ) {

      session_skipped(
        character(0)
      )


      next_id <- claim_next_chip(
        annotator = annotator_name(),
        exclude_ids = character(0)
      )
    }


    state$current_chip_id <- next_id

    state$mode <- "review"

    state$previous_label <- NULL

    state$history_ids <- character(0)

    state$history_index <- NA_integer_


    updateTextAreaInput(
      session,
      "notes",
      value = ""
    )
  }


  # ----------------------------------------------------------
  # Start correction mode from a visual history snapshot
  # ----------------------------------------------------------

  start_history_mode <- function(
    index = 1L
  ) {

    req(
      annotator_name()
    )


    history <- get_reviewer_history(
      annotator_name()
    )


    if (nrow(history) == 0) {

      showNotification(
        "You do not yet have any completed reviews to browse.",
        type = "warning",
        duration = 5
      )

      return(
        invisible(FALSE)
      )
    }


    # Release the unreviewed chip currently being held.
    release_current_claim()


    state$history_ids <- history$chip_id


    index <- max(
      1L,
      min(
        as.integer(index),
        length(state$history_ids)
      )
    )


    state$history_index <- index


    opened <- open_history_index(
      index
    )


    invisible(
      opened
    )
  }


  # ----------------------------------------------------------
  # Open one reviewed image from the history snapshot
  # ----------------------------------------------------------

  open_history_index <- function(
    index
  ) {

    req(
      annotator_name()
    )


    if (
      length(state$history_ids) == 0
    ) {
      return(
        invisible(FALSE)
      )
    }


    index <- max(
      1L,
      min(
        as.integer(index),
        length(state$history_ids)
      )
    )


    # Release any previously opened history chip.
    if (
      identical(
        state$mode,
        "history"
      ) &&
      !is.null(
        state$current_chip_id
      )
    ) {

      release_claim(
        state$current_chip_id,
        annotator_name()
      )
    }


    chip_id <- state$history_ids[
      index
    ]


    result <- claim_reviewed_chip(
      chip_id,
      annotator_name()
    )


    if (!isTRUE(result$ok)) {

      showNotification(
        result$message,
        type = "warning",
        duration = 6
      )

      return(
        invisible(FALSE)
      )
    }


    existing <- result$annotation


    state$current_chip_id <- chip_id

    state$mode <- "history"

    state$history_index <- index

    state$previous_label <- existing$label[1]


    updateTextAreaInput(
      session,
      "notes",
      value = ifelse(
        is.na(
          existing$notes[1]
        ),
        "",
        existing$notes[1]
      )
    )


    invisible(TRUE)
  }


  # ----------------------------------------------------------
  # Refresh current history chip after correction
  # without changing the frozen browsing order
  # ----------------------------------------------------------

  refresh_history_current <- function() {

    if (
      !identical(
        state$mode,
        "history"
      ) ||
      is.na(
        state$history_index
      ) ||
      length(state$history_ids) == 0
    ) {
      return(
        invisible(FALSE)
      )
    }


    chip_id <- state$history_ids[
      state$history_index
    ]


    result <- claim_reviewed_chip(
      chip_id,
      annotator_name()
    )


    if (!isTRUE(result$ok)) {

      showNotification(
        result$message,
        type = "warning",
        duration = 6
      )

      return(
        invisible(FALSE)
      )
    }


    existing <- result$annotation


    state$current_chip_id <- chip_id

    state$previous_label <- existing$label[1]


    updateTextAreaInput(
      session,
      "notes",
      value = ifelse(
        is.na(
          existing$notes[1]
        ),
        "",
        existing$notes[1]
      )
    )


    invisible(TRUE)
  }


  # ----------------------------------------------------------
  # Initial chip after login
  # ----------------------------------------------------------

  observeEvent(
    annotator_name(),
    {

      req(
        annotator_name()
      )

      load_next_chip()

    },
    ignoreInit = TRUE
  )


  # ----------------------------------------------------------
  # Keep current claim alive
  # ----------------------------------------------------------

  observe({

    invalidateLater(
      CLAIM_HEARTBEAT_SECONDS * 1000,
      session
    )


    current_id <- isolate(
      state$current_chip_id
    )


    reviewer <- isolate(
      annotator_name()
    )


    if (
      !is.null(current_id) &&
      !is.null(reviewer)
    ) {

      touch_claim(
        current_id,
        reviewer
      )
    }
  })


  # ----------------------------------------------------------
  # Release claim when browser closes
  # ----------------------------------------------------------

  session$onSessionEnded(
    function() {

      current_id <- isolate(
        state$current_chip_id
      )


      reviewer <- isolate(
        annotator_name()
      )


      if (
        !is.null(current_id) &&
        !is.null(reviewer)
      ) {

        release_claim(
          current_id,
          reviewer
        )
      }
    }
  )


  # ==========================================================
  # 19. CURRENT CHIP INFORMATION
  # ==========================================================

  current_chip_row <- reactive({

    req(
      state$current_chip_id
    )


    row <- chips[
      chips$chip_id ==
        state$current_chip_id,
      ,
      drop = FALSE
    ]


    req(
      nrow(row) == 1
    )


    row
  })


  output$chip_info <- renderUI({

    if (
      is.null(
        state$current_chip_id
      )
    ) {

      return(
        tagList(

          div(
            class = "chip-title",
            "Review complete"
          ),


          div(
            class = "small-line",
            "No unreviewed chips are currently available."
          )
        )
      )
    }


    current <- current_chip_row()


    elements <- list(

      div(
        class = "chip-title",
        current$chip_id
      ),


      div(
        class = "small-line",

        paste(
          "Candidate objects:",
          current$n_objects
        )
      )
    )


    if (
      identical(
        state$mode,
        "history"
      )
    ) {

      elements <- c(
        elements,
        list(

          div(
            class = "correction-banner",
            "CORRECTION MODE"
          ),


          div(
            class = "current-class",

            paste0(
              "Current classification: ",
              state$previous_label
            )
          )
        )
      )
    }


    tagList(
      elements
    )
  })


  # ==========================================================
  # 20. VISUAL HISTORY CONTROLS
  # ==========================================================

  output$history_controls <- renderUI({

    if (
      !identical(
        state$mode,
        "history"
      )
    ) {

      return(NULL)
    }


    n_history <- length(
      state$history_ids
    )


    idx <- state$history_index


    tagList(

      div(
        class = "small-line",
        style = "margin-top:6px;",

        paste0(
          "Reviewed image ",
          idx,
          " of ",
          n_history,
          " (most recent first)"
        )
      ),


      div(

        class = "history-nav-row",


        actionButton(
          "history_newer",
          "← Newer",
          disabled = if (
            idx <= 1
          ) "disabled" else NULL
        ),


        actionButton(
          "history_older",
          "Older →",
          disabled = if (
            idx >= n_history
          ) "disabled" else NULL
        )
      ),


      div(
        style = "height:5px;"
      ),


      actionButton(
        "return_to_review",
        "Return to unreviewed",
        width = "100%"
      ),


      div(
        class = "small-line",
        style = "margin-top:5px;",
        "Changing a class updates the current result but preserves the old decision in the audit history."
      )
    )
  })


  # ==========================================================
  # 21. IMAGE DISPLAY + PRELOAD
  # ==========================================================

  output$current_image <- renderUI({

    if (
      is.null(
        state$current_chip_id
      )
    ) {

      return(
        div(
          style = "
            color:white;
            font-size:24px;
            text-align:center;
          ",
          "No unreviewed chips remaining."
        )
      )
    }


    current <- current_chip_row()


    current_url <- paste0(
      "review_chips/",
      URLencode(
        current$image_file,
        reserved = TRUE
      )
    )


    preload_tags <- NULL


    # --------------------------------------------------------
    # Normal mode: preload likely next unreviewed images
    # --------------------------------------------------------

    if (
      identical(
        state$mode,
        "review"
      )
    ) {

      next_ids <- peek_next_chip_ids(
        n = N_PRELOAD,
        exclude_ids = c(
          state$current_chip_id,
          session_skipped()
        )
      )


      if (
        length(next_ids) > 0
      ) {

        next_rows <- chips[
          match(
            next_ids,
            chips$chip_id
          ),
          ,
          drop = FALSE
        ]


        preload_tags <- lapply(
          next_rows$image_file,
          function(fname) {

            tags$img(
              src = paste0(
                "review_chips/",
                URLencode(
                  fname,
                  reserved = TRUE
                )
              ),
              class = "preload-image"
            )
          }
        )
      }
    }


    # --------------------------------------------------------
    # Correction mode: preload adjacent reviewed images
    # --------------------------------------------------------

    if (
      identical(
        state$mode,
        "history"
      )
    ) {

      idx <- state$history_index

      history_indices <- unique(
        c(
          idx - 1L,
          idx + 1L
        )
      )


      history_indices <- history_indices[
        history_indices >= 1 &
          history_indices <=
          length(
            state$history_ids
          )
      ]


      if (
        length(history_indices) > 0
      ) {

        ids <- state$history_ids[
          history_indices
        ]


        rows <- chips[
          match(
            ids,
            chips$chip_id
          ),
          ,
          drop = FALSE
        ]


        preload_tags <- lapply(
          rows$image_file,
          function(fname) {

            tags$img(
              src = paste0(
                "review_chips/",
                URLencode(
                  fname,
                  reserved = TRUE
                )
              ),
              class = "preload-image"
            )
          }
        )
      }
    }


    tagList(

      tags$img(
        src = current_url,
        class = "review-image",
        alt = current$chip_id
      ),


      preload_tags
    )
  })


  # ==========================================================
  # 22. SAVE NORMAL REVIEW OR CORRECTION
  # ==========================================================

  classify_current <- function(
    label
  ) {

    req(
      annotator_name()
    )


    chip_id <- state$current_chip_id


    if (
      is.null(chip_id)
    ) {
      return(
        invisible(NULL)
      )
    }


    correction_mode <- identical(
      state$mode,
      "history"
    )


    old_label <- state$previous_label


    note_text <- if (
      is.null(input$notes)
    ) {

      ""

    } else {

      trimws(
        input$notes
      )
    }


    saved <- tryCatch(

      save_annotation(
        chip_id = chip_id,
        label = label,
        annotator = annotator_name(),
        notes = note_text
      ),

      error = function(e) {

        showNotification(
          paste0(
            "Could not save classification: ",
            conditionMessage(e)
          ),
          type = "error",
          duration = 8
        )

        FALSE
      }
    )


    if (
      !isTRUE(saved)
    ) {
      return(
        invisible(FALSE)
      )
    }


    provisional_object_count(
      NA_integer_
    )

    object_count_time(
      NULL
    )


    # --------------------------------------------------------
    # Correction mode:
    # stay on the same visual image so the user can verify the
    # corrected class, then browse backward/forward as desired.
    # --------------------------------------------------------

    if (
      correction_mode
    ) {

      showNotification(
        paste0(
          "Changed from ",
          old_label,
          " to ",
          label,
          "."
        ),
        type = "message",
        duration = 4
      )


      refresh_history_current()


      return(
        invisible(TRUE)
      )
    }


    # --------------------------------------------------------
    # Normal mode:
    # save and move directly to the next unreviewed chip.
    # --------------------------------------------------------

    load_next_chip()


    invisible(TRUE)
  }


  observeEvent(
    input$mark_waste,
    {

      classify_current(
        "waste"
      )
    }
  )


  observeEvent(
    input$mark_not_waste,
    {

      classify_current(
        "not_waste"
      )
    }
  )


  observeEvent(
    input$mark_mixed,
    {

      classify_current(
        "mixed"
      )
    }
  )


  # ==========================================================
  # 23. NORMAL SKIP / CORRECTION-MODE NEXT
  # ==========================================================

  observeEvent(
    input$skip_btn,
    {

      req(
        annotator_name()
      )


      if (
        identical(
          state$mode,
          "history"
        )
      ) {

        # Do not change the current classification.
        # S simply advances to the next older reviewed image.
        if (
          state$history_index <
            length(
              state$history_ids
            )
        ) {

          open_history_index(
            state$history_index + 1L
          )

        } else {

          showNotification(
            "You are at the oldest reviewed image in this browsing session.",
            type = "message",
            duration = 4
          )
        }


        return()
      }


      old_chip <- state$current_chip_id


      if (
        !is.null(old_chip)
      ) {

        session_skipped(
          unique(
            c(
              session_skipped(),
              old_chip
            )
          )
        )


        release_claim(
          old_chip,
          annotator_name()
        )
      }


      load_next_chip()
    }
  )


  # ==========================================================
  # 24. EDIT LAST REVIEW
  # ==========================================================

  observeEvent(
    input$edit_last,
    {

      req(
        annotator_name()
      )


      start_history_mode(
        1L
      )
    }
  )


  # ==========================================================
  # 25. BROWSE REVIEWED IMAGES
  # ==========================================================

  observeEvent(
    input$browse_reviews,
    {

      req(
        annotator_name()
      )


      start_history_mode(
        1L
      )
    }
  )


  # ----------------------------------------------------------
  # Newer review
  # ----------------------------------------------------------

  observeEvent(
    input$history_newer,
    {

      req(
        identical(
          state$mode,
          "history"
        )
      )


      if (
        state$history_index > 1L
      ) {

        open_history_index(
          state$history_index - 1L
        )
      }
    }
  )


  # ----------------------------------------------------------
  # Older review
  # ----------------------------------------------------------

  observeEvent(
    input$history_older,
    {

      req(
        identical(
          state$mode,
          "history"
        )
      )


      if (
        state$history_index <
          length(
            state$history_ids
          )
      ) {

        open_history_index(
          state$history_index + 1L
        )
      }
    }
  )


  # ----------------------------------------------------------
  # Return to ordinary queue
  # ----------------------------------------------------------

  observeEvent(
    input$return_to_review,
    {

      req(
        annotator_name()
      )


      release_current_claim()


      load_next_chip()
    }
  )


  # ==========================================================
  # 26. LIGHTWEIGHT LIVE STATISTICS
  # ==========================================================

  stats_timer <- reactiveTimer(
    STATS_REFRESH_MS,
    session
  )


  stats_tick <- reactive({

    stats_timer()

    get_stats()
  })


  output$progress_ui <- renderUI({

    s <- stats_tick()


    remaining <- TOTAL_CHIPS -
      s$reviewed


    pct <- if (
      TOTAL_CHIPS > 0
    ) {

      round(
        100 *
          s$reviewed /
          TOTAL_CHIPS,
        1
      )

    } else {

      0
    }


    tagList(

      div(
        class = "metric",

        strong(
          paste0(
            pct,
            "% reviewed"
          )
        )
      ),


      div(
        class = "metric",

        paste0(
          s$reviewed,
          " / ",
          TOTAL_CHIPS,
          " chips"
        )
      ),


      div(
        class = "metric",

        paste(
          "Remaining:",
          remaining
        )
      ),


      hr(),


      div(
        class = "metric",

        paste(
          "Waste:",
          s$waste
        )
      ),


      div(
        class = "metric",

        paste(
          "Not waste:",
          s$not_waste
        )
      ),


      div(
        class = "metric",

        paste(
          "Mixed:",
          s$mixed
        )
      )
    )
  })


  output$leaderboard_table <- renderTable(
    {

      stats_tick()$board
    },

    striped = TRUE,
    spacing = "xs"
  )


  output$image_counter <- renderText({

    s <- stats_tick()


    if (
      identical(
        state$mode,
        "history"
      )
    ) {

      paste0(
        "CORRECTION MODE | ",
        state$history_index,
        " / ",
        length(
          state$history_ids
        ),
        " of your reviewed images"
      )

    } else {

      paste0(
        "Reviewed ",
        s$reviewed,
        " / ",
        TOTAL_CHIPS
      )
    }
  })


  # ==========================================================
  # 27. PILE COUNT ON DEMAND
  # ==========================================================

  observeEvent(
    input$update_object_count,
    {

      showNotification(
        "Calculating provisional remaining pile count...",
        type = "message",
        duration = 2
      )


      count <- calculate_remaining_objects()


      provisional_object_count(
        count
      )


      object_count_time(
        format(
          Sys.time(),
          "%H:%M:%S"
        )
      )
    }
  )


  output$object_count_text <- renderText({

    count <- provisional_object_count()


    if (
      is.na(count)
    ) {

      return(
        paste0(
          "Original candidate objects: ",
          format(
            TOTAL_OBJECTS,
            big.mark = ","
          ),
          "\nClick 'Update pile count' when needed."
        )
      )
    }


    paste0(
      "Provisional objects remaining: ",
      format(
        count,
        big.mark = ","
      ),
      "\nUpdated ",
      object_count_time()
    )
  })


  # ==========================================================
  # 28. EXPORT SPATIAL RESULTS
  # ==========================================================

  observeEvent(
    input$export_results,
    {

      export_message(
        "Exporting..."
      )


      showNotification(
        "Exporting review grid and prediction layers...",
        type = "message",
        duration = 3
      )


      result <- tryCatch(

        export_review_results(),

        error = function(e) {

          export_message(
            paste0(
              "Export failed:\n",
              conditionMessage(e)
            )
          )


          showNotification(
            paste0(
              "Export failed: ",
              conditionMessage(e)
            ),
            type = "error",
            duration = 10
          )


          NULL
        }
      )


      if (!is.null(result)) {

        s <- result$summary


        get_n <- function(name) {

          x <- s$value[
            s$measure == name
          ]


          if (length(x) == 0) {
            return(0)
          }


          x[1]
        }


        export_message(
          paste0(
            "Export complete\n",
            "Retained locations: ",
            get_n("retained_locations"),
            "\nSecond-review locations: ",
            get_n("second_review_locations"),
            "\nRejected locations: ",
            get_n("rejected_locations"),
            "\nPending locations: ",
            get_n("pending_locations"),
            "\n\nRetained predictions: ",
            get_n("retained_predictions"),
            "\nSecond-review predictions: ",
            get_n("second_review_predictions"),
            "\nRejected predictions: ",
            get_n("rejected_predictions"),
            "\nPending predictions: ",
            get_n("pending_predictions"),
            "\n\nSaved to:\n",
            EXPORT_DIR
          )
        )


        showNotification(
          "Spatial review export completed.",
          type = "message",
          duration = 5
        )
      }
    }
  )


  output$export_status <- renderText({

    export_message()
  })

}


# ============================================================
# 29. RUN APP
# ============================================================

shinyApp(
  ui = ui,
  server = server
)
