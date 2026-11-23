# ============================================================
# TWO-STAGE WASTE PREDICTION REVIEW APP
# ============================================================
#
# STAGE 1: 50 m x 50 m location review
#   1 = Waste
#   2 = Not waste
#   3 = Waste + false positives
#   4 = Not sure
#
# STAGE 2: object-level review of ambiguous predictions only
#   1 = Waste
#   2 = Not waste
#   3 = Not sure
#
# STAGE 3: final positive audit of every retained waste object
#   1 = Keep
#   2 = Delete
#   3 = Needs edit
#   4 = Not sure
#
# Stage 2 uses the EXISTING first-stage PNGs. It does not create
# thousands of new image files. The target prediction is located
# with a thick yellow bounding box with a fully transparent interior.
# A cyan outline shows the exact predicted object geometry.
#
# IMPORTANT:
# Stage 2 applies a minimum predicted-object area of 0.25 m2.
# Smaller predicted fragments are preserved in the final export
# as "below_min_area_predictions" but are not manually reviewed.
#
# The application preserves:
#   - current first-stage classifications
#   - full first-stage history
#   - current second-stage classifications
#   - full second-stage history
#   - multi-user claims
#   - spatial exports
#
# EXPECTED PROJECT STRUCTURE
# ------------------------------------------------------------
#
# mapping-waste-urban-africa/
# |
# +-- data/
# |   +-- optical.tiff                  # not read by this app
# |
# |   +-- predictions/
# |   |   +-- waste_piles_vector/waste_piles.shp
# |   |
# |   +-- annotations/
# |       +-- waste_review_app/
#         +-- app.R
#         +-- review_log.sqlite         # created automatically
#         +-- data/
#         |   +-- chips.csv
#         |   +-- object_chip_map.csv
#         |   +-- review_grid.gpkg
#         |   +-- second_review_candidates.csv  # created
#         |
#         +-- www/
#         |   +-- chips/
#         |       +-- C00001.png
#         |       +-- C00002.png
#         |       +-- ...
#         |
#         +-- exports/
#             +-- waste_review_export.gpkg
#             +-- first_review_results.csv
#             +-- second_review_results.csv
#             +-- review_export_summary.csv
#
# chips.csv requires at least:
#   chip_id, image_file
#
# object_chip_map.csv requires:
#   candidate_id, chip_id
#
# ============================================================


# ============================================================
# 0. PACKAGES
# ============================================================

required_packages <- c(
  "shiny",
  "DBI",
  "RSQLite",
  "sf",
  "data.table",
  "zip"
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
  "data",
  "annotations",
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

SECOND_CANDIDATE_CSV <- file.path(
  DATA_DIR,
  "second_review_candidates.csv"
)


THIRD_CANDIDATE_CSV <- file.path(
  DATA_DIR,
  "third_review_candidates.csv"
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

FIRST_REVIEW_CSV <- file.path(
  EXPORT_DIR,
  "first_review_results.csv"
)

SECOND_REVIEW_CSV <- file.path(
  EXPORT_DIR,
  "second_review_results.csv"
)


THIRD_REVIEW_CSV <- file.path(
  EXPORT_DIR,
  "third_review_results.csv"
)

EXPORT_SUMMARY_CSV <- file.path(
  EXPORT_DIR,
  "review_export_summary.csv"
)


FINAL_PILE_SUMMARY_CSV <- file.path(
  EXPORT_DIR,
  "final_verified_waste_summary.csv"
)

FINAL_SIZE_CLASS_CSV <- file.path(
  EXPORT_DIR,
  "final_verified_waste_size_classes.csv"
)


FINAL_AUDIT_CSV <- file.path(
  EXPORT_DIR,
  "final_inventory_audit.csv"
)

FINAL_SHP_DIR <- file.path(
  EXPORT_DIR,
  "final_verified_waste_shapefile"
)

FINAL_SHP_ZIP <- file.path(
  EXPORT_DIR,
  "final_verified_waste_shapefile.zip"
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

# Stage 2 uses only one look-ahead image to reduce browser/network load.
N_PRELOAD_SECOND <- 5

STATS_REFRESH_MS <- 5000

# Same minimum object area used in the targeted error assessment.
MIN_SECOND_REVIEW_AREA_M2 <- 0.25

# Normalized coordinate space for the SVG target overlay.
SVG_SIZE <- 1000


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
# 4. SERVE FIRST-STAGE PNG IMAGES
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
# 5. READ PREPARED FIRST-STAGE METADATA
# ============================================================

chips <- read.csv(
  CHIP_METADATA_PATH,
  stringsAsFactors = FALSE
)

object_chip_map <- read.csv(
  OBJECT_CHIP_MAP_PATH,
  stringsAsFactors = FALSE
)

chips$chip_id <- as.character(
  chips$chip_id
)

object_chip_map$chip_id <- as.character(
  object_chip_map$chip_id
)

object_chip_map$candidate_id <- as.character(
  object_chip_map$candidate_id
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
# Confirm image files exist
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
    "Some image files listed in chips.csv do not exist.\n",
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
# Add object counts if not already present
# ------------------------------------------------------------

if (!"n_objects" %in% names(chips)) {

  chip_object_counts <- table(
    object_chip_map$chip_id
  )

  chips$n_objects <- as.integer(
    chip_object_counts[
      match(
        chips$chip_id,
        names(chip_object_counts)
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

TOTAL_RAW_PREDICTIONS <- length(
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
  "TWO-STAGE WASTE REVIEW APP\n",
  "============================================\n",
  "App folder: ", APP_DIR, "\n",
  "Chip folder: ", CHIPS_DIR, "\n",
  "Stage-1 review locations: ",
  format(TOTAL_CHIPS, big.mark = ","),
  "\n",
  "Mapped raw candidate IDs: ",
  format(TOTAL_RAW_PREDICTIONS, big.mark = ","),
  "\n",
  "Stage-2 minimum area: ",
  MIN_SECOND_REVIEW_AREA_M2,
  " m2\n",
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


  # ----------------------------------------------------------
  # Stage 1 current classifications
  # ----------------------------------------------------------

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


  # ----------------------------------------------------------
  # Stage 1 history
  # ----------------------------------------------------------

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


  # ----------------------------------------------------------
  # Stage 1 claims
  # ----------------------------------------------------------

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


  # ----------------------------------------------------------
  # Stage 2 current classifications
  # ----------------------------------------------------------

  DBI::dbExecute(
    con,
    "
    CREATE TABLE IF NOT EXISTS second_review (
      candidate_id TEXT PRIMARY KEY,
      label        TEXT NOT NULL,
      annotator    TEXT NOT NULL,
      notes        TEXT,
      timestamp    TEXT NOT NULL
    );
    "
  )


  # ----------------------------------------------------------
  # Stage 2 history
  # ----------------------------------------------------------

  DBI::dbExecute(
    con,
    "
    CREATE TABLE IF NOT EXISTS second_review_history (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      candidate_id TEXT NOT NULL,
      label        TEXT NOT NULL,
      annotator    TEXT NOT NULL,
      notes        TEXT,
      timestamp    TEXT NOT NULL
    );
    "
  )


  # ----------------------------------------------------------
  # Stage 2 claims
  # ----------------------------------------------------------

  DBI::dbExecute(
    con,
    "
    CREATE TABLE IF NOT EXISTS second_claims (
      candidate_id TEXT PRIMARY KEY,
      annotator    TEXT NOT NULL,
      claimed_at   TEXT NOT NULL
    );
    "
  )


  # ----------------------------------------------------------
  # Stage 3 current classifications
  # ----------------------------------------------------------

  DBI::dbExecute(
    con,
    "
    CREATE TABLE IF NOT EXISTS third_review (
      candidate_id TEXT PRIMARY KEY,
      label        TEXT NOT NULL,
      annotator    TEXT NOT NULL,
      notes        TEXT,
      timestamp    TEXT NOT NULL
    );
    "
  )


  # ----------------------------------------------------------
  # Stage 3 history
  # ----------------------------------------------------------

  DBI::dbExecute(
    con,
    "
    CREATE TABLE IF NOT EXISTS third_review_history (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      candidate_id TEXT NOT NULL,
      label        TEXT NOT NULL,
      annotator    TEXT NOT NULL,
      notes        TEXT,
      timestamp    TEXT NOT NULL
    );
    "
  )


  # ----------------------------------------------------------
  # Stage 3 claims
  # ----------------------------------------------------------

  DBI::dbExecute(
    con,
    "
    CREATE TABLE IF NOT EXISTS third_claims (
      candidate_id TEXT PRIMARY KEY,
      annotator    TEXT NOT NULL,
      claimed_at   TEXT NOT NULL
    );
    "
  )


  # ----------------------------------------------------------
  # Indices
  # ----------------------------------------------------------

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
    CREATE INDEX IF NOT EXISTS idx_second_label
    ON second_review(label);
    "
  )

  DBI::dbExecute(
    con,
    "
    CREATE INDEX IF NOT EXISTS idx_second_annotator
    ON second_review(annotator);
    "
  )

  DBI::dbExecute(
    con,
    "
    CREATE INDEX IF NOT EXISTS idx_third_label
    ON third_review(label);
    "
  )

  DBI::dbExecute(
    con,
    "
    CREATE INDEX IF NOT EXISTS idx_third_annotator
    ON third_review(annotator);
    "
  )
}


db_init()


# ============================================================
# 8. STAGE 1 DATABASE HELPERS
# ============================================================

release_expired_first_claims <- function(
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


touch_first_claim <- function(
  chip_id,
  annotator
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


release_first_claim <- function(
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


get_first_annotations <- function() {

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


get_first_annotation <- function(
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


get_first_unavailable <- function(
  con
) {

  release_expired_first_claims(
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


claim_next_first <- function(
  annotator,
  exclude_ids = character(0)
) {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  unavailable <- get_first_unavailable(
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


peek_next_first <- function(
  n = 3,
  exclude_ids = character(0)
) {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  unavailable <- get_first_unavailable(
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


save_first_annotation <- function(
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
        "mixed",
        "not_sure"
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


get_first_stats <- function() {

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
        CASE WHEN label = 'waste'
        THEN 1 ELSE 0 END
      ) AS waste,

      SUM(
        CASE WHEN label = 'not_waste'
        THEN 1 ELSE 0 END
      ) AS not_waste,

      SUM(
        CASE WHEN label = 'mixed'
        THEN 1 ELSE 0 END
      ) AS mixed,

      SUM(
        CASE WHEN label = 'not_sure'
        THEN 1 ELSE 0 END
      ) AS not_sure

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
      "mixed",
      "not_sure"
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
    reviewed = as.integer(
      counts$reviewed[1]
    ),

    waste = as.integer(
      counts$waste[1]
    ),

    not_waste = as.integer(
      counts$not_waste[1]
    ),

    mixed = as.integer(
      counts$mixed[1]
    ),

    not_sure = as.integer(
      counts$not_sure[1]
    ),

    board = board
  )
}


get_first_reviewer_history <- function(
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


claim_first_reviewed <- function(
  chip_id,
  annotator
) {

  annotation <- get_first_annotation(
    chip_id
  )


  if (is.null(annotation)) {

    return(
      list(
        ok = FALSE,
        message = "This first-stage location has not been reviewed."
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
          "This location is currently assigned to ",
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


  release_expired_first_claims(
    con
  )


  existing <- DBI::dbGetQuery(
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


  if (
    nrow(existing) > 0 &&
    existing$annotator[1] != annotator
  ) {

    return(
      list(
        ok = FALSE,
        message = "This location is open in another review session."
      )
    )
  }


  if (nrow(existing) == 0) {

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
          message = "Could not open this location for correction."
        )
      )
    }

  } else {

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
  }


  list(
    ok = TRUE,
    annotation = annotation
  )
}


# ============================================================
# 9. STAGE 2 DATABASE HELPERS
# ============================================================

release_expired_second_claims <- function(
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
    DELETE FROM second_claims
    WHERE claimed_at < ?;
    ",
    params = list(
      cutoff
    )
  )
}


touch_second_claim <- function(
  candidate_id,
  annotator
) {

  if (
    is.null(candidate_id) ||
    !nzchar(candidate_id)
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
    UPDATE second_claims
    SET claimed_at = ?
    WHERE candidate_id = ?
      AND annotator = ?;
    ",
    params = list(
      format(
        Sys.time(),
        "%Y-%m-%d %H:%M:%S"
      ),
      candidate_id,
      annotator
    )
  )


  invisible(NULL)
}


release_second_claim <- function(
  candidate_id,
  annotator = NULL
) {

  if (
    is.null(candidate_id) ||
    !nzchar(candidate_id)
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
      DELETE FROM second_claims
      WHERE candidate_id = ?;
      ",
      params = list(
        candidate_id
      )
    )

  } else {

    DBI::dbExecute(
      con,
      "
      DELETE FROM second_claims
      WHERE candidate_id = ?
        AND annotator = ?;
      ",
      params = list(
        candidate_id,
        annotator
      )
    )
  }


  invisible(NULL)
}


get_second_reviews <- function() {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  DBI::dbGetQuery(
    con,
    "
    SELECT
      candidate_id,
      label,
      annotator,
      notes,
      timestamp
    FROM second_review;
    "
  )
}


get_second_review <- function(
  candidate_id
) {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  result <- DBI::dbGetQuery(
    con,
    "
    SELECT
      candidate_id,
      label,
      annotator,
      notes,
      timestamp
    FROM second_review
    WHERE candidate_id = ?;
    ",
    params = list(
      candidate_id
    )
  )


  if (nrow(result) == 0) {
    return(NULL)
  }


  result
}


claim_next_second <- function(
  annotator,
  active_ids,
  exclude_ids = character(0)
) {

  active_ids <- as.character(
    active_ids
  )


  if (length(active_ids) == 0) {
    return(NULL)
  }


  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  release_expired_second_claims(
    con
  )


  reviewed <- DBI::dbGetQuery(
    con,
    "
    SELECT candidate_id
    FROM second_review;
    "
  )$candidate_id


  claimed <- DBI::dbGetQuery(
    con,
    "
    SELECT candidate_id
    FROM second_claims;
    "
  )$candidate_id


  unavailable <- unique(
    c(
      reviewed,
      claimed,
      exclude_ids
    )
  )


  available <- active_ids[
    !active_ids %in%
      unavailable
  ]


  if (length(available) == 0) {
    return(NULL)
  }


  for (candidate_id in available) {

    inserted <- DBI::dbExecute(
      con,
      "
      INSERT OR IGNORE INTO second_claims (
        candidate_id,
        annotator,
        claimed_at
      )
      VALUES (?, ?, ?);
      ",
      params = list(
        candidate_id,
        annotator,
        format(
          Sys.time(),
          "%Y-%m-%d %H:%M:%S"
        )
      )
    )


    if (inserted == 1) {
      return(
        candidate_id
      )
    }
  }


  NULL
}


peek_next_second <- function(
  active_ids,
  n = 3,
  exclude_ids = character(0)
) {

  active_ids <- as.character(
    active_ids
  )


  if (length(active_ids) == 0) {
    return(
      character(0)
    )
  }


  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  release_expired_second_claims(
    con
  )


  reviewed <- DBI::dbGetQuery(
    con,
    "
    SELECT candidate_id
    FROM second_review;
    "
  )$candidate_id


  claimed <- DBI::dbGetQuery(
    con,
    "
    SELECT candidate_id
    FROM second_claims;
    "
  )$candidate_id


  unavailable <- unique(
    c(
      reviewed,
      claimed,
      exclude_ids
    )
  )


  available <- active_ids[
    !active_ids %in%
      unavailable
  ]


  head(
    available,
    n
  )
}


save_second_review <- function(
  candidate_id,
  label,
  annotator,
  notes = ""
) {

  stopifnot(
    label %in%
      c(
        "waste",
        "not_waste",
        "not_sure"
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

      DBI::dbExecute(
        con,
        "
        INSERT INTO second_review_history (
          candidate_id,
          label,
          annotator,
          notes,
          timestamp
        )
        VALUES (?, ?, ?, ?, ?);
        ",
        params = list(
          candidate_id,
          label,
          annotator,
          notes,
          now
        )
      )


      DBI::dbExecute(
        con,
        "
        INSERT INTO second_review (
          candidate_id,
          label,
          annotator,
          notes,
          timestamp
        )
        VALUES (?, ?, ?, ?, ?)

        ON CONFLICT(candidate_id)
        DO UPDATE SET
          label = excluded.label,
          annotator = excluded.annotator,
          notes = excluded.notes,
          timestamp = excluded.timestamp;
        ",
        params = list(
          candidate_id,
          label,
          annotator,
          notes,
          now
        )
      )


      DBI::dbExecute(
        con,
        "
        DELETE FROM second_claims
        WHERE candidate_id = ?;
        ",
        params = list(
          candidate_id
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


get_second_stats <- function(
  active_ids
) {

  active_ids <- as.character(
    active_ids
  )


  reviews <- get_second_reviews()


  if (length(active_ids) == 0) {

    active_reviews <- reviews[
      FALSE,
      ,
      drop = FALSE
    ]

  } else {

    active_reviews <- reviews[
      reviews$candidate_id %in%
        active_ids,
      ,
      drop = FALSE
    ]
  }


  counts <- table(
    factor(
      active_reviews$label,
      levels = c(
        "waste",
        "not_waste",
        "not_sure"
      )
    )
  )


  list(

    total = length(
      active_ids
    ),

    reviewed = nrow(
      active_reviews
    ),

    waste = as.integer(
      counts["waste"]
    ),

    not_waste = as.integer(
      counts["not_waste"]
    ),

    not_sure = as.integer(
      counts["not_sure"]
    )
  )
}



get_second_unresolved_ids <- function(
  active_ids
) {

  reviews <- get_second_reviews()

  unresolved <- reviews$candidate_id[
    reviews$label == "not_sure"
  ]

  active_ids[
    active_ids %in%
      unresolved
  ]
}


claim_second_for_resolution <- function(
  candidate_id,
  annotator,
  active_ids
) {

  candidate_id <- as.character(
    candidate_id
  )

  active_ids <- as.character(
    active_ids
  )


  if (
    !candidate_id %in%
      active_ids
  ) {

    return(
      list(
        ok = FALSE,
        message = "This object is no longer in the active Stage-2 pool."
      )
    )
  }


  review <- get_second_review(
    candidate_id
  )


  if (
    is.null(review) ||
    review$label[1] != "not_sure"
  ) {

    return(
      list(
        ok = FALSE,
        message = "This object is no longer classified as Not sure."
      )
    )
  }


  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  release_expired_second_claims(
    con
  )


  existing <- DBI::dbGetQuery(
    con,
    "
    SELECT
      candidate_id,
      annotator
    FROM second_claims
    WHERE candidate_id = ?;
    ",
    params = list(
      candidate_id
    )
  )


  if (
    nrow(existing) > 0 &&
    existing$annotator[1] !=
      annotator
  ) {

    return(
      list(
        ok = FALSE,
        message = "This unresolved object is open in another review session."
      )
    )
  }


  if (nrow(existing) == 0) {

    inserted <- DBI::dbExecute(
      con,
      "
      INSERT OR IGNORE INTO second_claims (
        candidate_id,
        annotator,
        claimed_at
      )
      VALUES (?, ?, ?);
      ",
      params = list(
        candidate_id,
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
          message = "Could not claim this unresolved object."
        )
      )
    }

  } else {

    DBI::dbExecute(
      con,
      "
      UPDATE second_claims
      SET claimed_at = ?
      WHERE candidate_id = ?
        AND annotator = ?;
      ",
      params = list(
        format(
          Sys.time(),
          "%Y-%m-%d %H:%M:%S"
        ),
        candidate_id,
        annotator
      )
    )
  }


  list(
    ok = TRUE,
    review = review
  )
}


peek_next_unresolved <- function(
  active_ids,
  n = 3,
  exclude_ids = character(0)
) {

  active_ids <- as.character(
    active_ids
  )

  unresolved <- get_second_unresolved_ids(
    active_ids
  )

  unresolved <- unresolved[
    !unresolved %in%
      as.character(
        exclude_ids
      )
  ]

  head(
    unresolved,
    n
  )
}


get_second_reviewer_history <- function(
  annotator,
  active_ids
) {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  result <- DBI::dbGetQuery(
    con,
    "
    SELECT
      candidate_id,
      label,
      annotator,
      notes,
      timestamp
    FROM second_review
    WHERE annotator = ?
    ORDER BY timestamp DESC;
    ",
    params = list(
      annotator
    )
  )


  result[
    result$candidate_id %in%
      as.character(
        active_ids
      ),
    ,
    drop = FALSE
  ]
}


claim_second_reviewed <- function(
  candidate_id,
  annotator,
  active_ids
) {

  if (
    !candidate_id %in%
      as.character(active_ids)
  ) {

    return(
      list(
        ok = FALSE,
        message = "This object is no longer in the current second-review pool."
      )
    )
  }


  review <- get_second_review(
    candidate_id
  )


  if (is.null(review)) {

    return(
      list(
        ok = FALSE,
        message = "This object has not yet been reviewed in stage 2."
      )
    )
  }


  if (
    review$annotator[1] !=
      annotator
  ) {

    return(
      list(
        ok = FALSE,
        message = paste0(
          "This object is currently assigned to ",
          review$annotator[1],
          ". You can only correct your own current reviews."
        )
      )
    )
  }


  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  release_expired_second_claims(
    con
  )


  existing <- DBI::dbGetQuery(
    con,
    "
    SELECT
      candidate_id,
      annotator
    FROM second_claims
    WHERE candidate_id = ?;
    ",
    params = list(
      candidate_id
    )
  )


  if (
    nrow(existing) > 0 &&
    existing$annotator[1] != annotator
  ) {

    return(
      list(
        ok = FALSE,
        message = "This object is open in another review session."
      )
    )
  }


  if (nrow(existing) == 0) {

    inserted <- DBI::dbExecute(
      con,
      "
      INSERT OR IGNORE INTO second_claims (
        candidate_id,
        annotator,
        claimed_at
      )
      VALUES (?, ?, ?);
      ",
      params = list(
        candidate_id,
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
          message = "Could not open this second-stage object for correction."
        )
      )
    }

  } else {

    DBI::dbExecute(
      con,
      "
      UPDATE second_claims
      SET claimed_at = ?
      WHERE candidate_id = ?
        AND annotator = ?;
      ",
      params = list(
        format(
          Sys.time(),
          "%Y-%m-%d %H:%M:%S"
        ),
        candidate_id,
        annotator
      )
    )
  }


  list(
    ok = TRUE,
    review = review
  )
}



# ============================================================
# 10. STAGE 3 DATABASE HELPERS
# ============================================================

release_expired_third_claims <- function(
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
    DELETE FROM third_claims
    WHERE claimed_at < ?;
    ",
    params = list(
      cutoff
    )
  )
}


touch_third_claim <- function(
  candidate_id,
  annotator
) {

  if (
    is.null(candidate_id) ||
    !nzchar(candidate_id)
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
    UPDATE third_claims
    SET claimed_at = ?
    WHERE candidate_id = ?
      AND annotator = ?;
    ",
    params = list(
      format(
        Sys.time(),
        "%Y-%m-%d %H:%M:%S"
      ),
      candidate_id,
      annotator
    )
  )


  invisible(NULL)
}


release_third_claim <- function(
  candidate_id,
  annotator = NULL
) {

  if (
    is.null(candidate_id) ||
    !nzchar(candidate_id)
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
      DELETE FROM third_claims
      WHERE candidate_id = ?;
      ",
      params = list(
        candidate_id
      )
    )

  } else {

    DBI::dbExecute(
      con,
      "
      DELETE FROM third_claims
      WHERE candidate_id = ?
        AND annotator = ?;
      ",
      params = list(
        candidate_id,
        annotator
      )
    )
  }


  invisible(NULL)
}


get_third_reviews <- function() {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  DBI::dbGetQuery(
    con,
    "
    SELECT
      candidate_id,
      label,
      annotator,
      notes,
      timestamp
    FROM third_review;
    "
  )
}


get_third_review <- function(
  candidate_id
) {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  result <- DBI::dbGetQuery(
    con,
    "
    SELECT
      candidate_id,
      label,
      annotator,
      notes,
      timestamp
    FROM third_review
    WHERE candidate_id = ?;
    ",
    params = list(
      candidate_id
    )
  )


  if (nrow(result) == 0) {
    return(NULL)
  }


  result
}


claim_next_third <- function(
  annotator,
  active_ids,
  exclude_ids = character(0)
) {

  active_ids <- as.character(
    active_ids
  )


  if (length(active_ids) == 0) {
    return(NULL)
  }


  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  release_expired_third_claims(
    con
  )


  reviewed <- DBI::dbGetQuery(
    con,
    "
    SELECT candidate_id
    FROM third_review;
    "
  )$candidate_id


  claimed <- DBI::dbGetQuery(
    con,
    "
    SELECT candidate_id
    FROM third_claims;
    "
  )$candidate_id


  unavailable <- unique(
    c(
      reviewed,
      claimed,
      exclude_ids
    )
  )


  available <- active_ids[
    !active_ids %in%
      unavailable
  ]


  if (length(available) == 0) {
    return(NULL)
  }


  for (candidate_id in available) {

    inserted <- DBI::dbExecute(
      con,
      "
      INSERT OR IGNORE INTO third_claims (
        candidate_id,
        annotator,
        claimed_at
      )
      VALUES (?, ?, ?);
      ",
      params = list(
        candidate_id,
        annotator,
        format(
          Sys.time(),
          "%Y-%m-%d %H:%M:%S"
        )
      )
    )


    if (inserted == 1) {
      return(
        candidate_id
      )
    }
  }


  NULL
}


peek_next_third <- function(
  active_ids,
  n = 5,
  exclude_ids = character(0)
) {

  active_ids <- as.character(
    active_ids
  )


  if (length(active_ids) == 0) {
    return(
      character(0)
    )
  }


  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  release_expired_third_claims(
    con
  )


  reviewed <- DBI::dbGetQuery(
    con,
    "
    SELECT candidate_id
    FROM third_review;
    "
  )$candidate_id


  claimed <- DBI::dbGetQuery(
    con,
    "
    SELECT candidate_id
    FROM third_claims;
    "
  )$candidate_id


  unavailable <- unique(
    c(
      reviewed,
      claimed,
      exclude_ids
    )
  )


  available <- active_ids[
    !active_ids %in%
      unavailable
  ]


  head(
    available,
    n
  )
}


save_third_review <- function(
  candidate_id,
  label,
  annotator,
  notes = ""
) {

  stopifnot(
    label %in%
      c(
        "keep",
        "delete",
        "needs_edit",
        "not_sure"
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

      DBI::dbExecute(
        con,
        "
        INSERT INTO third_review_history (
          candidate_id,
          label,
          annotator,
          notes,
          timestamp
        )
        VALUES (?, ?, ?, ?, ?);
        ",
        params = list(
          candidate_id,
          label,
          annotator,
          notes,
          now
        )
      )


      DBI::dbExecute(
        con,
        "
        INSERT INTO third_review (
          candidate_id,
          label,
          annotator,
          notes,
          timestamp
        )
        VALUES (?, ?, ?, ?, ?)

        ON CONFLICT(candidate_id)
        DO UPDATE SET
          label = excluded.label,
          annotator = excluded.annotator,
          notes = excluded.notes,
          timestamp = excluded.timestamp;
        ",
        params = list(
          candidate_id,
          label,
          annotator,
          notes,
          now
        )
      )


      DBI::dbExecute(
        con,
        "
        DELETE FROM third_claims
        WHERE candidate_id = ?;
        ",
        params = list(
          candidate_id
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


get_third_stats <- function(
  active_ids
) {

  active_ids <- as.character(
    active_ids
  )


  reviews <- get_third_reviews()


  active_reviews <- reviews[
    reviews$candidate_id %in%
      active_ids,
    ,
    drop = FALSE
  ]


  counts <- table(
    factor(
      active_reviews$label,
      levels = c(
        "keep",
        "delete",
        "needs_edit",
        "not_sure"
      )
    )
  )


  list(
    total = length(
      active_ids
    ),

    reviewed = nrow(
      active_reviews
    ),

    keep = as.integer(
      counts["keep"]
    ),

    delete = as.integer(
      counts["delete"]
    ),

    needs_edit = as.integer(
      counts["needs_edit"]
    ),

    not_sure = as.integer(
      counts["not_sure"]
    )
  )
}


get_third_reviewer_history <- function(
  annotator,
  active_ids
) {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  result <- DBI::dbGetQuery(
    con,
    "
    SELECT
      candidate_id,
      label,
      annotator,
      notes,
      timestamp
    FROM third_review
    WHERE annotator = ?
    ORDER BY timestamp DESC;
    ",
    params = list(
      annotator
    )
  )


  result[
    result$candidate_id %in%
      as.character(
        active_ids
      ),
    ,
    drop = FALSE
  ]
}


claim_third_reviewed <- function(
  candidate_id,
  annotator,
  active_ids
) {

  if (
    !candidate_id %in%
      as.character(
        active_ids
      )
  ) {

    return(
      list(
        ok = FALSE,
        message = "This object is no longer in the current Stage-3 pool."
      )
    )
  }


  review <- get_third_review(
    candidate_id
  )


  if (is.null(review)) {

    return(
      list(
        ok = FALSE,
        message = "This object has not yet been reviewed in Stage 3."
      )
    )
  }


  if (
    review$annotator[1] !=
      annotator
  ) {

    return(
      list(
        ok = FALSE,
        message = paste0(
          "This Stage-3 object is currently assigned to ",
          review$annotator[1],
          ". You can only correct your own current reviews."
        )
      )
    )
  }


  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  release_expired_third_claims(
    con
  )


  existing <- DBI::dbGetQuery(
    con,
    "
    SELECT
      candidate_id,
      annotator
    FROM third_claims
    WHERE candidate_id = ?;
    ",
    params = list(
      candidate_id
    )
  )


  if (
    nrow(existing) > 0 &&
    existing$annotator[1] !=
      annotator
  ) {

    return(
      list(
        ok = FALSE,
        message = "This Stage-3 object is open in another review session."
      )
    )
  }


  if (nrow(existing) == 0) {

    inserted <- DBI::dbExecute(
      con,
      "
      INSERT OR IGNORE INTO third_claims (
        candidate_id,
        annotator,
        claimed_at
      )
      VALUES (?, ?, ?);
      ",
      params = list(
        candidate_id,
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
          message = "Could not open this Stage-3 object for correction."
        )
      )
    }

  } else {

    DBI::dbExecute(
      con,
      "
      UPDATE third_claims
      SET claimed_at = ?
      WHERE candidate_id = ?
        AND annotator = ?;
      ",
      params = list(
        format(
          Sys.time(),
          "%Y-%m-%d %H:%M:%S"
        ),
        candidate_id,
        annotator
      )
    )
  }


  list(
    ok = TRUE,
    review = review
  )
}


# ============================================================
# 11. BUILD CURRENT SPATIAL STATE
# ============================================================

classify_stage1_statuses <- function(
  x
) {

  x <- unique(
    as.character(x)
  )


  if ("unreviewed" %in% x) {
    return(
      "pending"
    )
  }


  if (
    any(
      x %in%
        c(
          "mixed",
          "not_sure"
        )
    )
  ) {
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


  # For example an object crossing one "waste" location
  # and one "not_waste" location.
  "second_review"
}


build_object_status_table <- function(
  review_grid
) {

  status_lookup <- setNames(
    review_grid$review_class,
    review_grid$chip_id
  )


  dt <- data.table::as.data.table(
    object_chip_map[
      ,
      c(
        "candidate_id",
        "chip_id"
      )
    ]
  )


  dt <- unique(
    dt
  )


  dt[
    ,
    status :=
      status_lookup[
        chip_id
      ]
  ]


  dt[
    is.na(status),
    status := "unreviewed"
  ]


  object_status <- dt[
    ,
    {

      statuses <- sort(
        unique(
          status
        )
      )


      second_rows <- which(
        status %in%
          c(
            "mixed",
            "not_sure"
          )
      )


      representative_chip <- if (
        length(second_rows) > 0
      ) {

        chip_id[
          second_rows[1]
        ]

      } else {

        chip_id[1]
      }


      list(

        stage1_class =
          classify_stage1_statuses(
            statuses
          ),

        stage1_reason =
          paste(
            statuses,
            collapse = " + "
          ),

        representative_chip_id =
          representative_chip
      )
    },
    by = candidate_id
  ]


  as.data.frame(
    object_status,
    stringsAsFactors = FALSE
  )
}


build_spatial_state <- function() {

  annotations <- get_first_annotations()


  # ----------------------------------------------------------
  # Review grid
  # ----------------------------------------------------------

  review_grid <- sf::st_read(
    REVIEW_GRID_PATH,
    quiet = TRUE
  )


  review_grid$chip_id <- as.character(
    review_grid$chip_id
  )


  if (!"chip_id" %in% names(review_grid)) {
    stop(
      "review_grid.gpkg does not contain a chip_id field."
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
  # Candidate status from all intersecting first-stage cells
  # ----------------------------------------------------------

  object_status <- build_object_status_table(
    review_grid
  )


  object_status$candidate_id <- as.character(
    object_status$candidate_id
  )


  # ----------------------------------------------------------
  # Prediction polygons
  # ----------------------------------------------------------

  predictions <- sf::st_read(
    PREDICTION_PATH,
    quiet = TRUE
  )


  if (!"candidate_id" %in% names(predictions)) {

    # This matches the preparation workflow when candidate_id
    # was generated from row order.
    predictions$candidate_id <- seq_len(
      nrow(predictions)
    )
  }


  predictions$candidate_id <- as.character(
    predictions$candidate_id
  )


  if (
    !isTRUE(
      sf::st_crs(predictions) ==
        sf::st_crs(review_grid)
    )
  ) {

    predictions <- sf::st_transform(
      predictions,
      sf::st_crs(review_grid)
    )
  }


  j <- match(
    predictions$candidate_id,
    object_status$candidate_id
  )


  predictions$stage1_class <- ifelse(
    is.na(j),
    "pending",
    object_status$stage1_class[j]
  )


  predictions$stage1_reason <- ifelse(
    is.na(j),
    "not_mapped_to_review_grid",
    object_status$stage1_reason[j]
  )


  predictions$representative_chip_id <- ifelse(
    is.na(j),
    NA_character_,
    object_status$representative_chip_id[j]
  )


  predictions$area_m2 <- as.numeric(
    sf::st_area(
      predictions
    )
  )


  list(
    review_grid = review_grid,
    predictions = predictions,
    object_status = object_status
  )
}


# ============================================================
# 12. PREPARE / REFRESH SECOND-REVIEW POOL
# ============================================================

prepare_second_review_pool <- function() {

  first_stats <- get_first_stats()


  if (
    first_stats$reviewed <
      TOTAL_CHIPS
  ) {

    stop(
      paste0(
        "First review is not complete. ",
        TOTAL_CHIPS -
          first_stats$reviewed,
        " locations remain."
      )
    )
  }


  spatial_state <- build_spatial_state()


  review_grid <- spatial_state$review_grid

  predictions <- spatial_state$predictions


  # ----------------------------------------------------------
  # Only ambiguous objects meeting minimum area
  # ----------------------------------------------------------

  keep <- (
    predictions$stage1_class ==
      "second_review"
  ) &
    (
      predictions$area_m2 >=
        MIN_SECOND_REVIEW_AREA_M2
    ) &
    !is.na(
      predictions$representative_chip_id
    )


  second_sf <- predictions[
    keep,
    ,
    drop = FALSE
  ]


  # ----------------------------------------------------------
  # Prepare the Stage-2 display ONCE.
  #
  # This chooses a review cell that actually contains each object
  # and pre-calculates CSS bounding-box coordinates. Nothing
  # spatial is recalculated while the reviewer moves through items.
  # ----------------------------------------------------------

  second_sf <- add_stage2_display_metadata(
    second_sf,
    review_grid
  )


  # Review objects from the same 50 m image consecutively.
  # This avoids reloading a different 1400 px PNG after every click
  # and makes Stage 2 behave much more like the fast Stage-1 review.
  candidate_num <- suppressWarnings(
    as.numeric(
      as.character(
        second_sf$candidate_id
      )
    )
  )

  candidate_num[
    is.na(candidate_num)
  ] <- Inf

  second_order <- order(
    second_sf$display_chip_id,
    candidate_num,
    as.character(second_sf$candidate_id)
  )

  second_sf <- second_sf[
    second_order,
    ,
    drop = FALSE
  ]


  missing_base <- is.na(
    second_sf$base_image_file
  )


  if (any(missing_base)) {

    stop(
      paste0(
        sum(missing_base),
        " second-review objects could not be linked to a first-stage PNG."
      )
    )
  }


  second_meta <- sf::st_drop_geometry(
    second_sf
  )


  keep_cols <- intersect(
    c(
      "candidate_id",
      "area_m2",
      "stage1_class",
      "stage1_reason",
      "representative_chip_id",
      "display_chip_id",
      "base_image_file",
      "box_left_pct",
      "box_top_pct",
      "box_width_pct",
      "box_height_pct"
    ),
    names(
      second_meta
    )
  )


  second_meta <- second_meta[
    ,
    keep_cols,
    drop = FALSE
  ]


  write.csv(
    second_meta,
    SECOND_CANDIDATE_CSV,
    row.names = FALSE
  )


  list(
    review_grid = review_grid,
    second_sf = second_sf,
    second_meta = second_meta,
    spatial_state = spatial_state
  )
}


# ============================================================
# 13. STAGE-2 / STAGE-3 DISPLAY METADATA
# ============================================================
#
# Stage 2 uses the existing 50 m PNGs. To keep object-to-object
# navigation fast, the display chip and bounding-box coordinates
# are calculated ONCE when the Stage-2 pool is prepared.
#
# During review the browser only draws a CSS rectangle. No sf
# intersection, validation or geometry conversion is done when
# the reviewer clicks a classification button.
# ============================================================

choose_stage2_display_chip <- function(
  second_sf,
  review_grid
) {

  if (nrow(second_sf) == 0) {
    return(integer(0))
  }


  # A point guaranteed to lie on/in each polygon is used first.
  # This normally selects the review cell containing the main
  # body of the object rather than a neighbouring cell touched
  # only by its edge.
  target_points <- suppressWarnings(
    sf::st_point_on_surface(
      second_sf
    )
  )


  point_hits <- sf::st_intersects(
    target_points,
    review_grid
  )


  polygon_hits <- sf::st_intersects(
    second_sf,
    review_grid
  )


  result <- integer(
    nrow(second_sf)
  )


  for (i in seq_len(nrow(second_sf))) {

    candidates <- point_hits[[i]]


    if (length(candidates) == 0) {
      candidates <- polygon_hits[[i]]
    }


    if (length(candidates) == 0) {

      # Defensive fallback. This should rarely be needed, but it
      # guarantees that every Stage-2 object has a display image.
      candidates <- sf::st_nearest_feature(
        target_points[i, ],
        review_grid
      )
    }


    # Prefer a cell that itself was marked Mixed / Not sure when
    # one is available. Conflict objects may instead be linked to
    # Waste + Not-waste cells and will use the first intersecting
    # cell containing the target point.
    ambiguous_candidates <- candidates[
      review_grid$review_class[candidates] %in%
        c(
          "mixed",
          "not_sure"
        )
    ]


    if (length(ambiguous_candidates) > 0) {
      candidates <- ambiguous_candidates
    }


    result[i] <- candidates[1]
  }


  result
}


make_stage2_css_bbox <- function(
  target_geometry,
  chip_geometry,
  padding_pct = 0.50,
  min_box_pct = 3.0
) {

  target_bb <- sf::st_bbox(
    target_geometry
  )

  chip_bb <- sf::st_bbox(
    chip_geometry
  )


  chip_width <- as.numeric(
    chip_bb["xmax"] -
      chip_bb["xmin"]
  )

  chip_height <- as.numeric(
    chip_bb["ymax"] -
      chip_bb["ymin"]
  )


  if (
    !is.finite(chip_width) ||
    !is.finite(chip_height) ||
    chip_width <= 0 ||
    chip_height <= 0
  ) {
    return(
      c(
        left = 48.5,
        top = 48.5,
        width = 3,
        height = 3
      )
    )
  }


  # Constrain the object extent to the displayed review cell.
  xmin <- max(
    as.numeric(target_bb["xmin"]),
    as.numeric(chip_bb["xmin"])
  )

  xmax <- min(
    as.numeric(target_bb["xmax"]),
    as.numeric(chip_bb["xmax"])
  )

  ymin <- max(
    as.numeric(target_bb["ymin"]),
    as.numeric(chip_bb["ymin"])
  )

  ymax <- min(
    as.numeric(target_bb["ymax"]),
    as.numeric(chip_bb["ymax"])
  )


  # If numerical precision causes a non-overlap, locate a compact
  # box around the target centroid projected into the display cell.
  if (
    !is.finite(xmin) ||
    !is.finite(xmax) ||
    !is.finite(ymin) ||
    !is.finite(ymax) ||
    xmax <= xmin ||
    ymax <= ymin
  ) {

    cx_map <- mean(
      c(
        as.numeric(target_bb["xmin"]),
        as.numeric(target_bb["xmax"])
      )
    )

    cy_map <- mean(
      c(
        as.numeric(target_bb["ymin"]),
        as.numeric(target_bb["ymax"])
      )
    )


    cx <- 100 *
      (
        cx_map -
          as.numeric(chip_bb["xmin"])
      ) /
      chip_width

    cy <- 100 *
      (
        as.numeric(chip_bb["ymax"]) -
          cy_map
      ) /
      chip_height


    cx <- min(
      100 - min_box_pct / 2,
      max(
        min_box_pct / 2,
        cx
      )
    )

    cy <- min(
      100 - min_box_pct / 2,
      max(
        min_box_pct / 2,
        cy
      )
    )


    return(
      c(
        left = cx - min_box_pct / 2,
        top = cy - min_box_pct / 2,
        width = min_box_pct,
        height = min_box_pct
      )
    )
  }


  left <- 100 *
    (
      xmin -
        as.numeric(chip_bb["xmin"])
    ) /
    chip_width

  right <- 100 *
    (
      xmax -
        as.numeric(chip_bb["xmin"])
    ) /
    chip_width

  top <- 100 *
    (
      as.numeric(chip_bb["ymax"]) -
        ymax
    ) /
    chip_height

  bottom <- 100 *
    (
      as.numeric(chip_bb["ymax"]) -
        ymin
    ) /
    chip_height


  # Only a small margin is added around the actual object.
  left <- left - padding_pct
  right <- right + padding_pct
  top <- top - padding_pct
  bottom <- bottom + padding_pct


  width <- right - left
  height <- bottom - top


  # Tiny objects still need a visible locator, but the minimum is
  # deliberately only 3% of the 50 m chip (~1.5 m), much smaller
  # than the original oversized box.
  if (width < min_box_pct) {

    centre <- (left + right) / 2

    left <- centre - min_box_pct / 2
    right <- centre + min_box_pct / 2
  }


  if (height < min_box_pct) {

    centre <- (top + bottom) / 2

    top <- centre - min_box_pct / 2
    bottom <- centre + min_box_pct / 2
  }


  # Shift rather than shrink the box at the chip edge.
  if (left < 0.3) {
    shift <- 0.3 - left
    left <- left + shift
    right <- right + shift
  }

  if (right > 99.7) {
    shift <- right - 99.7
    left <- left - shift
    right <- right - shift
  }

  if (top < 0.3) {
    shift <- 0.3 - top
    top <- top + shift
    bottom <- bottom + shift
  }

  if (bottom > 99.7) {
    shift <- bottom - 99.7
    top <- top - shift
    bottom <- bottom - shift
  }


  c(
    left = left,
    top = top,
    width = right - left,
    height = bottom - top
  )
}


add_stage2_display_metadata <- function(
  second_sf,
  review_grid
) {

  if (nrow(second_sf) == 0) {

    second_sf$display_chip_id <- character(0)
    second_sf$base_image_file <- character(0)
    second_sf$box_left_pct <- numeric(0)
    second_sf$box_top_pct <- numeric(0)
    second_sf$box_width_pct <- numeric(0)
    second_sf$box_height_pct <- numeric(0)

    return(second_sf)
  }


  display_index <- choose_stage2_display_chip(
    second_sf,
    review_grid
  )


  second_sf$display_chip_id <- as.character(
    review_grid$chip_id[
      display_index
    ]
  )


  second_sf$base_image_file <- chips$image_file[
    match(
      second_sf$display_chip_id,
      chips$chip_id
    )
  ]


  boxes <- lapply(
    seq_len(nrow(second_sf)),
    function(i) {

      make_stage2_css_bbox(
        sf::st_geometry(second_sf)[i],
        sf::st_geometry(review_grid)[display_index[i]]
      )
    }
  )


  box_matrix <- do.call(
    rbind,
    boxes
  )


  second_sf$box_left_pct <- box_matrix[, "left"]
  second_sf$box_top_pct <- box_matrix[, "top"]
  second_sf$box_width_pct <- box_matrix[, "width"]
  second_sf$box_height_pct <- box_matrix[, "height"]


  second_sf
}


# ============================================================
# 14. PREPARE / REFRESH STAGE-3 FINAL POSITIVE AUDIT
# ============================================================

prepare_third_review_pool <- function() {

  spatial_state <- build_spatial_state()

  review_grid <- spatial_state$review_grid

  predictions <- spatial_state$predictions


  eligible <- predictions$area_m2 >=
    MIN_SECOND_REVIEW_AREA_M2


  active_second <- predictions[
    eligible &
      predictions$stage1_class ==
        "second_review",
    ,
    drop = FALSE
  ]


  second_reviews <- get_second_reviews()


  active_second_reviews <- second_reviews[
    second_reviews$candidate_id %in%
      active_second$candidate_id,
    ,
    drop = FALSE
  ]


  second_match <- match(
    active_second$candidate_id,
    active_second_reviews$candidate_id
  )


  active_second$second_label <- ifelse(
    is.na(second_match),
    "unreviewed",
    active_second_reviews$label[
      second_match
    ]
  )


  n_unreviewed <- sum(
    active_second$second_label ==
      "unreviewed"
  )


  n_not_sure <- sum(
    active_second$second_label ==
      "not_sure"
  )


  if (
    n_unreviewed > 0 ||
    n_not_sure > 0
  ) {

    stop(
      paste0(
        "Stage 2 must be fully resolved before Stage 3. ",
        n_unreviewed,
        " Stage-2 object(s) are unreviewed and ",
        n_not_sure,
        " are still marked Not sure."
      )
    )
  }


  # Attach current Stage-2 labels to the master prediction table.
  predictions$second_label <- NA_character_


  p_second <- match(
    predictions$candidate_id,
    active_second$candidate_id
  )


  has_second <- !is.na(
    p_second
  )


  predictions$second_label[
    has_second
  ] <- active_second$second_label[
    p_second[
      has_second
    ]
  ]


  # Positive objects entering Stage 3.
  stage1_positive <- predictions[
    eligible &
      predictions$stage1_class ==
        "retained",
    ,
    drop = FALSE
  ]


  stage1_positive$review_src <- "stage1"


  stage2_positive <- predictions[
    eligible &
      predictions$stage1_class ==
        "second_review" &
      predictions$second_label ==
        "waste",
    ,
    drop = FALSE
  ]


  stage2_positive$review_src <- "stage2"


  third_sf <- rbind(
    stage1_positive,
    stage2_positive
  )


  third_sf <- third_sf[
    !duplicated(
      third_sf$candidate_id
    ),
    ,
    drop = FALSE
  ]


  third_sf <- add_stage2_display_metadata(
    third_sf,
    review_grid
  )


  candidate_num <- suppressWarnings(
    as.numeric(
      as.character(
        third_sf$candidate_id
      )
    )
  )


  candidate_num[
    is.na(candidate_num)
  ] <- Inf


  third_order <- order(
    third_sf$display_chip_id,
    candidate_num,
    as.character(
      third_sf$candidate_id
    )
  )


  third_sf <- third_sf[
    third_order,
    ,
    drop = FALSE
  ]


  missing_base <- is.na(
    third_sf$base_image_file
  )


  if (any(missing_base)) {

    stop(
      paste0(
        sum(missing_base),
        " Stage-3 object(s) could not be linked to a review PNG."
      )
    )
  }


  third_meta <- sf::st_drop_geometry(
    third_sf
  )


  keep_cols <- intersect(
    c(
      "candidate_id",
      "area_m2",
      "stage1_class",
      "stage1_reason",
      "second_label",
      "review_src",
      "representative_chip_id",
      "display_chip_id",
      "base_image_file",
      "box_left_pct",
      "box_top_pct",
      "box_width_pct",
      "box_height_pct"
    ),
    names(
      third_meta
    )
  )


  third_meta <- third_meta[
    ,
    keep_cols,
    drop = FALSE
  ]


  write.csv(
    third_meta,
    THIRD_CANDIDATE_CSV,
    row.names = FALSE
  )


  list(
    review_grid = review_grid,
    third_sf = third_sf,
    third_meta = third_meta,
    spatial_state = spatial_state
  )
}


# ============================================================
# 15. FINAL EXPORT
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


export_all_results <- function() {

  spatial_state <- build_spatial_state()


  review_grid <- spatial_state$review_grid

  predictions <- spatial_state$predictions


  # ----------------------------------------------------------
  # First-stage CSV
  # ----------------------------------------------------------

  write.csv(
    sf::st_drop_geometry(
      review_grid
    ),
    FIRST_REVIEW_CSV,
    row.names = FALSE
  )


  # ----------------------------------------------------------
  # Current active second-review candidate set
  # ----------------------------------------------------------

  active_second <- predictions[
    predictions$stage1_class ==
      "second_review" &
      predictions$area_m2 >=
        MIN_SECOND_REVIEW_AREA_M2,
    ,
    drop = FALSE
  ]


  second_reviews <- get_second_reviews()


  # Only classifications belonging to the CURRENT Stage-2 pool
  # are allowed to contribute to the final inventory. This avoids
  # stale Stage-2 decisions from an earlier pool being reused.
  active_second_reviews <- second_reviews[
    second_reviews$candidate_id %in%
      active_second$candidate_id,
    ,
    drop = FALSE
  ]


  second_match <- match(
    active_second$candidate_id,
    active_second_reviews$candidate_id
  )


  active_second$second_label <- ifelse(
    is.na(second_match),
    "unreviewed",
    active_second_reviews$label[
      second_match
    ]
  )


  active_second$second_annotator <- ifelse(
    is.na(second_match),
    NA_character_,
    active_second_reviews$annotator[
      second_match
    ]
  )


  active_second$second_notes <- ifelse(
    is.na(second_match),
    NA_character_,
    active_second_reviews$notes[
      second_match
    ]
  )


  active_second$second_reviewed_at <- ifelse(
    is.na(second_match),
    NA_character_,
    active_second_reviews$timestamp[
      second_match
    ]
  )


  # ----------------------------------------------------------
  # Attach current Stage-2 decisions back to the master
  # predictions table. This avoids relying on separate sf objects
  # when the final inventory is assembled.
  # ----------------------------------------------------------

  predictions$second_label <- NA_character_
  predictions$second_annotator <- NA_character_
  predictions$second_notes <- NA_character_
  predictions$second_reviewed_at <- NA_character_


  p_second <- match(
    predictions$candidate_id,
    active_second$candidate_id
  )


  has_second <- !is.na(
    p_second
  )


  predictions$second_label[
    has_second
  ] <- active_second$second_label[
    p_second[
      has_second
    ]
  ]


  predictions$second_annotator[
    has_second
  ] <- active_second$second_annotator[
    p_second[
      has_second
    ]
  ]


  predictions$second_notes[
    has_second
  ] <- active_second$second_notes[
    p_second[
      has_second
    ]
  ]


  predictions$second_reviewed_at[
    has_second
  ] <- active_second$second_reviewed_at[
    p_second[
      has_second
    ]
  ]


  second_csv <- sf::st_drop_geometry(
    active_second
  )


  second_keep_cols <- intersect(
    c(
      "candidate_id",
      "area_m2",
      "stage1_reason",
      "representative_chip_id",
      "second_label",
      "second_annotator",
      "second_notes",
      "second_reviewed_at"
    ),
    names(
      second_csv
    )
  )


  write.csv(
    second_csv[
      ,
      second_keep_cols,
      drop = FALSE
    ],
    SECOND_REVIEW_CSV,
    row.names = FALSE
  )


  # ----------------------------------------------------------
  # Build the final retained object set explicitly
  # ----------------------------------------------------------

  # Minimum mapping unit is applied consistently to the final
  # object inventory. Features smaller than 0.25 m2 are not
  # included as final waste objects.
  below_min <- predictions$area_m2 <
    MIN_SECOND_REVIEW_AREA_M2


  eligible <- !below_min


  # Stage 1: prediction objects belonging to locations classified
  # as Waste during the 50 m review.
  stage1_verified <- predictions[
    eligible &
      predictions$stage1_class ==
        "retained",
    ,
    drop = FALSE
  ]


  stage1_verified$review_src <- "stage1"


  # Stage 2: ONLY CURRENT Stage-2 candidate IDs explicitly marked
  # Waste. The selection is made from the master predictions table
  # after current Stage-2 labels have been attached.
  stage2_verified <- predictions[
    eligible &
      predictions$stage1_class ==
        "second_review" &
      !is.na(
        predictions$second_label
      ) &
      predictions$second_label ==
        "waste",
    ,
    drop = FALSE
  ]


  stage2_verified$review_src <- "stage2"


  final_verified <- rbind(
    stage1_verified,
    stage2_verified
  )


  final_verified <- final_verified[
    !duplicated(
      final_verified$candidate_id
    ),
    ,
    drop = FALSE
  ]


  # Stage-1 rejected predictions.
  stage1_rejected <- predictions[
    eligible &
      predictions$stage1_class ==
        "rejected",
    ,
    drop = FALSE
  ]


  # Stage-2 predictions explicitly marked Not waste.
  stage2_rejected <- predictions[
    eligible &
      predictions$stage1_class ==
        "second_review" &
      !is.na(
        predictions$second_label
      ) &
      predictions$second_label ==
        "not_waste",
    ,
    drop = FALSE
  ]


  final_rejected <- rbind(
    stage1_rejected,
    stage2_rejected
  )


  final_rejected <- final_rejected[
    !duplicated(
      final_rejected$candidate_id
    ),
    ,
    drop = FALSE
  ]


  stage2_unresolved <- predictions[
    eligible &
      predictions$stage1_class ==
        "second_review" &
      (
        is.na(
          predictions$second_label
        ) |
          predictions$second_label %in%
            c(
              "unreviewed",
              "not_sure"
            )
      ),
    ,
    drop = FALSE
  ]


  final_unresolved <- predictions[
    eligible &
      predictions$stage1_class ==
        "pending",
    ,
    drop = FALSE
  ]


  final_unresolved <- rbind(
    final_unresolved,
    stage2_unresolved
  )


  final_unresolved <- final_unresolved[
    !duplicated(
      final_unresolved$candidate_id
    ),
    ,
    drop = FALSE
  ]


  below_min_predictions <- predictions[
    below_min,
    ,
    drop = FALSE
  ]


  # ----------------------------------------------------------
  # HARD VALIDATION OF FINAL DECISIONS
  # ----------------------------------------------------------

  stage2_not_waste_ids <- stage2_rejected$candidate_id

  stage2_unresolved_ids <- stage2_unresolved$candidate_id

  final_ids <- final_verified$candidate_id


  wrongly_retained_not_waste <- intersect(
    final_ids,
    stage2_not_waste_ids
  )


  wrongly_retained_unresolved <- intersect(
    final_ids,
    stage2_unresolved_ids
  )


  if (length(wrongly_retained_not_waste) > 0) {

    stop(
      paste0(
        "FINAL-INVENTORY VALIDATION FAILED: ",
        length(wrongly_retained_not_waste),
        " Stage-2 object(s) marked Not waste are present in the final inventory. Candidate IDs: ",
        paste(
          head(
            wrongly_retained_not_waste,
            20
          ),
          collapse = ", "
        )
      )
    )
  }


  if (length(wrongly_retained_unresolved) > 0) {

    stop(
      paste0(
        "FINAL-INVENTORY VALIDATION FAILED: ",
        length(wrongly_retained_unresolved),
        " unresolved Stage-2 object(s) are present in the final inventory."
      )
    )
  }


  expected_stage2_waste_ids <- predictions$candidate_id[
    eligible &
      predictions$stage1_class ==
        "second_review" &
      !is.na(
        predictions$second_label
      ) &
      predictions$second_label ==
        "waste"
  ]


  exported_stage2_waste_ids <- stage2_verified$candidate_id


  if (
    !setequal(
      expected_stage2_waste_ids,
      exported_stage2_waste_ids
    )
  ) {

    stop(
      "FINAL-INVENTORY VALIDATION FAILED: exported Stage-2 Waste IDs do not match the current Stage-2 Waste decisions."
    )
  }


  # ----------------------------------------------------------
  # Stage 3 final positive audit
  # ----------------------------------------------------------

  pre_stage3_verified <- final_verified


  third_reviews <- get_third_reviews()


  active_third_reviews <- third_reviews[
    third_reviews$candidate_id %in%
      pre_stage3_verified$candidate_id,
    ,
    drop = FALSE
  ]


  third_match <- match(
    pre_stage3_verified$candidate_id,
    active_third_reviews$candidate_id
  )


  pre_stage3_verified$third_label <- ifelse(
    is.na(third_match),
    "unreviewed",
    active_third_reviews$label[
      third_match
    ]
  )


  pre_stage3_verified$third_annotator <- ifelse(
    is.na(third_match),
    NA_character_,
    active_third_reviews$annotator[
      third_match
    ]
  )


  pre_stage3_verified$third_notes <- ifelse(
    is.na(third_match),
    NA_character_,
    active_third_reviews$notes[
      third_match
    ]
  )


  pre_stage3_verified$third_reviewed_at <- ifelse(
    is.na(third_match),
    NA_character_,
    active_third_reviews$timestamp[
      third_match
    ]
  )


  third_kept <- pre_stage3_verified[
    pre_stage3_verified$third_label ==
      "keep",
    ,
    drop = FALSE
  ]


  third_deleted <- pre_stage3_verified[
    pre_stage3_verified$third_label ==
      "delete",
    ,
    drop = FALSE
  ]


  third_needs_edit <- pre_stage3_verified[
    pre_stage3_verified$third_label ==
      "needs_edit",
    ,
    drop = FALSE
  ]


  third_unresolved <- pre_stage3_verified[
    pre_stage3_verified$third_label %in%
      c(
        "unreviewed",
        "not_sure"
      ),
    ,
    drop = FALSE
  ]


  # From this point onward, "final_verified" means objects that
  # survived the Stage-3 final positive audit.
  final_verified <- third_kept


  if (nrow(third_deleted) > 0) {

    third_deleted_aligned <- third_deleted[
      ,
      intersect(
        names(final_rejected),
        names(third_deleted)
      ),
      drop = FALSE
    ]

    # Keep Stage-3 deletes as their own output layer below rather
    # than forcing them into the earlier rejection schema.
  }


  final_unresolved <- rbind(
    final_unresolved,
    third_needs_edit[
      ,
      names(final_unresolved),
      drop = FALSE
    ],
    third_unresolved[
      ,
      names(final_unresolved),
      drop = FALSE
    ]
  )


  final_unresolved <- final_unresolved[
    !duplicated(
      final_unresolved$candidate_id
    ),
    ,
    drop = FALSE
  ]


  third_csv <- sf::st_drop_geometry(
    pre_stage3_verified
  )


  third_keep_cols <- intersect(
    c(
      "candidate_id",
      "area_m2",
      "review_src",
      "stage1_class",
      "stage1_reason",
      "second_label",
      "third_label",
      "third_annotator",
      "third_notes",
      "third_reviewed_at"
    ),
    names(
      third_csv
    )
  )


  write.csv(
    third_csv[
      ,
      third_keep_cols,
      drop = FALSE
    ],
    THIRD_REVIEW_CSV,
    row.names = FALSE
  )


  # ----------------------------------------------------------
  # Spatial subsets
  # ----------------------------------------------------------

  retained_locations <- review_grid[
    review_grid$review_class ==
      "waste",
    ,
    drop = FALSE
  ]


  mixed_locations <- review_grid[
    review_grid$review_class ==
      "mixed",
    ,
    drop = FALSE
  ]


  not_sure_locations <- review_grid[
    review_grid$review_class ==
      "not_sure",
    ,
    drop = FALSE
  ]


  rejected_locations <- review_grid[
    review_grid$review_class ==
      "not_waste",
    ,
    drop = FALSE
  ]


  pending_locations <- review_grid[
    review_grid$review_class ==
      "unreviewed",
    ,
    drop = FALSE
  ]


  second_review_locations <- review_grid[
    review_grid$review_class %in%
      c(
        "mixed",
        "not_sure"
      ),
    ,
    drop = FALSE
  ]


  stage1_retained_predictions <- predictions[
    predictions$stage1_class ==
      "retained",
    ,
    drop = FALSE
  ]


  stage1_second_predictions <- predictions[
    predictions$stage1_class ==
      "second_review",
    ,
    drop = FALSE
  ]


  stage1_rejected_predictions <- predictions[
    predictions$stage1_class ==
      "rejected",
    ,
    drop = FALSE
  ]


  stage1_pending_predictions <- predictions[
    predictions$stage1_class ==
      "pending",
    ,
    drop = FALSE
  ]


  second_confirmed_waste <- active_second[
    active_second$second_label ==
      "waste",
    ,
    drop = FALSE
  ]


  second_rejected <- active_second[
    active_second$second_label ==
      "not_waste",
    ,
    drop = FALSE
  ]


  second_unresolved <- active_second[
    active_second$second_label %in%
      c(
        "unreviewed",
        "not_sure"
      ),
    ,
    drop = FALSE
  ]




  # ----------------------------------------------------------
  # Fresh GeoPackage
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

  write_gpkg_layer(
    retained_locations,
    "retained_locations"
  )

  write_gpkg_layer(
    mixed_locations,
    "mixed_locations"
  )

  write_gpkg_layer(
    not_sure_locations,
    "not_sure_locations"
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


  write_gpkg_layer(
    stage1_retained_predictions,
    "stage1_retained_predictions"
  )

  write_gpkg_layer(
    stage1_second_predictions,
    "stage1_second_review_predictions"
  )

  write_gpkg_layer(
    stage1_rejected_predictions,
    "stage1_rejected_predictions"
  )

  write_gpkg_layer(
    stage1_pending_predictions,
    "stage1_pending_predictions"
  )


  write_gpkg_layer(
    active_second,
    "second_review_candidates"
  )

  write_gpkg_layer(
    second_confirmed_waste,
    "second_review_waste"
  )

  write_gpkg_layer(
    second_rejected,
    "second_review_rejected"
  )

  write_gpkg_layer(
    second_unresolved,
    "second_review_unresolved"
  )


  write_gpkg_layer(
    pre_stage3_verified,
    "stage3_candidates"
  )

  write_gpkg_layer(
    third_kept,
    "stage3_keep"
  )

  write_gpkg_layer(
    third_deleted,
    "stage3_delete"
  )

  write_gpkg_layer(
    third_needs_edit,
    "stage3_needs_edit"
  )

  write_gpkg_layer(
    third_unresolved,
    "stage3_unresolved"
  )


  write_gpkg_layer(
    final_verified,
    "final_verified_waste"
  )

  write_gpkg_layer(
    final_rejected,
    "final_rejected_predictions"
  )

  write_gpkg_layer(
    final_unresolved,
    "final_unresolved_predictions"
  )

  write_gpkg_layer(
    below_min_predictions,
    "below_min_area_predictions"
  )


  # ----------------------------------------------------------
  # AUDIT RETAINED OBJECTS AGAINST STAGE-1 REJECTED CELLS
  # ----------------------------------------------------------

  rejected_review_cells <- review_grid[
    review_grid$review_class ==
      "not_waste",
    ,
    drop = FALSE
  ]


  final_verified$n_rejcell <- 0L

  final_verified$touch_rej <- FALSE


  if (
    nrow(final_verified) > 0 &&
    nrow(rejected_review_cells) > 0
  ) {

    rej_hits <- sf::st_intersects(
      final_verified,
      rejected_review_cells
    )


    final_verified$n_rejcell <- lengths(
      rej_hits
    )


    final_verified$touch_rej <-
      final_verified$n_rejcell > 0
  }


  # ----------------------------------------------------------
  # COMPLETE DECISION AUDIT TABLE
  # ----------------------------------------------------------

  audit_table <- sf::st_drop_geometry(
    predictions
  )


  audit_table$third_label <- NA_character_


  audit_third_match <- match(
    audit_table$candidate_id,
    pre_stage3_verified$candidate_id
  )


  audit_has_third <- !is.na(
    audit_third_match
  )


  audit_table$third_label[
    audit_has_third
  ] <- pre_stage3_verified$third_label[
    audit_third_match[
      audit_has_third
    ]
  ]


  audit_table$final_decision <- "unresolved"


  audit_table$final_decision[
    below_min
  ] <- "below_min_area"


  audit_table$final_decision[
    eligible &
      predictions$stage1_class ==
        "retained"
  ] <- "retained_stage1"


  audit_table$final_decision[
    eligible &
      predictions$stage1_class ==
        "rejected"
  ] <- "rejected_stage1"


  audit_table$final_decision[
    eligible &
      predictions$stage1_class ==
        "second_review" &
      predictions$second_label ==
        "waste"
  ] <- "retained_stage2"


  audit_table$final_decision[
    eligible &
      predictions$stage1_class ==
        "second_review" &
      predictions$second_label ==
        "not_waste"
  ] <- "rejected_stage2"


  audit_table$final_decision[
    eligible &
      predictions$stage1_class ==
        "second_review" &
      (
        is.na(
          predictions$second_label
        ) |
          predictions$second_label %in%
            c(
              "unreviewed",
              "not_sure"
            )
      )
  ] <- "unresolved_stage2"


  # Stage-3 decisions supersede earlier positive decisions.
  audit_table$final_decision[
    audit_table$third_label ==
      "keep"
  ] <- "final_keep_stage3"


  audit_table$final_decision[
    audit_table$third_label ==
      "delete"
  ] <- "deleted_stage3"


  audit_table$final_decision[
    audit_table$third_label ==
      "needs_edit"
  ] <- "needs_edit_stage3"


  audit_table$final_decision[
    audit_table$third_label %in%
      c(
        "unreviewed",
        "not_sure"
      )
  ] <- "unresolved_stage3"


  audit_keep <- intersect(
    c(
      "candidate_id",
      "area_m2",
      "stage1_class",
      "stage1_reason",
      "representative_chip_id",
      "second_label",
      "second_annotator",
      "second_notes",
      "second_reviewed_at",
      "third_label",
      "final_decision"
    ),
    names(
      audit_table
    )
  )


  write.csv(
    audit_table[
      ,
      audit_keep,
      drop = FALSE
    ],
    FINAL_AUDIT_CSV,
    row.names = FALSE
  )


  # ----------------------------------------------------------
  # Combine retained objects into contiguous waste piles
  # ----------------------------------------------------------

  # IMPORTANT:
  # Do NOT cast a dissolved geometry to POLYGON here. A retained
  # prediction can be multipart, and st_cast(..., "POLYGON") can
  # split one retained object into several tiny polygons. That was
  # the reason the earlier summary reported 4,599 "piles", including
  # many ~0.001 m2 pieces.
  #
  # Instead, touching/intersecting RETAINED OBJECTS are grouped into
  # connected components. Each connected component becomes one final
  # pile. Therefore:
  #
  # final pile count <= retained prediction-object count.

  final_unit_count <- 0L

  final_unit_area_m2 <- 0

  final_units <- NULL


  if (nrow(final_verified) > 0) {

    valid_final <- suppressWarnings(
      sf::st_make_valid(
        final_verified
      )
    )


    adjacency <- sf::st_intersects(
      valid_final
    )


    n_obj <- nrow(
      valid_final
    )


    component_id <- rep(
      NA_integer_,
      n_obj
    )


    component_counter <- 0L


    for (i in seq_len(n_obj)) {

      if (!is.na(component_id[i])) {
        next
      }


      component_counter <- component_counter + 1L

      queue <- i

      component_id[i] <- component_counter


      while (length(queue) > 0) {

        current <- queue[1]

        queue <- queue[-1]


        neighbours <- adjacency[[current]]

        new_neighbours <- neighbours[
          is.na(
            component_id[neighbours]
          )
        ]


        if (length(new_neighbours) > 0) {

          component_id[new_neighbours] <-
            component_counter

          queue <- c(
            queue,
            new_neighbours
          )
        }
      }
    }


    unit_rows <- lapply(
      seq_len(component_counter),
      function(comp) {

        ix <- which(
          component_id ==
            comp
        )


        geom <- suppressWarnings(
          sf::st_union(
            sf::st_geometry(
              valid_final[ix, ]
            )
          )
        )


        src_values <- unique(
          valid_final$review_src[ix]
        )


        source_value <- if (
          all(
            c(
              "stage1",
              "stage2"
            ) %in%
              src_values
          )
        ) {

          "both"

        } else {

          src_values[1]
        }


        sf::st_sf(

          waste_id = sprintf(
            "W%06d",
            comp
          ),

          source = source_value,

          n_parts = length(ix),

          n_s1 = sum(
            valid_final$review_src[ix] ==
              "stage1"
          ),

          n_s2 = sum(
            valid_final$review_src[ix] ==
              "stage2"
          ),

          geometry = geom
        )
      }
    )


    final_units <- do.call(
      rbind,
      unit_rows
    )


    sf::st_crs(final_units) <-
      sf::st_crs(
        final_verified
      )


    final_units$area_m2 <- as.numeric(
      sf::st_area(
        final_units
      )
    )


    final_unit_count <- nrow(
      final_units
    )


    final_unit_area_m2 <- sum(
      final_units$area_m2,
      na.rm = TRUE
    )


    write_gpkg_layer(
      final_units,
      "final_verified_waste_units"
    )
  }


  # ----------------------------------------------------------
  # Final verified waste shapefile
  # ----------------------------------------------------------

  if (dir.exists(FINAL_SHP_DIR)) {

    unlink(
      FINAL_SHP_DIR,
      recursive = TRUE,
      force = TRUE
    )
  }


  dir.create(
    FINAL_SHP_DIR,
    recursive = TRUE,
    showWarnings = FALSE
  )


  if (file.exists(FINAL_SHP_ZIP)) {

    file.remove(
      FINAL_SHP_ZIP
    )
  }


  if (
    nrow(final_verified) > 0
  ) {

    # Exact combined retained objects:
    # Stage-1 retained prediction polygons + Stage-2 Waste polygons.
    verified_objects <- sf::st_sf(

      cand_id = as.character(
        final_verified$candidate_id
      ),

      area_m2 = as.numeric(
        final_verified$area_m2
      ),

      source = as.character(
        final_verified$review_src
      ),

      s1_class = as.character(
        final_verified$stage1_class
      ),

      s1_reason = as.character(
        final_verified$stage1_reason
      ),

      s2_label = as.character(
        ifelse(
          is.na(
            final_verified$second_label
          ),
          "",
          final_verified$second_label
        )
      ),

      s3_label = as.character(
        final_verified$third_label
      ),

      n_rejcell = as.integer(
        final_verified$n_rejcell
      ),

      touch_rej = as.integer(
        final_verified$touch_rej
      ),

      geometry = sf::st_geometry(
        final_verified
      ),

      crs = sf::st_crs(
        final_verified
      )
    )


    object_shp_path <- file.path(
      FINAL_SHP_DIR,
      "verified_waste_objects.shp"
    )


    sf::st_write(
      verified_objects,
      object_shp_path,
      delete_layer = TRUE,
      quiet = TRUE
    )
  }


  if (
    nrow(final_rejected) > 0
  ) {

    rejected_objects <- sf::st_sf(

      cand_id = as.character(
        final_rejected$candidate_id
      ),

      area_m2 = as.numeric(
        final_rejected$area_m2
      ),

      s1_class = as.character(
        final_rejected$stage1_class
      ),

      s1_reason = as.character(
        final_rejected$stage1_reason
      ),

      s2_label = as.character(
        ifelse(
          is.na(
            final_rejected$second_label
          ),
          "",
          final_rejected$second_label
        )
      ),

      geometry = sf::st_geometry(
        final_rejected
      ),

      crs = sf::st_crs(
        final_rejected
      )
    )


    rejected_shp_path <- file.path(
      FINAL_SHP_DIR,
      "rejected_waste_objects.shp"
    )


    sf::st_write(
      rejected_objects,
      rejected_shp_path,
      delete_layer = TRUE,
      quiet = TRUE
    )
  }


  if (
    nrow(third_deleted) > 0
  ) {

    sf::st_write(
      third_deleted[
        ,
        intersect(
          c(
            "candidate_id",
            "area_m2",
            "review_src",
            "third_label",
            "geometry"
          ),
          names(third_deleted)
        ),
        drop = FALSE
      ],
      file.path(
        FINAL_SHP_DIR,
        "stage3_deleted_objects.shp"
      ),
      delete_layer = TRUE,
      quiet = TRUE
    )
  }


  if (
    nrow(third_needs_edit) > 0
  ) {

    sf::st_write(
      third_needs_edit[
        ,
        intersect(
          c(
            "candidate_id",
            "area_m2",
            "review_src",
            "third_label",
            "geometry"
          ),
          names(third_needs_edit)
        ),
        drop = FALSE
      ],
      file.path(
        FINAL_SHP_DIR,
        "stage3_needs_edit_objects.shp"
      ),
      delete_layer = TRUE,
      quiet = TRUE
    )
  }


  if (
    !is.null(final_units) &&
    nrow(final_units) > 0
  ) {

    # Contiguous pile layer used for pile counts and size summaries.
    pile_shp_path <- file.path(
      FINAL_SHP_DIR,
      "verified_waste_piles.shp"
    )


    sf::st_write(
      final_units[
        ,
        c(
          "waste_id",
          "area_m2",
          "source",
          "n_parts",
          "n_s1",
          "n_s2",
          "geometry"
        )
      ],
      pile_shp_path,
      delete_layer = TRUE,
      quiet = TRUE
    )
  }


  shp_files <- list.files(
    FINAL_SHP_DIR,
    full.names = TRUE
  )


  if (length(shp_files) > 0) {

    zip::zipr(
      zipfile = FINAL_SHP_ZIP,
      files = shp_files
    )
  }


  # ----------------------------------------------------------
  # Final pile summary
  # ----------------------------------------------------------

  unresolved_final_n <- nrow(
    final_unresolved
  )


  if (
    !is.null(final_units) &&
    nrow(final_units) > 0
  ) {

    pile_area <- final_units$area_m2


    pile_summary <- data.frame(

      statistic = c(
        "Stage-1 objects entering Stage 3",
        "Stage-2 objects entering Stage 3",
        "Total objects entering Stage 3",
        "Stage-3 kept objects",
        "Stage-3 deleted objects",
        "Stage-3 needs-edit objects",
        "Stage-3 unresolved objects",
        "Final contiguous waste piles",
        "Total mapped waste area (m²)",
        "Total mapped waste area (ha)",
        "Mean pile area (m²)",
        "Median pile area (m²)",
        "Standard deviation (m²)",
        "Minimum pile area (m²)",
        "25th percentile (m²)",
        "75th percentile (m²)",
        "Interquartile range (m²)",
        "90th percentile (m²)",
        "95th percentile (m²)",
        "Maximum pile area (m²)",
        "Final piles containing Stage-1 objects only",
        "Final piles containing Stage-2 objects only",
        "Final piles containing both Stage-1 and Stage-2 objects",
        "Unresolved Stage-2 objects",
        "Unresolved / needs-edit Stage-3 objects"
      ),

      value = c(
        nrow(stage1_verified),
        nrow(stage2_verified),
        nrow(pre_stage3_verified),
        nrow(third_kept),
        nrow(third_deleted),
        nrow(third_needs_edit),
        nrow(third_unresolved),
        nrow(final_units),
        sum(pile_area, na.rm = TRUE),
        sum(pile_area, na.rm = TRUE) / 10000,
        mean(pile_area, na.rm = TRUE),
        median(pile_area, na.rm = TRUE),
        stats::sd(pile_area, na.rm = TRUE),
        min(pile_area, na.rm = TRUE),
        stats::quantile(
          pile_area,
          0.25,
          na.rm = TRUE,
          names = FALSE
        ),
        stats::quantile(
          pile_area,
          0.75,
          na.rm = TRUE,
          names = FALSE
        ),
        stats::IQR(
          pile_area,
          na.rm = TRUE
        ),
        stats::quantile(
          pile_area,
          0.90,
          na.rm = TRUE,
          names = FALSE
        ),
        stats::quantile(
          pile_area,
          0.95,
          na.rm = TRUE,
          names = FALSE
        ),
        max(pile_area, na.rm = TRUE),
        sum(final_units$source == "stage1"),
        sum(final_units$source == "stage2"),
        sum(final_units$source == "both"),
        nrow(stage2_unresolved),
        nrow(third_unresolved) + nrow(third_needs_edit)
      ),

      stringsAsFactors = FALSE
    )


    size_breaks <- c(
      MIN_SECOND_REVIEW_AREA_M2,
      0.5,
      1,
      2,
      5,
      10,
      25,
      50,
      100,
      Inf
    )


    size_labels <- c(
      "0.25–<0.5 m²",
      "0.5–<1 m²",
      "1–<2 m²",
      "2–<5 m²",
      "5–<10 m²",
      "10–<25 m²",
      "25–<50 m²",
      "50–<100 m²",
      "≥100 m²"
    )


    pile_size_class <- cut(
      pile_area,
      breaks = size_breaks,
      labels = size_labels,
      right = FALSE,
      include.lowest = TRUE
    )


    size_table <- as.data.frame(
      table(
        pile_size_class,
        useNA = "no"
      ),
      stringsAsFactors = FALSE
    )


    names(size_table) <- c(
      "size_class",
      "n_piles"
    )


    size_table$percentage <- if (
      sum(size_table$n_piles) > 0
    ) {

      100 *
        size_table$n_piles /
        sum(
          size_table$n_piles
        )

    } else {

      0
    }


    size_table$total_area_m2 <- vapply(
      size_labels,
      function(lbl) {

        sum(
          pile_area[
            pile_size_class ==
              lbl
          ],
          na.rm = TRUE
        )
      },
      numeric(1)
    )

  } else {

    pile_summary <- data.frame(
      statistic = c(
        "Stage-1 objects entering Stage 3",
        "Stage-2 objects entering Stage 3",
        "Total objects entering Stage 3",
        "Stage-3 kept objects",
        "Stage-3 deleted objects",
        "Stage-3 needs-edit objects",
        "Stage-3 unresolved objects",
        "Final contiguous waste piles",
        "Unresolved Stage-2 objects"
      ),
      value = c(
        nrow(stage1_verified),
        nrow(stage2_verified),
        nrow(pre_stage3_verified),
        nrow(third_kept),
        nrow(third_deleted),
        nrow(third_needs_edit),
        nrow(third_unresolved),
        0,
        nrow(stage2_unresolved)
      ),
      stringsAsFactors = FALSE
    )


    size_table <- data.frame(
      size_class = character(0),
      n_piles = integer(0),
      percentage = numeric(0),
      total_area_m2 = numeric(0),
      stringsAsFactors = FALSE
    )
  }


  write.csv(
    pile_summary,
    FINAL_PILE_SUMMARY_CSV,
    row.names = FALSE
  )


  write.csv(
    size_table,
    FINAL_SIZE_CLASS_CSV,
    row.names = FALSE
  )


  # ----------------------------------------------------------
  # Summary
  # ----------------------------------------------------------

  second_counts <- table(
    factor(
      active_second$second_label,
      levels = c(
        "waste",
        "not_waste",
        "not_sure",
        "unreviewed"
      )
    )
  )


  summary_table <- data.frame(

    measure = c(
      "total_first_review_locations",
      "retained_locations",
      "mixed_locations",
      "not_sure_locations",
      "rejected_locations",
      "pending_locations",

      "raw_prediction_polygons",
      "predictions_below_min_area",
      "stage1_retained_predictions",
      "stage1_second_review_predictions",
      "stage1_rejected_predictions",
      "stage1_pending_predictions",

      "active_second_review_candidates",
      "second_review_waste",
      "second_review_not_waste",
      "second_review_not_sure",
      "second_review_unreviewed",

      "stage1_verified_objects",
      "stage2_verified_objects",
      "stage3_candidate_objects",
      "stage3_kept_objects",
      "stage3_deleted_objects",
      "stage3_needs_edit_objects",
      "stage3_unresolved_objects",
      "final_verified_prediction_polygons",
      "final_rejected_prediction_polygons",
      "final_unresolved_prediction_polygons",
      "final_verified_contiguous_units",
      "final_verified_total_area_m2",
      "final_verified_mean_area_m2",
      "final_verified_median_area_m2",
      "final_verified_max_area_m2",
      "final_retained_touching_stage1_rejected_cells"
    ),

    value = c(
      nrow(review_grid),
      nrow(retained_locations),
      nrow(mixed_locations),
      nrow(not_sure_locations),
      nrow(rejected_locations),
      nrow(pending_locations),

      nrow(predictions),
      nrow(below_min_predictions),
      nrow(stage1_retained_predictions),
      nrow(stage1_second_predictions),
      nrow(stage1_rejected_predictions),
      nrow(stage1_pending_predictions),

      nrow(active_second),
      as.integer(second_counts["waste"]),
      as.integer(second_counts["not_waste"]),
      as.integer(second_counts["not_sure"]),
      as.integer(second_counts["unreviewed"]),

      nrow(stage1_verified),
      nrow(stage2_verified),
      nrow(pre_stage3_verified),
      nrow(third_kept),
      nrow(third_deleted),
      nrow(third_needs_edit),
      nrow(third_unresolved),
      nrow(final_verified),
      nrow(final_rejected),
      nrow(final_unresolved),
      final_unit_count,
      final_unit_area_m2,
      if (
        !is.null(final_units) &&
        nrow(final_units) > 0
      ) mean(final_units$area_m2) else 0,
      if (
        !is.null(final_units) &&
        nrow(final_units) > 0
      ) median(final_units$area_m2) else 0,
      if (
        !is.null(final_units) &&
        nrow(final_units) > 0
      ) max(final_units$area_m2) else 0,
      sum(
        final_verified$touch_rej,
        na.rm = TRUE
      )
    ),

    stringsAsFactors = FALSE
  )


  write.csv(
    summary_table,
    EXPORT_SUMMARY_CSV,
    row.names = FALSE
  )


  list(
    summary = summary_table,
    gpkg = EXPORT_GPKG,
    first_csv = FIRST_REVIEW_CSV,
    second_csv = SECOND_REVIEW_CSV,
    third_csv = THIRD_REVIEW_CSV,
    summary_csv = EXPORT_SUMMARY_CSV,
    pile_summary = pile_summary,
    size_table = size_table,
    pile_summary_csv = FINAL_PILE_SUMMARY_CSV,
    size_class_csv = FINAL_SIZE_CLASS_CSV,
    audit_csv = FINAL_AUDIT_CSV,
    shapefile_zip = FINAL_SHP_ZIP,
    final_units = final_units
  )
}


# ============================================================
# 14. UI
# ============================================================

ui <- fluidPage(


  tags$head(


    # --------------------------------------------------------
    # Keyboard shortcuts
    # --------------------------------------------------------

    tags$script(
      HTML(
        "

        function positionStage2TargetBox() {

          var img = document.getElementById('stage2-review-image');
          var box = document.getElementById('stage2-target-box');

          if (!img || !box) {
            return;
          }

          if (!img.complete || !img.naturalWidth || !img.naturalHeight) {
            return;
          }

          var leftPct = parseFloat(box.dataset.left);
          var topPct = parseFloat(box.dataset.top);
          var widthPct = parseFloat(box.dataset.width);
          var heightPct = parseFloat(box.dataset.height);

          if (
            !Number.isFinite(leftPct) ||
            !Number.isFinite(topPct) ||
            !Number.isFinite(widthPct) ||
            !Number.isFinite(heightPct)
          ) {
            // Defensive fallback: a visible locator in the middle.
            leftPct = 47;
            topPct = 47;
            widthPct = 6;
            heightPct = 6;
          }

          var elementWidth = img.clientWidth;
          var elementHeight = img.clientHeight;

          if (elementWidth <= 0 || elementHeight <= 0) {
            return;
          }

          var naturalRatio = img.naturalWidth / img.naturalHeight;
          var elementRatio = elementWidth / elementHeight;

          var renderedWidth;
          var renderedHeight;
          var offsetX;
          var offsetY;

          if (naturalRatio > elementRatio) {

            renderedWidth = elementWidth;
            renderedHeight = elementWidth / naturalRatio;

            offsetX = 0;
            offsetY = (elementHeight - renderedHeight) / 2;

          } else {

            renderedHeight = elementHeight;
            renderedWidth = elementHeight * naturalRatio;

            offsetX = (elementWidth - renderedWidth) / 2;
            offsetY = 0;
          }

          var left = offsetX + renderedWidth * leftPct / 100;
          var top = offsetY + renderedHeight * topPct / 100;
          var width = renderedWidth * widthPct / 100;
          var height = renderedHeight * heightPct / 100;

          // Keep tiny objects easy to locate without making the
          // locator excessively large.
          var minPx = 34;

          if (width < minPx) {
            left = left + width / 2 - minPx / 2;
            width = minPx;
          }

          if (height < minPx) {
            top = top + height / 2 - minPx / 2;
            height = minPx;
          }

          // Keep the complete locator within the rendered image.
          var minLeft = offsetX + 2;
          var maxRight = offsetX + renderedWidth - 2;
          var minTop = offsetY + 2;
          var maxBottom = offsetY + renderedHeight - 2;

          if (left < minLeft) {
            left = minLeft;
          }

          if (top < minTop) {
            top = minTop;
          }

          if (left + width > maxRight) {
            left = Math.max(minLeft, maxRight - width);
          }

          if (top + height > maxBottom) {
            top = Math.max(minTop, maxBottom - height);
          }

          box.style.left = left + 'px';
          box.style.top = top + 'px';
          box.style.width = width + 'px';
          box.style.height = height + 'px';
          box.style.visibility = 'visible';
        }

        window.addEventListener('resize', function() {
          window.requestAnimationFrame(positionStage2TargetBox);
        });

        document.addEventListener('shiny:value', function(e) {
          if (e.name === 'current_image') {
            setTimeout(positionStage2TargetBox, 0);
            setTimeout(positionStage2TargetBox, 50);
            setTimeout(positionStage2TargetBox, 150);
            setTimeout(positionStage2TargetBox, 300);
          }
        });

        document.addEventListener('keydown', function(e) {

          var tag = document.activeElement.tagName.toLowerCase();

          if (tag === 'input' || tag === 'textarea') {
            return;
          }

          if (e.key === '1') {
            var b = document.getElementById('mark_waste');
            if (b) b.click();
          }

          if (e.key === '2') {
            var b = document.getElementById('mark_not_waste');
            if (b) b.click();
          }

          if (e.key === '3') {
            var mixed = document.getElementById('mark_mixed');
            var unsure = document.getElementById('mark_not_sure');

            if (mixed) {
              mixed.click();
            } else if (unsure) {
              unsure.click();
            }
          }

          if (e.key === '4') {
            var b = document.getElementById('mark_not_sure');
            if (b) b.click();
          }

          if (e.key.toLowerCase() === 's') {
            var b = document.getElementById('skip_btn');
            if (b) b.click();
          }
        });
        "
      )
    ),


    # --------------------------------------------------------
    # CSS
    # --------------------------------------------------------

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
          font-size: 27px;
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

        .stage-banner {
          font-weight: 700;
          font-size: 15px;
          margin-bottom: 4px;
        }

        .chip-title {
          font-size: 18px;
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

        .target-banner {
          background: #fffbd6;
          border: 1px solid #d8ce5c;
          border-radius: 5px;
          padding: 5px 7px;
          margin-top: 5px;
          font-size: 13px;
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

        .btn-unsure {
          background-color: #eeeeee;
          border-color: #bdbdbd;
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
          font-size: 17px;
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

        /* uiOutput() inserts this wrapper. It MUST have a height,
           otherwise a 100%-height image-stage can collapse to zero. */
        .image-container > .shiny-html-output {
          width: 100%;
          height: 100%;
          display: block;
          position: relative;
        }

        .image-stage {
          position: relative;
          width: 100%;
          height: 100%;
          overflow: hidden;
        }

        .review-image {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          object-fit: contain;
          display: block;
        }

        .target-svg {
          position: absolute;
          left: 0;
          top: 0;
          width: 100%;
          height: 100%;
          pointer-events: none;
          z-index: 20;
        }

        .stage2-target-box {
          position: absolute;
          display: block;
          background: transparent !important;
          border: 6px solid #ffff00 !important;
          outline: 3px solid #000000;
          box-shadow:
            0 0 0 1px #ffffff,
            0 0 10px rgba(0, 0, 0, 0.95);
          box-sizing: border-box;
          pointer-events: none;
          z-index: 99999;
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
          "reviewer_ui"
        )
      ),


      div(
        class = "box",
        uiOutput(
          "stage_ui"
        )
      ),


      div(
        class = "box",
        uiOutput(
          "item_info"
        )
      ),


      div(
        class = "box",
        uiOutput(
          "classification_ui"
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


      div(
        class = "box",
        uiOutput(
          "correction_ui"
        )
      ),


      div(
        class = "box",
        uiOutput(
          "progress_ui"
        )
      ),


      div(

        class = "box",

        h4(
          "Export"
        ),

        actionButton(
          "export_results",
          "Export all spatial results",
          width = "100%"
        ),

        div(
          style = "height:5px;"
        ),

        downloadButton(
          "download_verified_shp",
          "Download verified waste shapefiles",
          width = "100%"
        ),

        div(
          style = "height:5px;"
        ),

        actionButton(
          "summarise_final",
          "Summarise verified waste piles",
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
            textOutput(
              "image_title",
              inline = TRUE
            )
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
# 15. SERVER
# ============================================================

server <- function(
  input,
  output,
  session
) {


  # ==========================================================
  # REVIEWER LOGIN
  # ==========================================================

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


      if (nzchar(nm)) {

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


  # ==========================================================
  # REACTIVE STATE
  # ==========================================================

  state <- reactiveValues(

    stage = "first",

    mode = "queue",

    current_id = NULL,

    previous_label = NULL,

    history_ids = character(0),

    history_index = NA_integer_,

    resolution_total = 0L

  )


  first_skipped <- reactiveVal(
    character(0)
  )


  second_skipped <- reactiveVal(
    character(0)
  )


  resolution_skipped <- reactiveVal(
    character(0)
  )


  pile_summary_rv <- reactiveVal(
    NULL
  )


  pile_size_rv <- reactiveVal(
    NULL
  )


  second_ready <- reactiveVal(
    FALSE
  )


  second_meta_rv <- reactiveVal(
    NULL
  )


  second_sf_rv <- reactiveVal(
    NULL
  )


  second_grid_rv <- reactiveVal(
    NULL
  )


  third_ready <- reactiveVal(
    FALSE
  )


  third_meta_rv <- reactiveVal(
    NULL
  )


  third_sf_rv <- reactiveVal(
    NULL
  )


  third_grid_rv <- reactiveVal(
    NULL
  )


  third_skipped <- reactiveVal(
    character(0)
  )


  export_message <- reactiveVal(
    ""
  )


  provisional_object_count <- reactiveVal(
    NA_integer_
  )


  object_count_time <- reactiveVal(
    NULL
  )


  # ==========================================================
  # LIVE TIMERS
  # ==========================================================

  stats_timer <- reactiveTimer(
    STATS_REFRESH_MS,
    session
  )


  first_stats_tick <- reactive({

    stats_timer()

    get_first_stats()
  })


  second_stats_tick <- reactive({

    stats_timer()

    meta <- second_meta_rv()


    ids <- if (
      is.null(meta)
    ) {

      character(0)

    } else {

      as.character(
        meta$candidate_id
      )
    }


    get_second_stats(
      ids
    )
  })



  third_stats_tick <- reactive({

    stats_timer()

    meta <- third_meta_rv()


    ids <- if (
      is.null(meta)
    ) {

      character(0)

    } else {

      as.character(
        meta$candidate_id
      )
    }


    get_third_stats(
      ids
    )
  })


  # ==========================================================
  # RELEASE CURRENT CLAIM
  # ==========================================================

  release_current_claim <- function() {

    current_id <- state$current_id


    if (
      is.null(current_id) ||
      is.null(
        annotator_name()
      )
    ) {
      return(
        invisible(NULL)
      )
    }


    if (
      state$stage ==
        "first"
    ) {

      release_first_claim(
        current_id,
        annotator_name()
      )

    } else if (
      state$stage ==
        "second"
    ) {

      release_second_claim(
        current_id,
        annotator_name()
      )

    } else {

      release_third_claim(
        current_id,
        annotator_name()
      )
    }


    invisible(NULL)
  }


  # ==========================================================
  # LOAD NEXT FIRST-STAGE LOCATION
  # ==========================================================

  load_next_first <- function() {

    req(
      annotator_name()
    )


    next_id <- claim_next_first(
      annotator = annotator_name(),
      exclude_ids = first_skipped()
    )


    if (
      is.null(next_id) &&
      length(first_skipped()) > 0
    ) {

      first_skipped(
        character(0)
      )


      next_id <- claim_next_first(
        annotator = annotator_name(),
        exclude_ids = character(0)
      )
    }


    state$stage <- "first"

    state$mode <- "queue"

    state$current_id <- next_id

    state$previous_label <- NULL

    state$history_ids <- character(0)

    state$history_index <- NA_integer_


    updateTextAreaInput(
      session,
      "notes",
      value = ""
    )
  }


  # ==========================================================
  # LOAD NEXT SECOND-STAGE OBJECT
  # ==========================================================

  load_next_second <- function() {

    req(
      annotator_name()
    )


    meta <- second_meta_rv()


    if (
      is.null(meta)
    ) {

      state$current_id <- NULL

      return(
        invisible(NULL)
      )
    }


    active_ids <- as.character(
      meta$candidate_id
    )


    next_id <- claim_next_second(
      annotator = annotator_name(),
      active_ids = active_ids,
      exclude_ids = second_skipped()
    )


    if (
      is.null(next_id) &&
      length(second_skipped()) > 0
    ) {

      second_skipped(
        character(0)
      )


      next_id <- claim_next_second(
        annotator = annotator_name(),
        active_ids = active_ids,
        exclude_ids = character(0)
      )
    }


    state$stage <- "second"

    state$mode <- "queue"

    state$current_id <- next_id

    state$previous_label <- NULL

    state$history_ids <- character(0)

    state$history_index <- NA_integer_


    updateTextAreaInput(
      session,
      "notes",
      value = ""
    )
  }




  # ==========================================================
  # LOAD NEXT STAGE-3 POSITIVE-AUDIT OBJECT
  # ==========================================================

  load_next_third <- function() {

    req(
      annotator_name()
    )


    meta <- third_meta_rv()


    if (is.null(meta)) {

      state$current_id <- NULL

      return(
        invisible(NULL)
      )
    }


    active_ids <- as.character(
      meta$candidate_id
    )


    next_id <- claim_next_third(
      annotator = annotator_name(),
      active_ids = active_ids,
      exclude_ids = third_skipped()
    )


    if (
      is.null(next_id) &&
      length(third_skipped()) > 0
    ) {

      third_skipped(
        character(0)
      )


      next_id <- claim_next_third(
        annotator = annotator_name(),
        active_ids = active_ids,
        exclude_ids = character(0)
      )
    }


    state$stage <- "third"

    state$mode <- "queue"

    state$current_id <- next_id

    state$previous_label <- NULL

    state$history_ids <- character(0)

    state$history_index <- NA_integer_


    updateTextAreaInput(
      session,
      "notes",
      value = ""
    )
  }


  # ==========================================================
  # RESOLVE STAGE-2 "NOT SURE" OBJECTS
  # ==========================================================

  load_next_unresolved <- function() {

    req(
      annotator_name()
    )


    meta <- second_meta_rv()


    if (is.null(meta)) {

      showNotification(
        "Prepare Stage 2 before resolving unresolved objects.",
        type = "warning",
        duration = 5
      )

      return(
        invisible(NULL)
      )
    }


    active_ids <- as.character(
      meta$candidate_id
    )


    unresolved_ids <- get_second_unresolved_ids(
      active_ids
    )


    state$resolution_total <- length(
      unresolved_ids
    )


    if (length(unresolved_ids) == 0) {

      state$stage <- "second"

      state$mode <- "resolve"

      state$current_id <- NULL

      state$previous_label <- NULL


      showNotification(
        "All Stage-2 Not sure objects have been resolved.",
        type = "message",
        duration = 5
      )

      return(
        invisible(NULL)
      )
    }


    available <- unresolved_ids[
      !unresolved_ids %in%
        resolution_skipped()
    ]


    if (
      length(available) == 0 &&
      length(resolution_skipped()) > 0
    ) {

      resolution_skipped(
        character(0)
      )

      available <- unresolved_ids
    }


    next_id <- NULL

    next_review <- NULL


    for (candidate_id in available) {

      claimed <- claim_second_for_resolution(
        candidate_id = candidate_id,
        annotator = annotator_name(),
        active_ids = active_ids
      )


      if (isTRUE(claimed$ok)) {

        next_id <- candidate_id

        next_review <- claimed$review

        break
      }
    }


    state$stage <- "second"

    state$mode <- "resolve"

    state$current_id <- next_id

    state$previous_label <- "not_sure"

    state$history_ids <- character(0)

    state$history_index <- NA_integer_


    if (!is.null(next_id)) {

      updateTextAreaInput(
        session,
        "notes",
        value = if (
          is.null(next_review) ||
          is.na(next_review$notes[1])
        ) {
          ""
        } else {
          next_review$notes[1]
        }
      )
    }
  }


  # ==========================================================
  # INITIAL FIRST-STAGE LOCATION
  # ==========================================================

  observeEvent(
    annotator_name(),
    {

      req(
        annotator_name()
      )

      load_next_first()

    },
    ignoreInit = TRUE
  )


  # ==========================================================
  # CLAIM HEARTBEAT
  # ==========================================================

  observe({

    invalidateLater(
      CLAIM_HEARTBEAT_SECONDS * 1000,
      session
    )


    current_id <- isolate(
      state$current_id
    )


    reviewer <- isolate(
      annotator_name()
    )


    current_stage <- isolate(
      state$stage
    )


    if (
      !is.null(current_id) &&
      !is.null(reviewer)
    ) {

      if (
        current_stage ==
          "first"
      ) {

        touch_first_claim(
          current_id,
          reviewer
        )

      } else if (
        current_stage ==
          "second"
      ) {

        touch_second_claim(
          current_id,
          reviewer
        )

      } else {

        touch_third_claim(
          current_id,
          reviewer
        )
      }
    }
  })


  # ==========================================================
  # RELEASE CLAIM ON SESSION END
  # ==========================================================

  session$onSessionEnded(
    function() {

      current_id <- isolate(
        state$current_id
      )


      reviewer <- isolate(
        annotator_name()
      )


      current_stage <- isolate(
        state$stage
      )


      if (
        !is.null(current_id) &&
        !is.null(reviewer)
      ) {

        if (
          current_stage ==
            "first"
        ) {

          release_first_claim(
            current_id,
            reviewer
          )

        } else if (
          current_stage ==
            "second"
        ) {

          release_second_claim(
            current_id,
            reviewer
          )

        } else {

          release_third_claim(
            current_id,
            reviewer
          )
        }
      }
    }
  )


  # ==========================================================
  # REVIEWER PANEL
  # ==========================================================

  output$reviewer_ui <- renderUI({

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

        if (
          state$stage ==
            "first"
        ) {

          "Red outlines = model predictions."

        } else if (
          state$stage ==
            "second"
        ) {

          "Yellow box = current Stage-2 object. The image inside the box is unchanged."

        } else {

          "Yellow box = current Stage-3 retained object. Check whether it should remain in the final inventory."
        }
      )
    )
  })


  # ==========================================================
  # STAGE PANEL
  # ==========================================================

  output$stage_ui <- renderUI({

    first_stats <- first_stats_tick()


    first_remaining <- max(
      0,
      TOTAL_CHIPS -
        first_stats$reviewed
    )


    first_complete <- (
      first_remaining == 0
    )


    # --------------------------------------------------------
    # STAGE 1 SCREEN
    #
    # Stage 2 information is ALWAYS visible. The preparation
    # button is therefore never hidden simply because of a UI
    # count mismatch. The preparation function itself still
    # verifies that Stage 1 is complete before proceeding.
    # --------------------------------------------------------

    if (
      state$stage ==
        "first"
    ) {

      stage2_status <- if (
        first_complete
      ) {

        if (
          second_ready()
        ) {
          "Stage 2 pool is prepared and ready."
        } else {
          "Stage 1 is complete. Stage 2 is ready to prepare."
        }

      } else {

        paste0(
          "Stage 2 will use the Mixed and Not sure locations. ",
          first_remaining,
          " first-stage location",
          if (first_remaining == 1) "" else "s",
          " remain before Stage 2 can be prepared."
        )
      }


      button_label <- if (
        second_ready()
      ) {
        "Refresh / start second review"
      } else {
        "Prepare / start second review"
      }


      return(
        tagList(

          div(
            class = "stage-banner",
            "STAGE 1 — LOCATION REVIEW"
          ),


          div(
            class = "small-line",

            paste0(
              first_stats$reviewed,
              " / ",
              TOTAL_CHIPS,
              " locations reviewed"
            )
          ),


          hr(),


          div(
            class = "stage-banner",
            "STAGE 2 — OBJECT REVIEW"
          ),


          div(
            class = "small-line",
            stage2_status
          ),


          div(
            class = "small-line",

            paste0(
              "Second review checks individual ambiguous predictions ≥ ",
              MIN_SECOND_REVIEW_AREA_M2,
              " m²."
            )
          ),


          div(
            style = "height:6px;"
          ),


          actionButton(
            "prepare_second",
            button_label,
            width = "100%"
          ),


          hr(),


          div(
            class = "stage-banner",
            "STAGE 3 — FINAL POSITIVE AUDIT"
          ),


          div(
            class = "small-line",
            "If Stage 2 has already been completed and all Not sure objects are resolved, you can start the final positive audit directly."
          ),


          div(
            style = "height:6px;"
          ),


          actionButton(
            "prepare_third",
            if (
              third_ready()
            ) {
              "Refresh / start Stage 3"
            } else {
              "Prepare / start Stage 3"
            },
            width = "100%"
          )
        )
      )
    }


    # --------------------------------------------------------
    # STAGE 2 SCREEN
    # --------------------------------------------------------

    if (
      state$stage ==
        "second"
    ) {

      s2 <- second_stats_tick()


      return(
        tagList(

          div(
            class = "stage-banner",
            "STAGE 2 — OBJECT REVIEW"
          ),


          div(
            class = "small-line",

            paste0(
              s2$reviewed,
              " / ",
              s2$total,
              " ambiguous objects reviewed"
            )
          ),


          div(
            class = "small-line",

            paste0(
              "Minimum reviewed object area: ",
              MIN_SECOND_REVIEW_AREA_M2,
              " m²"
            )
          ),


          div(
            style = "height:5px;"
          ),


          if (
            s2$not_sure > 0
          ) {

            tagList(

              actionButton(
                "resolve_unresolved",
                paste0(
                  "Resolve Not sure objects (",
                  s2$not_sure,
                  ")"
                ),
                width = "100%"
              ),

              div(
                style = "height:5px;"
              )
            )

          } else {

            div(
              class = "small-line",
              "No Stage-2 Not sure objects remain."
            )
          },


          hr(),


          div(
            class = "stage-banner",
            "STAGE 3 — FINAL POSITIVE AUDIT"
          ),


          div(
            class = "small-line",
            "Stage 3 checks every object currently retained as waste from Stages 1 and 2."
          ),


          div(
            class = "small-line",

            if (
              s2$reviewed <
                s2$total ||
              s2$not_sure > 0
            ) {

              "Complete and resolve Stage 2 before preparing Stage 3."

            } else if (
              third_ready()
            ) {

              "Stage 3 pool is prepared and ready."

            } else {

              "Stage 2 is complete. Stage 3 is ready to prepare."
            }
          ),


          div(
            style = "height:6px;"
          ),


          actionButton(
            "prepare_third",
            if (
              third_ready()
            ) {
              "Refresh / start Stage 3"
            } else {
              "Prepare / start Stage 3"
            },
            width = "100%"
          ),


          div(
            style = "height:5px;"
          ),


          actionButton(
            "return_first_stage",
            "Return to first review",
            width = "100%"
          )
        )
      )
    }


    # --------------------------------------------------------
    # STAGE 3 SCREEN
    # --------------------------------------------------------

    s3 <- third_stats_tick()


    tagList(

      div(
        class = "stage-banner",
        "STAGE 3 — FINAL POSITIVE AUDIT"
      ),


      div(
        class = "small-line",

        paste0(
          s3$reviewed,
          " / ",
          s3$total,
          " retained objects checked"
        )
      ),


      div(
        class = "small-line",

        paste0(
          "Keep: ",
          s3$keep,
          " | Delete: ",
          s3$delete,
          " | Needs edit: ",
          s3$needs_edit,
          " | Not sure: ",
          s3$not_sure
        )
      ),


      div(
        style = "height:6px;"
      ),


      actionButton(
        "refresh_third",
        "Refresh Stage 3 pool",
        width = "100%"
      ),


      div(
        style = "height:5px;"
      ),


      actionButton(
        "return_second_stage",
        "Return to Stage 2",
        width = "100%"
      )
    )
  })


  # ==========================================================
  # PREPARE / REFRESH SECOND REVIEW
  # ==========================================================

  observeEvent(
    input$prepare_second,
    {

      req(
        annotator_name()
      )


      release_current_claim()


      result <- tryCatch(

        withProgress(
          message = "Preparing second review",
          value = 0,
          {

            incProgress(
              0.15,
              detail = "Reading current first-stage classifications..."
            )


            prepared <- prepare_second_review_pool()


            incProgress(
              0.75,
              detail = "Building object-level review pool..."
            )


            prepared
          }
        ),

        error = function(e) {

          showNotification(
            paste0(
              "Could not prepare second review: ",
              conditionMessage(e)
            ),
            type = "error",
            duration = 10
          )


          NULL
        }
      )


      if (is.null(result)) {

        load_next_first()

        return()
      }


      second_meta_rv(
        result$second_meta
      )


      second_sf_rv(
        result$second_sf
      )


      second_grid_rv(
        result$review_grid
      )


      second_ready(
        TRUE
      )


      second_skipped(
        character(0)
      )


      showNotification(
        paste0(
          "Second-review pool prepared: ",
          format(
            nrow(
              result$second_meta
            ),
            big.mark = ","
          ),
          " objects ≥ ",
          MIN_SECOND_REVIEW_AREA_M2,
          " m²."
        ),
        type = "message",
        duration = 7
      )


      load_next_second()
    }
  )




  # ==========================================================
  # PREPARE / REFRESH STAGE 3
  # ==========================================================

  prepare_and_start_third <- function() {

    req(
      annotator_name()
    )


    release_current_claim()


    result <- tryCatch(

      withProgress(
        message = "Preparing Stage 3",
        value = 0,
        {

          incProgress(
            0.20,
            detail = "Building retained positive-object pool..."
          )


          prepared <- prepare_third_review_pool()


          incProgress(
            0.80,
            detail = "Preparing review images and bounding boxes..."
          )


          prepared
        }
      ),

      error = function(e) {

        showNotification(
          paste0(
            "Could not prepare Stage 3: ",
            conditionMessage(e)
          ),
          type = "error",
          duration = 10
        )

        NULL
      }
    )


    if (is.null(result)) {
      return(
        invisible(FALSE)
      )
    }


    third_meta_rv(
      result$third_meta
    )


    third_sf_rv(
      result$third_sf
    )


    third_grid_rv(
      result$review_grid
    )


    third_ready(
      TRUE
    )


    third_skipped(
      character(0)
    )


    showNotification(
      paste0(
        "Stage-3 pool prepared: ",
        format(
          nrow(
            result$third_meta
          ),
          big.mark = ","
        ),
        " retained waste objects."
      ),
      type = "message",
      duration = 7
    )


    load_next_third()

    invisible(TRUE)
  }


  observeEvent(
    input$prepare_third,
    {

      prepare_and_start_third()
    }
  )


  observeEvent(
    input$refresh_third,
    {

      prepare_and_start_third()
    }
  )


  # ==========================================================
  # START RESOLUTION OF STAGE-2 NOT-SURE OBJECTS
  # ==========================================================

  observeEvent(
    input$resolve_unresolved,
    {

      req(
        annotator_name()
      )


      release_current_claim()


      resolution_skipped(
        character(0)
      )


      load_next_unresolved()
    }
  )


  # ==========================================================
  # RETURN TO FIRST STAGE
  # ==========================================================

  observeEvent(
    input$return_first_stage,
    {

      req(
        annotator_name()
      )


      release_current_claim()


      load_next_first()
    }
  )



  observeEvent(
    input$return_second_stage,
    {

      req(
        annotator_name()
      )


      release_current_claim()


      if (
        is.null(
          second_meta_rv()
        )
      ) {

        showNotification(
          "Stage 2 is not prepared in this session. Prepare it again from Stage 1.",
          type = "warning",
          duration = 6
        )

        load_next_first()

      } else {

        load_next_second()
      }
    }
  )


  # ==========================================================
  # CURRENT FIRST-STAGE ROW
  # ==========================================================

  current_first_row <- reactive({

    req(
      state$stage ==
        "first"
    )


    req(
      state$current_id
    )


    row <- chips[
      chips$chip_id ==
        state$current_id,
      ,
      drop = FALSE
    ]


    req(
      nrow(row) == 1
    )


    row
  })


  # ==========================================================
  # CURRENT SECOND-STAGE ROW
  # ==========================================================

  current_second_row <- reactive({

    req(
      state$stage ==
        "second"
    )


    req(
      state$current_id
    )


    meta <- second_meta_rv()


    req(
      !is.null(meta)
    )


    row <- meta[
      as.character(
        meta$candidate_id
      ) ==
        as.character(
          state$current_id
        ),
      ,
      drop = FALSE
    ]


    req(
      nrow(row) == 1
    )


    row
  })



  # ==========================================================
  # CURRENT THIRD-STAGE ROW
  # ==========================================================

  current_third_row <- reactive({

    req(
      state$stage ==
        "third"
    )


    req(
      state$current_id
    )


    meta <- third_meta_rv()


    req(
      !is.null(meta)
    )


    row <- meta[
      as.character(
        meta$candidate_id
      ) ==
        as.character(
          state$current_id
        ),
      ,
      drop = FALSE
    ]


    req(
      nrow(row) == 1
    )


    row
  })


  # ==========================================================
  # ITEM INFORMATION
  # ==========================================================

  output$item_info <- renderUI({

    if (
      is.null(
        state$current_id
      )
    ) {

      if (
        state$stage ==
          "first"
      ) {

        return(
          tagList(

            div(
              class = "chip-title",
              "First review complete"
            ),


            div(
              class = "small-line",
              "Use 'Prepare / start second review' above."
            )
          )
        )

      } else if (
        state$stage ==
          "second"
      ) {

        return(
          tagList(

            div(
              class = "chip-title",
              "Second review complete"
            ),


            div(
              class = "small-line",
              "No unreviewed Stage-2 objects remain. Prepare Stage 3 above."
            )
          )
        )

      } else {

        return(
          tagList(

            div(
              class = "chip-title",
              "Stage 3 complete"
            ),


            div(
              class = "small-line",
              "All retained positive objects have been checked."
            )
          )
        )
      }
    }


    if (
      state$stage ==
        "first"
    ) {

      current <- current_first_row()


      elements <- list(

        div(
          class = "chip-title",
          current$chip_id
        ),


        div(
          class = "small-line",

          paste(
            "Candidate predictions:",
            current$n_objects
          )
        )
      )

    } else if (
      state$stage ==
        "second"
    ) {

      current <- current_second_row()


      elements <- list(

        div(
          class = "chip-title",
          "Second-review object"
        ),


        div(
          class = "small-line",

          paste0(
            "Object ID: ",
            current$candidate_id
          )
        ),


        div(
          class = "small-line",

          paste0(
            "Predicted area: ",
            round(
              as.numeric(
                current$area_m2
              ),
              2
            ),
            " m²"
          )
        ),


        div(
          class = "small-line",

          paste0(
            "First-stage reason: ",
            current$stage1_reason
          )
        ),


        div(
          class = "target-banner",
          "Review only the object inside the YELLOW bounding box. The box is transparent and does not colour the target."
        )
      )

    } else {

      current <- current_third_row()


      elements <- list(

        div(
          class = "chip-title",
          "Stage-3 retained object"
        ),


        div(
          class = "small-line",

          paste0(
            "Object ID: ",
            current$candidate_id
          )
        ),


        div(
          class = "small-line",

          paste0(
            "Area: ",
            round(
              as.numeric(
                current$area_m2
              ),
              2
            ),
            " m²"
          )
        ),


        div(
          class = "small-line",

          paste0(
            "Entered Stage 3 from: ",
            ifelse(
              current$review_src ==
                "stage1",
              "Stage 1 retained object",
              "Stage 2 confirmed waste"
            )
          )
        ),


        div(
          class = "target-banner",
          "FINAL POSITIVE AUDIT: inspect the object inside the YELLOW box and decide whether it should remain in the final waste inventory."
        )
      )
    }


    if (
      identical(
        state$mode,
        "resolve"
      )
    ) {

      elements <- c(
        elements,
        list(

          div(
            class = "correction-banner",
            "RESOLUTION MODE — this object was previously marked Not sure."
          )
        )
      )
    }


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

            paste0(
              "CORRECTION MODE — current classification: ",
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
  # CLASSIFICATION BUTTONS
  # ==========================================================

  output$classification_ui <- renderUI({

    if (
      state$stage ==
        "first"
    ) {

      return(
        tagList(

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
            "mark_not_sure",
            "4  Not sure",
            class = "classification-button btn-unsure"
          ),


          actionButton(
            "skip_btn",
            "S  Skip",
            width = "100%"
          )
        )
      )
    }


    if (
      state$stage ==
        "third"
    ) {

      return(
        tagList(

          h4(
            "Stage-3 final audit"
          ),


          actionButton(
            "mark_waste",
            "1  Keep",
            class = "classification-button btn-waste"
          ),


          actionButton(
            "mark_not_waste",
            "2  Delete",
            class = "classification-button btn-nowaste"
          ),


          actionButton(
            "mark_mixed",
            "3  Needs edit",
            class = "classification-button btn-mixed"
          ),


          actionButton(
            "mark_not_sure",
            "4  Not sure",
            class = "classification-button btn-unsure"
          ),


          actionButton(
            "skip_btn",
            "S  Skip",
            width = "100%"
          )
        )
      )
    }


    if (
      state$mode ==
        "resolve"
    ) {

      return(
        tagList(

          h4(
            "Resolve this object"
          ),

          div(
            class = "small-line",
            "Choose a final decision: Waste or Not waste."
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
            "skip_btn",
            "S  Skip for now",
            width = "100%"
          )
        )
      )
    }


    tagList(

      h4(
        "Second-review classification"
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
        "mark_not_sure",
        "3  Not sure",
        class = "classification-button btn-unsure"
      ),


      actionButton(
        "skip_btn",
        "S  Skip",
        width = "100%"
      )
    )
  })


  # ==========================================================
  # CORRECTION PANEL
  # ==========================================================

  output$correction_ui <- renderUI({

    stage_text <- if (
      state$stage ==
        "first"
    ) {
      "Correct first-stage reviews"
    } else if (
      state$stage ==
        "second"
    ) {
      "Correct second-stage reviews"
    } else {
      "Correct Stage-3 reviews"
    }


    base <- list(

      h4(
        stage_text
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
      )
    )


    if (
      identical(
        state$mode,
        "history"
      )
    ) {

      n_history <- length(
        state$history_ids
      )


      idx <- state$history_index


      base <- c(
        base,
        list(

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
              "← Newer"
            ),


            actionButton(
              "history_older",
              "Older →"
            )
          ),


          div(
            style = "height:5px;"
          ),


          actionButton(
            "return_to_queue",
            if (
              state$stage ==
                "first"
            ) {
              "Return to unreviewed locations"
            } else if (
              state$stage ==
                "second"
            ) {
              "Return to unreviewed Stage-2 objects"
            } else {
              "Return to unchecked Stage-3 objects"
            },
            width = "100%"
          )
        )
      )
    }


    tagList(
      base
    )
  })


  # ==========================================================
  # FIRST-STAGE HISTORY MODE
  # ==========================================================

  start_first_history <- function(
    index = 1L
  ) {

    history <- get_first_reviewer_history(
      annotator_name()
    )


    if (nrow(history) == 0) {

      showNotification(
        "You do not yet have any first-stage reviews to browse.",
        type = "warning",
        duration = 5
      )

      return(
        invisible(FALSE)
      )
    }


    release_current_claim()


    state$history_ids <- as.character(
      history$chip_id
    )


    state$history_index <- max(
      1L,
      min(
        as.integer(index),
        length(
          state$history_ids
        )
      )
    )


    open_history_index(
      state$history_index
    )
  }


  # ==========================================================
  # SECOND-STAGE HISTORY MODE
  # ==========================================================

  start_second_history <- function(
    index = 1L
  ) {

    meta <- second_meta_rv()


    if (is.null(meta)) {

      showNotification(
        "Prepare the second-review pool first.",
        type = "warning",
        duration = 5
      )

      return(
        invisible(FALSE)
      )
    }


    history <- get_second_reviewer_history(
      annotator_name(),
      meta$candidate_id
    )


    if (nrow(history) == 0) {

      showNotification(
        "You do not yet have any second-stage reviews to browse.",
        type = "warning",
        duration = 5
      )

      return(
        invisible(FALSE)
      )
    }


    release_current_claim()


    state$history_ids <- as.character(
      history$candidate_id
    )


    state$history_index <- max(
      1L,
      min(
        as.integer(index),
        length(
          state$history_ids
        )
      )
    )


    open_history_index(
      state$history_index
    )
  }



  # ==========================================================
  # THIRD-STAGE HISTORY MODE
  # ==========================================================

  start_third_history <- function(
    index = 1L
  ) {

    meta <- third_meta_rv()


    if (is.null(meta)) {

      showNotification(
        "Prepare the Stage-3 pool first.",
        type = "warning",
        duration = 5
      )

      return(
        invisible(FALSE)
      )
    }


    history <- get_third_reviewer_history(
      annotator_name(),
      meta$candidate_id
    )


    if (nrow(history) == 0) {

      showNotification(
        "You do not yet have any Stage-3 reviews to browse.",
        type = "warning",
        duration = 5
      )

      return(
        invisible(FALSE)
      )
    }


    release_current_claim()


    state$history_ids <- as.character(
      history$candidate_id
    )


    state$history_index <- max(
      1L,
      min(
        as.integer(index),
        length(
          state$history_ids
        )
      )
    )


    open_history_index(
      state$history_index
    )
  }


  # ==========================================================
  # OPEN HISTORY INDEX
  # ==========================================================

  open_history_index <- function(
    index
  ) {

    if (
      length(
        state$history_ids
      ) == 0
    ) {
      return(
        invisible(FALSE)
      )
    }


    index <- max(
      1L,
      min(
        as.integer(index),
        length(
          state$history_ids
        )
      )
    )


    if (
      identical(
        state$mode,
        "history"
      ) &&
      !is.null(
        state$current_id
      )
    ) {

      release_current_claim()
    }


    requested_id <- state$history_ids[
      index
    ]


    if (
      state$stage ==
        "first"
    ) {

      result <- claim_first_reviewed(
        requested_id,
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

      label <- existing$label[1]

      notes <- existing$notes[1]

    } else if (
      state$stage ==
        "second"
    ) {

      meta <- second_meta_rv()


      result <- claim_second_reviewed(
        requested_id,
        annotator_name(),
        meta$candidate_id
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


      existing <- result$review

      label <- existing$label[1]

      notes <- existing$notes[1]

    } else {

      meta <- third_meta_rv()


      result <- claim_third_reviewed(
        requested_id,
        annotator_name(),
        meta$candidate_id
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


      existing <- result$review

      label <- existing$label[1]

      notes <- existing$notes[1]
    }


    state$current_id <- requested_id

    state$mode <- "history"

    state$history_index <- index

    state$previous_label <- label


    updateTextAreaInput(
      session,
      "notes",
      value = ifelse(
        is.na(notes),
        "",
        notes
      )
    )


    invisible(TRUE)
  }


  # ==========================================================
  # REFRESH SAME HISTORY ITEM AFTER CORRECTION
  # ==========================================================

  refresh_history_current <- function() {

    if (
      !identical(
        state$mode,
        "history"
      )
    ) {
      return(
        invisible(FALSE)
      )
    }


    open_history_index(
      state$history_index
    )
  }


  # ==========================================================
  # HISTORY BUTTON EVENTS
  # ==========================================================

  observeEvent(
    input$edit_last,
    {

      req(
        annotator_name()
      )


      if (
        state$stage ==
          "first"
      ) {

        start_first_history(
          1L
        )

      } else if (
        state$stage ==
          "second"
      ) {

        start_second_history(
          1L
        )

      } else {

        start_third_history(
          1L
        )
      }
    }
  )


  observeEvent(
    input$browse_reviews,
    {

      req(
        annotator_name()
      )


      if (
        state$stage ==
          "first"
      ) {

        start_first_history(
          1L
        )

      } else if (
        state$stage ==
          "second"
      ) {

        start_second_history(
          1L
        )

      } else {

        start_third_history(
          1L
        )
      }
    }
  )


  observeEvent(
    input$history_newer,
    {

      if (
        identical(
          state$mode,
          "history"
        ) &&
        state$history_index > 1L
      ) {

        open_history_index(
          state$history_index -
            1L
        )
      }
    }
  )


  observeEvent(
    input$history_older,
    {

      if (
        identical(
          state$mode,
          "history"
        ) &&
        state$history_index <
          length(
            state$history_ids
          )
      ) {

        open_history_index(
          state$history_index +
            1L
        )
      }
    }
  )


  observeEvent(
    input$return_to_queue,
    {

      req(
        annotator_name()
      )


      release_current_claim()


      if (
        state$stage ==
          "first"
      ) {

        load_next_first()

      } else if (
        state$stage ==
          "second"
      ) {

        load_next_second()

      } else {

        load_next_third()
      }
    }
  )


  # ==========================================================
  # CURRENT IMAGE
  # ==========================================================

  output$current_image <- renderUI({

    if (
      is.null(
        state$current_id
      )
    ) {

      text <- if (
        state$stage ==
          "first"
      ) {

        "First review complete."

      } else if (
        state$stage ==
          "second"
      ) {

        "Second review complete."

      } else {

        "Stage 3 complete."
      }


      return(
        div(
          style = "
            color:white;
            font-size:24px;
            text-align:center;
          ",
          text
        )
      )
    }


    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    if (
      state$stage ==
        "first"
    ) {

      current <- current_first_row()


      image_url <- paste0(
        "review_chips/",
        URLencode(
          current$image_file,
          reserved = TRUE
        )
      )


      preload_tags <- NULL


      if (
        state$mode ==
          "queue"
      ) {

        next_ids <- peek_next_first(
          n = N_PRELOAD,
          exclude_ids = c(
            state$current_id,
            first_skipped()
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
                class = "preload-image",
                loading = "eager",
                fetchpriority = "high"
              )
            }
          )
        }
      }


      return(
        tagList(

          div(

            class = "image-stage",


            tags$img(
              src = image_url,
              class = "review-image",
              loading = "eager",
              fetchpriority = "high",
              alt = current$chip_id
            )
          ),


          preload_tags
        )
      )
    }


    # --------------------------------------------------------
    # STAGE 2 / STAGE 3 OBJECT REVIEW
    # --------------------------------------------------------

    current <- if (
      state$stage ==
        "second"
    ) {

      current_second_row()

    } else {

      current_third_row()
    }


    image_url <- paste0(
      "review_chips/",
      URLencode(
        current$base_image_file,
        reserved = TRUE
      )
    )


    target_box <- list(
      left = as.numeric(current$box_left_pct),
      top = as.numeric(current$box_top_pct),
      width = as.numeric(current$box_width_pct),
      height = as.numeric(current$box_height_pct)
    )


    preload_tags <- NULL


    if (
      state$mode ==
        "queue" ||
      (
        state$stage ==
          "second" &&
        state$mode ==
          "resolve"
      )
    ) {

      if (
        state$stage ==
          "second"
      ) {

        meta <- second_meta_rv()


        next_ids <- if (
          state$mode ==
            "resolve"
        ) {

          peek_next_unresolved(
            active_ids = meta$candidate_id,
            n = N_PRELOAD_SECOND,
            exclude_ids = c(
              state$current_id,
              resolution_skipped()
            )
          )

        } else {

          peek_next_second(
            active_ids = meta$candidate_id,
            n = N_PRELOAD_SECOND,
            exclude_ids = c(
              state$current_id,
              second_skipped()
            )
          )
        }

      } else {

        meta <- third_meta_rv()


        next_ids <- peek_next_third(
          active_ids = meta$candidate_id,
          n = 5,
          exclude_ids = c(
            state$current_id,
            third_skipped()
          )
        )
      }


      if (
        length(next_ids) > 0
      ) {

        next_rows <- meta[
          match(
            next_ids,
            meta$candidate_id
          ),
          ,
          drop = FALSE
        ]


        next_files <- unique(
          next_rows$base_image_file
        )


        preload_urls <- paste0(
          "review_chips/",
          vapply(
            next_files,
            URLencode,
            character(1),
            reserved = TRUE
          )
        )


        preload_tags <- tagList(

          lapply(
            preload_urls,
            function(url) {

              tags$img(
                src = url,
                class = "preload-image",
                loading = "eager",
                fetchpriority = "high"
              )
            }
          ),


          tags$script(
            HTML(
              paste0(
                "(function(){",
                paste0(
                  "var p=new Image();p.src='",
                  preload_urls,
                  "';",
                  collapse = ""
                ),
                "})();"
              )
            )
          )
        )
      }
    }


    target_box_tag <- tags$div(

      id = "stage2-target-box",

      class = "stage2-target-box",

      `data-left` = sprintf(
        "%.6f",
        target_box$left
      ),

      `data-top` = sprintf(
        "%.6f",
        target_box$top
      ),

      `data-width` = sprintf(
        "%.6f",
        target_box$width
      ),

      `data-height` = sprintf(
        "%.6f",
        target_box$height
      ),

      style = sprintf(
        paste0(
          "left:%.6f%%;",
          "top:%.6f%%;",
          "width:%.6f%%;",
          "height:%.6f%%;",
          "visibility:visible;"
        ),
        target_box$left,
        target_box$top,
        target_box$width,
        target_box$height
      )
    )


    tagList(

      div(

        class = "image-stage",


        tags$img(
          id = "stage2-review-image",
          src = image_url,
          class = "review-image",
          loading = "eager",
          fetchpriority = "high",
          onload = "window.requestAnimationFrame(positionStage2TargetBox); setTimeout(positionStage2TargetBox, 60);",
          alt = paste0(
            if (
              state$stage ==
                "second"
            ) {
              "Second-review object "
            } else {
              "Stage-3 audit object "
            },
            current$candidate_id
          )
        ),


        target_box_tag
      ),


      preload_tags
    )
  })


  # ==========================================================
  # IMAGE HEADER
  # ==========================================================

  output$image_title <- renderText({

    if (
      state$stage ==
        "first"
    ) {

      "Stage 1: review location"

    } else if (
      state$stage ==
        "second"
    ) {

      if (
        state$mode ==
          "resolve"
      ) {
        "Stage 2: resolve previous Not sure object"
      } else {
        "Stage 2: review object inside yellow bounding box"
      }

    } else {

      "Stage 3: final positive audit"
    }
  })


  output$image_counter <- renderText({

    if (
      state$stage ==
        "first"
    ) {

      s <- first_stats_tick()

      if (
        state$mode ==
          "history"
      ) {

        paste0(
          "Correction ",
          state$history_index,
          " / ",
          length(
            state$history_ids
          )
        )

      } else {

        paste0(
          "Reviewed ",
          s$reviewed,
          " / ",
          TOTAL_CHIPS
        )
      }

    } else if (
      state$stage ==
        "second"
    ) {

      s <- second_stats_tick()

      if (
        state$mode ==
          "history"
      ) {

        paste0(
          "Correction ",
          state$history_index,
          " / ",
          length(
            state$history_ids
          )
        )

      } else {

        paste0(
          "Reviewed ",
          s$reviewed,
          " / ",
          s$total
        )
      }

    } else {

      s <- third_stats_tick()

      if (
        state$mode ==
          "history"
      ) {

        paste0(
          "Correction ",
          state$history_index,
          " / ",
          length(
            state$history_ids
          )
        )

      } else {

        paste0(
          "Checked ",
          s$reviewed,
          " / ",
          s$total
        )
      }
    }
  })


  # ==========================================================
  # SAVE CLASSIFICATION
  # ==========================================================

  classify_current <- function(
    label
  ) {

    req(
      annotator_name()
    )


    current_id <- state$current_id


    if (is.null(current_id)) {
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


    # --------------------------------------------------------
    # Stage 1 save
    # --------------------------------------------------------

    if (
      state$stage ==
        "first"
    ) {

      if (
        !label %in%
          c(
            "waste",
            "not_waste",
            "mixed",
            "not_sure"
          )
      ) {
        return(
          invisible(FALSE)
        )
      }


      saved <- tryCatch(

        save_first_annotation(
          chip_id = current_id,
          label = label,
          annotator = annotator_name(),
          notes = note_text
        ),

        error = function(e) {

          showNotification(
            paste0(
              "Could not save first-stage classification: ",
              conditionMessage(e)
            ),
            type = "error",
            duration = 8
          )

          FALSE
        }
      )


      if (!isTRUE(saved)) {
        return(
          invisible(FALSE)
        )
      }


      # Any first-stage change can alter the stage-2 pool.
      if (second_ready()) {

        second_ready(
          FALSE
        )

        second_meta_rv(
          NULL
        )

        second_sf_rv(
          NULL
        )

        second_grid_rv(
          NULL
        )
      }


      third_ready(
        FALSE
      )

      third_meta_rv(
        NULL
      )

      third_sf_rv(
        NULL
      )

      third_grid_rv(
        NULL
      )


      provisional_object_count(
        NA_integer_
      )


      object_count_time(
        NULL
      )


      if (correction_mode) {

        showNotification(
          paste0(
            "Changed from ",
            old_label,
            " to ",
            label,
            ". Second-review pool must be refreshed before continuing stage 2."
          ),
          type = "message",
          duration = 6
        )


        refresh_history_current()


      } else {

        load_next_first()
      }


      return(
        invisible(TRUE)
      )
    }


    # --------------------------------------------------------
    # Stage 3 save
    # --------------------------------------------------------

    if (
      state$stage ==
        "third"
    ) {

      third_label <- switch(
        label,
        "waste" = "keep",
        "not_waste" = "delete",
        "mixed" = "needs_edit",
        "not_sure" = "not_sure",
        NA_character_
      )


      if (is.na(third_label)) {
        return(
          invisible(FALSE)
        )
      }


      saved <- tryCatch(

        save_third_review(
          candidate_id = current_id,
          label = third_label,
          annotator = annotator_name(),
          notes = note_text
        ),

        error = function(e) {

          showNotification(
            paste0(
              "Could not save Stage-3 classification: ",
              conditionMessage(e)
            ),
            type = "error",
            duration = 8
          )

          FALSE
        }
      )


      if (!isTRUE(saved)) {
        return(
          invisible(FALSE)
        )
      }


      if (correction_mode) {

        showNotification(
          paste0(
            "Stage-3 decision changed from ",
            old_label,
            " to ",
            third_label,
            "."
          ),
          type = "message",
          duration = 4
        )


        refresh_history_current()

      } else {

        load_next_third()
      }


      return(
        invisible(TRUE)
      )
    }


    # --------------------------------------------------------
    # Stage 2 save
    # --------------------------------------------------------

    if (
      !label %in%
        c(
          "waste",
          "not_waste",
          "not_sure"
        )
    ) {
      return(
        invisible(FALSE)
      )
    }


    saved <- tryCatch(

      save_second_review(
        candidate_id = current_id,
        label = label,
        annotator = annotator_name(),
        notes = note_text
      ),

      error = function(e) {

        showNotification(
          paste0(
            "Could not save second-stage classification: ",
            conditionMessage(e)
          ),
          type = "error",
          duration = 8
        )

        FALSE
      }
    )


    if (!isTRUE(saved)) {
      return(
        invisible(FALSE)
      )
    }


    if (third_ready()) {

      third_ready(
        FALSE
      )

      third_meta_rv(
        NULL
      )

      third_sf_rv(
        NULL
      )

      third_grid_rv(
        NULL
      )
    }


    if (
      state$mode ==
        "resolve"
    ) {

      load_next_unresolved()


    } else if (correction_mode) {

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


    } else {

      load_next_second()
    }


    invisible(TRUE)
  }


  # ==========================================================
  # CLASSIFICATION EVENTS
  # ==========================================================

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

      if (
        state$stage %in%
          c(
            "first",
            "third"
          )
      ) {

        classify_current(
          "mixed"
        )
      }
    }
  )


  observeEvent(
    input$mark_not_sure,
    {

      classify_current(
        "not_sure"
      )
    }
  )


  # ==========================================================
  # SKIP
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

        # In correction mode, S simply advances to the next
        # older review without changing the current decision.
        if (
          state$history_index <
            length(
              state$history_ids
            )
        ) {

          open_history_index(
            state$history_index +
              1L
          )

        } else {

          showNotification(
            "You are at the oldest review in this browsing session.",
            type = "message",
            duration = 4
          )
        }


        return()
      }


      current_id <- state$current_id


      if (is.null(current_id)) {
        return()
      }


      if (
        state$stage ==
          "first"
      ) {

        first_skipped(
          unique(
            c(
              first_skipped(),
              current_id
            )
          )
        )


        release_first_claim(
          current_id,
          annotator_name()
        )


        load_next_first()


      } else if (
        state$stage ==
          "second"
      ) {

        if (
          state$mode ==
            "resolve"
        ) {

          resolution_skipped(
            unique(
              c(
                resolution_skipped(),
                current_id
              )
            )
          )


          release_second_claim(
            current_id,
            annotator_name()
          )


          load_next_unresolved()


        } else {

          second_skipped(
            unique(
              c(
                second_skipped(),
                current_id
              )
            )
          )


          release_second_claim(
            current_id,
            annotator_name()
          )


          load_next_second()
        }

      } else {

        third_skipped(
          unique(
            c(
              third_skipped(),
              current_id
            )
          )
        )


        release_third_claim(
          current_id,
          annotator_name()
        )


        load_next_third()
      }
    }
  )


  # ==========================================================
  # PROGRESS PANEL
  # ==========================================================

  output$progress_ui <- renderUI({

    if (
      state$stage ==
        "first"
    ) {

      s <- first_stats_tick()


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


      return(
        tagList(

          h4(
            "Stage 1 progress"
          ),


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
              " locations"
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
          ),


          div(
            class = "metric",

            paste(
              "Not sure:",
              s$not_sure
            )
          ),


          div(
            style = "height:5px;"
          ),


          actionButton(
            "update_object_count",
            "Update provisional object count",
            width = "100%"
          ),


          textOutput(
            "object_count_text"
          )
        )
      )
    }


    if (
      state$stage ==
        "second"
    ) {

      s <- second_stats_tick()


      remaining <- s$total -
        s$reviewed


      pct <- if (
        s$total > 0
      ) {

        round(
          100 *
            s$reviewed /
            s$total,
          1
        )

      } else {

        100
      }


      return(
        tagList(

          h4(
            "Stage 2 progress"
          ),


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
              s$total,
              " objects"
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
              "Not sure:",
              s$not_sure
            )
          )
        )
      )
    }


    s <- third_stats_tick()


    remaining <- s$total -
      s$reviewed


    pct <- if (
      s$total > 0
    ) {

      round(
        100 *
          s$reviewed /
          s$total,
        1
      )

    } else {

      100
    }


    tagList(

      h4(
        "Stage 3 progress"
      ),


      div(
        class = "metric",

        strong(
          paste0(
            pct,
            "% checked"
          )
        )
      ),


      div(
        class = "metric",

        paste0(
          s$reviewed,
          " / ",
          s$total,
          " retained objects"
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
          "Keep:",
          s$keep
        )
      ),


      div(
        class = "metric",
        paste(
          "Delete:",
          s$delete
        )
      ),


      div(
        class = "metric",
        paste(
          "Needs edit:",
          s$needs_edit
        )
      ),


      div(
        class = "metric",
        paste(
          "Not sure:",
          s$not_sure
        )
      )
    )
  })


  # ==========================================================
  # PROVISIONAL OBJECT COUNT — FIRST STAGE ONLY
  # ==========================================================

  calculate_remaining_first_objects <- function() {

    spatial_state <- build_spatial_state()


    sum(
      spatial_state$predictions$stage1_class %in%
        c(
          "retained",
          "second_review",
          "pending"
        ) &
        spatial_state$predictions$area_m2 >=
          MIN_SECOND_REVIEW_AREA_M2
    )
  }


  observeEvent(
    input$update_object_count,
    {

      if (
        state$stage !=
          "first"
      ) {
        return()
      }


      showNotification(
        "Calculating provisional prediction count...",
        type = "message",
        duration = 2
      )


      count <- calculate_remaining_first_objects()


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


    if (is.na(count)) {

      return(
        paste0(
          "Objects ≥ ",
          MIN_SECOND_REVIEW_AREA_M2,
          " m²: click to recalculate."
        )
      )
    }


    paste0(
      "Provisional objects not rejected: ",
      format(
        count,
        big.mark = ","
      ),
      "\nUpdated ",
      object_count_time()
    )
  })


  # ==========================================================
  # LEADERBOARD
  # ==========================================================

  output$leaderboard_table <- renderTable(
    {

      first_stats_tick()$board
    },

    striped = TRUE,
    spacing = "xs"
  )



  # ==========================================================
  # FINAL VERIFIED WASTE SUMMARY
  # ==========================================================

  observeEvent(
    input$summarise_final,
    {

      result <- tryCatch(

        withProgress(
          message = "Summarising verified waste piles",
          value = 0,
          {

            incProgress(
              0.25,
              detail = "Combining retained Stage-1 and Stage-2 waste..."
            )


            out <- export_all_results()


            incProgress(
              0.75,
              detail = "Calculating pile-size statistics..."
            )


            out
          }
        ),

        error = function(e) {

          showNotification(
            paste0(
              "Could not summarise final waste piles: ",
              conditionMessage(e)
            ),
            type = "error",
            duration = 10
          )

          NULL
        }
      )


      if (is.null(result)) {
        return()
      }


      pile_summary_rv(
        result$pile_summary
      )


      pile_size_rv(
        result$size_table
      )


      showModal(
        modalDialog(

          title = "Final verified waste piles",

          size = "l",

          easyClose = TRUE,

          div(
            class = "small-line",
            "Stage 3 audits every object retained after Stages 1 and 2. Only objects classified as Keep in Stage 3 enter the final verified inventory; Delete, Needs edit, Not sure and unreviewed objects are excluded from final pile statistics."
          ),

          h4(
            "Summary"
          ),

          tableOutput(
            "pile_summary_table"
          ),

          h4(
            "Pile-size distribution"
          ),

          tableOutput(
            "pile_size_table"
          ),

          footer = modalButton(
            "Close"
          )
        )
      )
    }
  )


  output$pile_summary_table <- renderTable(
    {

      x <- pile_summary_rv()

      req(
        !is.null(x)
      )


      out <- x

      out$value <- ifelse(
        grepl(
          "piles|objects",
          out$statistic,
          ignore.case = TRUE
        ),
        format(
          round(
            out$value
          ),
          big.mark = ",",
          scientific = FALSE
        ),
        format(
          round(
            out$value,
            3
          ),
          big.mark = ",",
          scientific = FALSE
        )
      )

      out
    },

    striped = TRUE,
    spacing = "s"
  )


  output$pile_size_table <- renderTable(
    {

      x <- pile_size_rv()

      req(
        !is.null(x)
      )


      if (nrow(x) == 0) {
        return(x)
      }


      x$percentage <- round(
        x$percentage,
        2
      )

      x$total_area_m2 <- round(
        x$total_area_m2,
        2
      )

      x
    },

    striped = TRUE,
    spacing = "s"
  )


  # ==========================================================
  # DOWNLOAD FINAL VERIFIED WASTE SHAPEFILE
  # ==========================================================

  output$download_verified_shp <- downloadHandler(

    filename = function() {

      "final_verified_waste_piles_shapefile.zip"
    },

    content = function(file) {

      result <- export_all_results()


      if (
        !file.exists(
          result$shapefile_zip
        )
      ) {

        stop(
          "The verified waste shapefile could not be created."
        )
      }


      ok <- file.copy(
        result$shapefile_zip,
        file,
        overwrite = TRUE
      )


      if (!isTRUE(ok)) {

        stop(
          "Could not copy the verified waste shapefile to the download."
        )
      }
    },

    contentType = "application/zip"
  )


  # ==========================================================
  # EXPORT
  # ==========================================================

  observeEvent(
    input$export_results,
    {

      export_message(
        "Exporting..."
      )


      showNotification(
        "Building complete first- and second-stage spatial export...",
        type = "message",
        duration = 3
      )


      result <- tryCatch(

        withProgress(
          message = "Exporting review results",
          value = 0,
          {

            incProgress(
              0.2,
              detail = "Rebuilding current spatial classifications..."
            )


            out <- export_all_results()


            incProgress(
              0.8,
              detail = "Writing final inventory..."
            )


            out
          }
        ),

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


      if (is.null(result)) {
        return()
      }


      s <- result$summary


      get_value <- function(name) {

        x <- s$value[
          s$measure ==
            name
        ]


        if (length(x) == 0) {
          return(0)
        }


        x[1]
      }


      export_message(
        paste0(
          "Export complete\n\n",

          "First-stage retained locations: ",
          get_value("retained_locations"),
          "\n",

          "Mixed locations: ",
          get_value("mixed_locations"),
          "\n",

          "Not-sure locations: ",
          get_value("not_sure_locations"),
          "\n",

          "Rejected locations: ",
          get_value("rejected_locations"),
          "\n\n",

          "Second-review candidates: ",
          get_value("active_second_review_candidates"),
          "\n",

          "Second-review waste: ",
          get_value("second_review_waste"),
          "\n",

          "Second-review not waste: ",
          get_value("second_review_not_waste"),
          "\n",

          "Second-review unresolved: ",
          get_value("second_review_not_sure") +
            get_value("second_review_unreviewed"),
          "\n\n",

          "STAGE-1 OBJECTS ENTERING STAGE 3: ",
          get_value("stage1_verified_objects"),
          "\n",

          "STAGE-2 OBJECTS ENTERING STAGE 3: ",
          get_value("stage2_verified_objects"),
          "\n",

          "TOTAL STAGE-3 CANDIDATES: ",
          get_value("stage3_candidate_objects"),
          "\n",

          "STAGE-3 KEEP: ",
          get_value("stage3_kept_objects"),
          "\n",

          "STAGE-3 DELETE: ",
          get_value("stage3_deleted_objects"),
          "\n",

          "STAGE-3 NEEDS EDIT: ",
          get_value("stage3_needs_edit_objects"),
          "\n",

          "STAGE-3 UNRESOLVED: ",
          get_value("stage3_unresolved_objects"),
          "\n",

          "FINAL VERIFIED OBJECTS: ",
          get_value("final_verified_prediction_polygons"),
          "\n",

          "FINAL CONTIGUOUS WASTE PILES: ",
          get_value("final_verified_contiguous_units"),
          "\n",

          "FINAL VERIFIED AREA (m²): ",
          round(
            get_value("final_verified_total_area_m2"),
            2
          ),
          "\n",

          "MEDIAN PILE AREA (m²): ",
          round(
            get_value("final_verified_median_area_m2"),
            3
          ),
          "\n",

          "MAXIMUM PILE AREA (m²): ",
          round(
            get_value("final_verified_max_area_m2"),
            2
          ),
          "\n",

          "RETAINED OBJECTS TOUCHING STAGE-1 REJECTED CELLS: ",
          get_value("final_retained_touching_stage1_rejected_cells"),
          "\n\n",

          "Verified waste shapefiles (objects + contiguous piles):\n",
          FINAL_SHP_ZIP,
          "\n\n",

          "Pile summary:\n",
          FINAL_PILE_SUMMARY_CSV,
          "\n\n",

          "Decision audit:\n",
          FINAL_AUDIT_CSV,
          "\n\n",

          "Saved to:\n",
          EXPORT_DIR
        )
      )


      showNotification(
        "Complete spatial export finished.",
        type = "message",
        duration = 6
      )
    }
  )


  output$export_status <- renderText({

    export_message()
  })

}


# ============================================================
# 16. RUN APP
# ============================================================

shinyApp(
  ui = ui,
  server = server
)
