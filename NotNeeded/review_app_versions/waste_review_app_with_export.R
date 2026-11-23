# ============================================================
# FAST WASTE PREDICTION REVIEW APP
# SQLite-backed, image-only Shiny reviewer
#
# Designed for pre-generated PNG review chips.
#
# Expected folder structure:
#
# waste_review_app/
# ├── app.R
# ├── review_log.sqlite          # created automatically
# ├── data/
# │   ├── chips.csv
# │   └── object_chip_map.csv
# └── www/
#     └── chips/
#         ├── C00001.png
#         ├── C00002.png
#         └── ...
#
# chips.csv requires:
#   chip_id, image_file, n_objects
#
# object_chip_map.csv requires:
#   candidate_id, chip_id
#
# Review classes:
#   waste
#   not_waste
#   mixed
#
# Performance design:
#   - no terra
#   - no sf
#   - no raster cropping inside Shiny
#   - images are served directly from www/chips
#   - next images are preloaded in the browser
#   - SQLite writes are small and transactional
#   - expensive remaining-object calculation runs only on demand
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
# 1. CONFIG
# ============================================================

BASE_PATH <- file.path(
  Sys.getenv("USERPROFILE"),
  "Documents",
  "GitHub",
  "mapping-waste-urban-africa"
)

BASE_DIR <- file.path(
  BASE_PATH,
  "05_finetuning",
  "waste_review_app"
)

DATA_DIR <- file.path(
  BASE_DIR,
  "data"
)

CHIPS_DIR <- file.path(
  BASE_DIR,
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
  BASE_DIR,
  "review_log.sqlite"
)

EXPORT_DIR <- file.path(
  BASE_DIR,
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

# If somebody closes the browser while holding a chip,
# the chip is released after this period.
CLAIM_TIMEOUT_MINUTES <- 30

# Keep the current claim alive while the browser session is open.
CLAIM_HEARTBEAT_SECONDS <- 120

# Number of likely upcoming images to preload invisibly.
N_PRELOAD <- 3

# Refresh simple review statistics every 5 seconds.
STATS_REFRESH_MS <- 5000


# ============================================================
# 2. CHECK INPUTS
# ============================================================

if (!dir.exists(CHIPS_DIR)) {
  stop(
    "Could not find:\n",
    CHIPS_DIR,
    "\n\nRun this app from the waste_review_app folder."
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

# Explicitly expose the prepared PNG folder to the browser.
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
# 3. LOAD PREPARED CHIP INFORMATION
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
    "chips.csv is missing: ",
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
    "object_chip_map.csv is missing: ",
    paste(
      missing_map_columns,
      collapse = ", "
    )
  )
}


# Check image existence once at startup.
image_paths <- file.path(
  CHIPS_DIR,
  chips$image_file
)

missing_images <- chips$image_file[
  !file.exists(image_paths)
]

if (length(missing_images) > 0) {
  stop(
    "Some PNG files listed in chips.csv are missing.\n",
    "Examples:\n",
    paste(
      head(missing_images, 10),
      collapse = "\n"
    )
  )
}


# If n_objects was not written, derive it.
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

TOTAL_OBJECTS <- length(
  unique(
    object_chip_map$candidate_id
  )
)


# Preserve the geographic/order sequence already stored in chips.csv.
chip_order <- setNames(
  seq_len(TOTAL_CHIPS),
  chips$chip_id
)


cat(
  "\nWaste review app\n",
  "----------------\n",
  "Review chips: ",
  format(TOTAL_CHIPS, big.mark = ","),
  "\nCandidate objects: ",
  format(TOTAL_OBJECTS, big.mark = ","),
  "\nImages: ",
  CHIPS_DIR,
  "\nDatabase: ",
  DB_PATH,
  "\n\n",
  sep = ""
)


# ============================================================
# 4. SQLITE CONNECTION
# ============================================================

db_connect <- function() {

  con <- DBI::dbConnect(
    RSQLite::SQLite(),
    DB_PATH
  )

  # WAL allows readers and writers to work concurrently.
  DBI::dbExecute(
    con,
    "PRAGMA journal_mode=WAL;"
  )

  # Wait briefly rather than immediately failing on a locked DB.
  DBI::dbExecute(
    con,
    "PRAGMA busy_timeout=5000;"
  )

  con
}


# ============================================================
# 5. INITIALISE DATABASE
# ============================================================

db_init <- function() {

  con <- db_connect()

  on.exit(
    DBI::dbDisconnect(con)
  )


  # One current decision per chip.
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


  # Complete audit history.
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


  # Temporary reservation of a chip by a reviewer.
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
}


db_init()


# ============================================================
# 6. DATABASE HELPERS
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


  if (
    is.null(annotator)
  ) {

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


# ------------------------------------------------------------
# Claim the next available chip.
#
# exclude_ids is useful for chips skipped within the same
# browser session, so "Skip" does not immediately return the
# same chip.
# ------------------------------------------------------------

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


  if (
    length(available) == 0
  ) {
    return(NULL)
  }


  available <- available[
    order(
      chip_order[
        available
      ]
    )
  ]


  # INSERT OR IGNORE protects against two reviewers trying
  # to claim the same chip at the same moment.
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


    if (
      inserted == 1
    ) {
      return(
        chip_id
      )
    }
  }


  NULL
}


# ------------------------------------------------------------
# Look ahead without claiming.
# Used only to preload likely next images in the browser.
# ------------------------------------------------------------

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


  if (
    length(available) == 0
  ) {
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


# ------------------------------------------------------------
# Save classification.
# ------------------------------------------------------------

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

      # Full audit trail
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


      # Current classification
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


      # Release chip
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


# ------------------------------------------------------------
# Lightweight statistics only.
# ------------------------------------------------------------

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
      SUM(CASE WHEN label = 'waste' THEN 1 ELSE 0 END) AS waste,
      SUM(CASE WHEN label = 'not_waste' THEN 1 ELSE 0 END) AS not_waste,
      SUM(CASE WHEN label = 'mixed' THEN 1 ELSE 0 END) AS mixed
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
    ORDER BY n_reviewed DESC,
             annotator ASC;
    "
  )


  # SQLite SUM() returns NA when table is empty.
  for (
    nm in c(
      "reviewed",
      "waste",
      "not_waste",
      "mixed"
    )
  ) {

    if (
      is.na(counts[[nm]][1])
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
# 7. PROVISIONAL OBJECT COUNT
#
# This can be relatively expensive, so it runs only when the
# reviewer explicitly clicks "Update pile count".
# ============================================================

calculate_remaining_objects <- function() {

  annotations <- get_annotations()


  if (
    nrow(annotations) == 0
  ) {
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


  # Remove a predicted object only when every chip it intersects
  # has been explicitly labelled not_waste.
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
# 8. EXPORT REVIEW RESULTS
#
# This is run only when the user presses "Export spatial results".
# sf is therefore not used while moving between review images.
#
# Spatial export logic:
#
# Location layers:
#   waste      -> retained_locations
#   mixed      -> second_review_locations
#   not_waste  -> rejected_locations
#   unreviewed -> pending_locations
#
# Object layers:
#   any unreviewed chip            -> pending
#   any mixed chip                 -> second_review
#   all waste                      -> retained
#   all not_waste                  -> rejected
#   mixture of waste + not_waste   -> second_review
#
# This conservative rule prevents a prediction crossing several
# chips from being accepted before all relevant locations have
# been resolved.
# ============================================================

write_gpkg_layer <- function(
  x,
  layer
) {

  if (nrow(x) == 0) {
    return(invisible(FALSE))
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
  # Read review grid and attach SQLite classifications
  # ----------------------------------------------------------

  review_grid <- sf::st_read(
    REVIEW_GRID_PATH,
    quiet = TRUE
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
  # Chip-level CSV audit table
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
  # Start a fresh GeoPackage
  # ----------------------------------------------------------

  if (file.exists(EXPORT_GPKG)) {
    file.remove(EXPORT_GPKG)
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
  # Determine one review class for every candidate object
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
      return("pending")
    }

    if ("mixed" %in% x) {
      return("second_review")
    }

    if (all(x == "waste")) {
      return("retained")
    }

    if (all(x == "not_waste")) {
      return("rejected")
    }

    # Conflicting waste + not_waste across adjacent chips.
    "second_review"
  }


  object_review <- vapply(
    status_by_object,
    classify_object,
    character(1)
  )


  object_class_table <- data.frame(
    candidate_id = names(object_review),
    review_class = unname(object_review),
    stringsAsFactors = FALSE
  )


  # Candidate IDs can be numeric in one file and character in another.
  object_class_table$candidate_id <- as.character(
    object_class_table$candidate_id
  )


  # ----------------------------------------------------------
  # Read the original prediction polygons
  # ----------------------------------------------------------

  predictions <- sf::st_read(
    PREDICTION_PATH,
    quiet = TRUE
  )


  # The preparation script generated candidate_id from row order
  # when the shapefile did not already contain a permanent ID.
  if (!"candidate_id" %in% names(predictions)) {
    predictions$candidate_id <- seq_len(
      nrow(predictions)
    )
  }


  predictions$candidate_id <- as.character(
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
  # Summary table
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
    stringsAsFactors = FALSE
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
# 9. UI
# ============================================================

ui <- fluidPage(


  tags$head(


    # --------------------------------------------------------
    # Keyboard shortcuts
    #
    # 1 = Waste
    # 2 = Not waste
    # 3 = Mixed
    # S = Skip
    #
    # Shortcuts are disabled while typing in a text field.
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
          icon = icon("forward"),
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

        div(
          style = "height:4px;"
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
# 9. SERVER
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
        "Keys: 1 = waste, 2 = not waste, 3 = mixed, S = skip."
      )
    )
  })


  # ----------------------------------------------------------
  # Per-session state
  # ----------------------------------------------------------

  state <- reactiveValues(
    current_chip_id = NULL
  )


  # Chips skipped by this reviewer during this browser session.
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
  # Load next chip
  # ----------------------------------------------------------

  load_next_chip <- function() {

    req(
      annotator_name()
    )


    next_id <- claim_next_chip(
      annotator = annotator_name(),
      exclude_ids = session_skipped()
    )


    # If everything still available has been skipped during this
    # session, clear the skip memory and try again.
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


    updateTextAreaInput(
      session,
      "notes",
      value = ""
    )
  }


  # Initial chip after reviewer logs in.
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
  # Keep active claim alive.
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


  # Release claim when browser closes.
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
  # 10. CURRENT CHIP
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


    tagList(

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
  })


  # ==========================================================
  # 11. CURRENT IMAGE + PRELOAD LIKELY NEXT IMAGES
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


    # Look ahead to likely next unreviewed chips.
    next_ids <- peek_next_chip_ids(
      n = N_PRELOAD,
      exclude_ids = c(
        state$current_chip_id,
        session_skipped()
      )
    )


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
            fname
          ),
          class = "preload-image"
        )
      }
    )


    tagList(

      # Visible current image
      tags$img(
        src = paste0(
          "review_chips/",
          current$image_file
        ),
        class = "review-image"
      ),


      # Hidden images force the browser to cache likely next chips.
      preload_tags
    )
  })


  # ==========================================================
  # 12. CLASSIFY
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
      isTRUE(saved)
    ) {

      # A newly completed annotation changes the pile inventory,
      # so mark the on-demand pile count as stale.
      provisional_object_count(
        NA_integer_
      )

      object_count_time(
        NULL
      )


      load_next_chip()
    }
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
  # 13. SKIP
  # ==========================================================

  observeEvent(
    input$skip_btn,
    {

      req(
        annotator_name()
      )


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
  # 14. LIGHTWEIGHT LIVE STATISTICS
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


    paste0(
      "Reviewed ",
      s$reviewed,
      " / ",
      TOTAL_CHIPS
    )
  })


  # ==========================================================
  # 15. PILE COUNT — ON DEMAND ONLY
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
  # EXPORT SPATIAL RESULTS
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
# 16. RUN APP
# ============================================================

shinyApp(
  ui = ui,
  server = server
)
