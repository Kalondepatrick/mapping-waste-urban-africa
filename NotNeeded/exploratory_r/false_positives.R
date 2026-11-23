# ============================================================
# WASTE PREDICTION REVIEW APP
#
# Review unit:
#   50 m x 50 m spatial chip
#
# Left:
#   RGB orthomosaic + predicted waste outlines
#
# Right:
#   Continuous P(waste)
#
# Review classes:
#   1. waste
#   2. not_waste
#   3. mixed
#
# Rejected predictions are retained as hard-negative examples.
# ============================================================


# ============================================================
# 0. PACKAGES
# ============================================================

required_packages <- c(
  "shiny",
  "sf",
  "terra"
)

missing_packages <- required_packages[
  !required_packages %in% rownames(installed.packages())
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(shiny)
library(sf)
library(terra)

sf_use_s2(FALSE)


# ============================================================
# 1. PATHS
# ============================================================

BASE_PATH <- file.path(
  Sys.getenv("USERPROFILE"),
  "Documents",
  "GitHub",
  "mapping-waste-urban-africa"
)


# ------------------------------------------------------------
# Original drone imagery
# ------------------------------------------------------------

ORTHO_PATH <- file.path(
  BASE_PATH,
  "data",
  "optical.tiff"
)


# ------------------------------------------------------------
# Continuous waste-probability raster
# ------------------------------------------------------------

PROBABILITY_PATH <- file.path(
  BASE_PATH,
  "outputs",
  "waste_probability_full_seamfree.tif"
)


# ------------------------------------------------------------
# Predicted waste polygons
#
# At present this uses the current full-area prediction.
# ------------------------------------------------------------

PREDICTION_PATH <- file.path(
  BASE_PATH,
  "outputs",
  "waste_piles.shp"
)


# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

OUTPUT_DIR <- file.path(
  BASE_PATH,
  "results",
  "shiny_waste_review"
)

dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)


REVIEW_CSV <- file.path(
  OUTPUT_DIR,
  "chip_reviews.csv"
)

REVIEW_GRID_GPKG <- file.path(
  OUTPUT_DIR,
  "review_grid.gpkg"
)

RETAINED_GPKG <- file.path(
  OUTPUT_DIR,
  "retained_candidate_predictions.gpkg"
)

REJECTED_GPKG <- file.path(
  OUTPUT_DIR,
  "rejected_false_positive_fragments.gpkg"
)

MIXED_GPKG <- file.path(
  OUTPUT_DIR,
  "mixed_candidate_fragments.gpkg"
)

WASTE_GPKG <- file.path(
  OUTPUT_DIR,
  "waste_chip_predictions.gpkg"
)

SUMMARY_CSV <- file.path(
  OUTPUT_DIR,
  "review_summary.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

CHIP_SIZE_M <- 50

MIN_FINAL_AREA_M2 <- 0.25

RGB_BANDS <- c(
  1,
  2,
  3
)

PROBABILITY_MIN <- 0
PROBABILITY_MAX <- 1


# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

required_files <- c(
  ORTHO_PATH,
  PROBABILITY_PATH,
  PREDICTION_PATH
)

missing_files <- required_files[
  !file.exists(required_files)
]

if (length(missing_files) > 0) {
  
  stop(
    paste(
      "Missing required files:\n",
      paste(
        missing_files,
        collapse = "\n"
      )
    )
  )
}


cat(
  "\nUsing candidate predictions:\n",
  PREDICTION_PATH,
  "\n"
)


# ============================================================
# 4. OPEN RASTERS
# ============================================================

orthomosaic <- rast(
  ORTHO_PATH
)

probability <- rast(
  PROBABILITY_PATH
)


if (nlyr(probability) != 1) {
  
  stop(
    "Probability raster must contain exactly one band."
  )
}


# ============================================================
# 5. READ PREDICTIONS
# ============================================================

predictions <- st_read(
  PREDICTION_PATH,
  quiet = TRUE
)


if (nrow(predictions) == 0) {
  
  stop(
    "Prediction file contains no polygons."
  )
}


# ------------------------------------------------------------
# Remove empty geometries
# ------------------------------------------------------------

predictions <- predictions[
  !st_is_empty(predictions),
  ,
  drop = FALSE
]


# ------------------------------------------------------------
# Repair invalid geometry
# ------------------------------------------------------------

predictions <- st_make_valid(
  predictions
)


# ------------------------------------------------------------
# Permanent candidate ID
# ------------------------------------------------------------

if (!"candidate_id" %in% names(predictions)) {
  
  predictions$candidate_id <- seq_len(
    nrow(predictions)
  )
}


# ============================================================
# 6. CRS
# ============================================================

raster_crs <- crs(
  orthomosaic,
  proj = TRUE
)


if (
  is.na(raster_crs) ||
  raster_crs == ""
) {
  
  stop(
    "Orthomosaic does not contain a CRS."
  )
}


predictions <- st_transform(
  predictions,
  raster_crs
)


if (st_is_longlat(predictions)) {
  
  stop(
    paste(
      "The prediction layer is in a geographic CRS.",
      "A projected metre-based CRS is required",
      "for 50 m x 50 m chips."
    )
  )
}


if (!same.crs(
  orthomosaic,
  probability
)) {
  
  stop(
    paste(
      "Orthomosaic and waste probability raster",
      "do not have the same CRS."
    )
  )
}


# ============================================================
# 7. PROBABILITY RANGE
# ============================================================

prob_minmax <- global(
  probability,
  fun = c(
    "min",
    "max"
  ),
  na.rm = TRUE
)


cat(
  "\nProbability raster range:\n"
)

print(
  prob_minmax
)


# ============================================================
# 8. BUILD 50 x 50 m REVIEW GRID
# ============================================================

prediction_bbox <- st_bbox(
  predictions
)


cat(
  "\nPrediction bounding box:\n"
)

print(
  prediction_bbox
)


# ------------------------------------------------------------
# IMPORTANT
#
# as.numeric() removes inherited bbox names and avoids
# xmin.xmin / ymin.ymin naming problems.
# ------------------------------------------------------------

xmin_grid <- floor(
  as.numeric(
    prediction_bbox["xmin"]
  ) / CHIP_SIZE_M
) * CHIP_SIZE_M


ymin_grid <- floor(
  as.numeric(
    prediction_bbox["ymin"]
  ) / CHIP_SIZE_M
) * CHIP_SIZE_M


xmax_grid <- ceiling(
  as.numeric(
    prediction_bbox["xmax"]
  ) / CHIP_SIZE_M
) * CHIP_SIZE_M


ymax_grid <- ceiling(
  as.numeric(
    prediction_bbox["ymax"]
  ) / CHIP_SIZE_M
) * CHIP_SIZE_M


grid_bbox <- st_bbox(
  c(
    xmin = xmin_grid,
    ymin = ymin_grid,
    xmax = xmax_grid,
    ymax = ymax_grid
  ),
  crs = st_crs(
    predictions
  )
)


cat(
  "\n50 m grid bounding box:\n"
)

print(
  grid_bbox
)


grid_all <- st_make_grid(
  st_as_sfc(
    grid_bbox
  ),
  cellsize = c(
    CHIP_SIZE_M,
    CHIP_SIZE_M
  ),
  square = TRUE,
  what = "polygons"
)


grid_all <- st_sf(
  geometry = grid_all
)


# ------------------------------------------------------------
# Keep only cells intersecting predicted waste
# ------------------------------------------------------------

cat(
  "\nFinding 50 m chips containing predictions...\n"
)


grid_has_prediction <- lengths(
  st_intersects(
    grid_all,
    predictions
  )
) > 0


chips <- grid_all[
  grid_has_prediction,
  ,
  drop = FALSE
]


# ------------------------------------------------------------
# Sort from north-west toward south-east
# ------------------------------------------------------------

chip_centroids <- suppressWarnings(
  st_centroid(
    chips
  )
)


chip_xy <- st_coordinates(
  chip_centroids
)


chips$centre_x <- chip_xy[, 1]
chips$centre_y <- chip_xy[, 2]


chips <- chips[
  order(
    -chips$centre_y,
    chips$centre_x
  ),
  ,
  drop = FALSE
]


chips$chip_id <- sprintf(
  "C%05d",
  seq_len(
    nrow(chips)
  )
)


chips <- chips[
  c(
    "chip_id",
    "centre_x",
    "centre_y",
    "geometry"
  )
]


cat(
  "\nReview chips containing predictions:",
  format(
    nrow(chips),
    big.mark = ","
  ),
  "\n"
)


# ============================================================
# 9. PRECOMPUTE OBJECT-CHIP RELATIONSHIPS
# ============================================================

cat(
  "\nCalculating object-chip relationships...\n"
)


object_chip_index <- st_intersects(
  predictions,
  chips
)


object_chip_ids <- lapply(
  object_chip_index,
  function(x) {
    
    if (length(x) == 0) {
      
      character(0)
      
    } else {
      
      chips$chip_id[x]
      
    }
  }
)


cat(
  "Done.\n"
)


# ============================================================
# 10. REVIEW TABLE
# ============================================================

review_table_initial <- data.frame(
  
  chip_id = chips$chip_id,
  
  status = "unreviewed",
  
  notes = "",
  
  reviewed_at = "",
  
  stringsAsFactors = FALSE
)


# ------------------------------------------------------------
# Resume previous work
# ------------------------------------------------------------

if (file.exists(
  REVIEW_CSV
)) {
  
  old_reviews <- read.csv(
    REVIEW_CSV,
    stringsAsFactors = FALSE
  )
  
  
  matched <- match(
    review_table_initial$chip_id,
    old_reviews$chip_id
  )
  
  
  valid_match <- !is.na(
    matched
  )
  
  
  review_table_initial$status[
    valid_match
  ] <- old_reviews$status[
    matched[
      valid_match
    ]
  ]
  
  
  review_table_initial$notes[
    valid_match
  ] <- old_reviews$notes[
    matched[
      valid_match
    ]
  ]
  
  
  review_table_initial$reviewed_at[
    valid_match
  ] <- old_reviews$reviewed_at[
    matched[
      valid_match
    ]
  ]
}


# ============================================================
# 11. HELPER FUNCTIONS
# ============================================================

save_reviews <- function(
    review_table
) {
  
  write.csv(
    review_table,
    REVIEW_CSV,
    row.names = FALSE
  )
}


# ------------------------------------------------------------
# Provisional object count
#
# An object is excluded only where all chips intersecting that
# object have been classified as not_waste.
# ------------------------------------------------------------

get_remaining_candidate_count <- function(
    review_table
) {
  
  excluded_chips <- review_table$chip_id[
    review_table$status == "not_waste"
  ]
  
  
  if (
    length(excluded_chips) == 0
  ) {
    
    return(
      nrow(predictions)
    )
  }
  
  
  keep_object <- vapply(
    
    object_chip_ids,
    
    function(chip_ids) {
      
      if (
        length(chip_ids) == 0
      ) {
        
        return(
          TRUE
        )
      }
      
      
      !all(
        chip_ids %in%
          excluded_chips
      )
    },
    
    logical(1)
  )
  
  
  sum(
    keep_object
  )
}


# ------------------------------------------------------------
# Next unreviewed chip
# ------------------------------------------------------------

find_next_unreviewed <- function(
    current_index,
    review_table
) {
  
  n <- nrow(
    chips
  )
  
  
  if (n == 0) {
    
    return(
      1
    )
  }
  
  
  search_order <- c(
    
    seq(
      current_index + 1,
      n
    ),
    
    seq(
      1,
      current_index
    )
  )
  
  
  search_order <- search_order[
    search_order >= 1 &
      search_order <= n
  ]
  
  
  for (i in search_order) {
    
    chip_id <- chips$chip_id[i]
    
    
    status <- review_table$status[
      match(
        chip_id,
        review_table$chip_id
      )
    ]
    
    
    if (
      status == "unreviewed"
    ) {
      
      return(
        i
      )
    }
  }
  
  
  current_index
}


# ------------------------------------------------------------
# Safe writer
# ------------------------------------------------------------

write_layer <- function(
    x,
    path,
    layer
) {
  
  if (
    nrow(x) == 0
  ) {
    
    cat(
      "\nNo features to write:",
      layer,
      "\n"
    )
    
    return(
      invisible(NULL)
    )
  }
  
  
  st_write(
    x,
    path,
    layer = layer,
    delete_layer = TRUE,
    quiet = TRUE
  )
}


# ============================================================
# 12. SINGLE-SCREEN USER INTERFACE
# ============================================================

ui <- fluidPage(
  
  
  tags$head(
    
    
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
          background-color: #f4f4f4;
        }


        .container-fluid {
          height: 100vh;
          overflow: hidden;
          padding: 6px 10px;
        }


        .app-title {
          height: 5vh;
          min-height: 42px;
          display: flex;
          align-items: center;
          font-size: 29px;
          font-weight: 500;
          padding-left: 4px;
          margin: 0;
        }


        .main-row {
          height: calc(95vh - 12px);
          margin: 0;
        }


        .sidebar-fixed {
          height: 100%;
          overflow-y: auto;
          overflow-x: hidden;
          padding-right: 7px;
        }


        .main-fixed {
          height: 100%;
          overflow: hidden;
          padding-left: 7px;
        }


        .review-box {
          background: white;
          border-radius: 7px;
          padding: 7px 9px;
          margin-bottom: 6px;
        }


        .review-box h4 {
          font-size: 17px;
          margin-top: 0;
          margin-bottom: 5px;
        }


        .chip-title {
          font-size: 18px;
          font-weight: 700;
          margin-bottom: 4px;
        }


        .review-box p {
          margin: 2px 0;
          font-size: 14px;
        }


        .metric {
          font-size: 13px;
          line-height: 1.25;
          margin-bottom: 2px;
        }


        .btn {
          padding: 5px 8px;
          font-size: 14px;
        }


        .classification-button {
          margin-bottom: 5px;
        }


        .form-group {
          margin-bottom: 5px;
        }


        textarea.form-control {
          height: 45px !important;
          min-height: 45px !important;
          resize: none;
        }


        .plot-row {
          height: 100%;
          margin: 0;
        }


        .plot-column {
          height: 100%;
          padding-left: 5px;
          padding-right: 5px;
        }


        .plot-box {
          height: 100%;
          background: white;
          border-radius: 7px;
          padding: 7px;
          overflow: hidden;
        }


        .plot-title {
          height: 30px;
          font-size: 19px;
          font-weight: 500;
          display: flex;
          align-items: center;
          padding-left: 3px;
        }


        .plot-container {
          height: calc(100% - 30px);
          width: 100%;
          overflow: hidden;
        }


        .shiny-plot-output {
          width: 100% !important;
          height: 100% !important;
        }


        hr {
          margin-top: 5px;
          margin-bottom: 5px;
        }


        pre {
          font-size: 11px;
          padding: 3px;
          margin: 3px 0 0 0;
          border: none;
          background: transparent;
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
    # LEFT SIDEBAR
    # ========================================================
    
    column(
      
      width = 2,
      
      class = "sidebar-fixed",
      
      
      # ------------------------------------------------------
      # Chip information
      # ------------------------------------------------------
      
      div(
        
        class = "review-box",
        
        uiOutput(
          "chip_information"
        )
      ),
      
      
      # ------------------------------------------------------
      # Classification
      # ------------------------------------------------------
      
      div(
        
        class = "review-box",
        
        h4(
          "Classification"
        ),
        
        
        actionButton(
          "mark_waste",
          "Waste",
          width = "100%",
          class = "classification-button"
        ),
        
        
        actionButton(
          "mark_not_waste",
          "Not waste",
          width = "100%",
          class = "classification-button"
        ),
        
        
        actionButton(
          "mark_mixed",
          "Waste + false positives",
          width = "100%",
          class = "classification-button"
        )
      ),
      
      
      # ------------------------------------------------------
      # Notes
      # ------------------------------------------------------
      
      div(
        
        class = "review-box",
        
        textAreaInput(
          "notes",
          "Notes",
          value = "",
          rows = 2,
          width = "100%"
        ),
        
        
        actionButton(
          "save_note",
          "Save note",
          width = "100%"
        )
      ),
      
      
      # ------------------------------------------------------
      # Navigation
      # ------------------------------------------------------
      
      div(
        
        class = "review-box",
        
        
        fluidRow(
          
          column(
            
            6,
            
            actionButton(
              "previous_chip",
              "Previous",
              width = "100%"
            )
          ),
          
          
          column(
            
            6,
            
            actionButton(
              "next_chip",
              "Next",
              width = "100%"
            )
          )
        ),
        
        
        div(
          style = "height:5px;"
        ),
        
        
        actionButton(
          "next_unreviewed",
          "Next unreviewed",
          width = "100%"
        )
      ),
      
      
      # ------------------------------------------------------
      # Progress
      # ------------------------------------------------------
      
      div(
        
        class = "review-box",
        
        h4(
          "Progress"
        ),
        
        uiOutput(
          "review_summary"
        )
      ),
      
      
      # ------------------------------------------------------
      # Export
      # ------------------------------------------------------
      
      div(
        
        class = "review-box",
        
        actionButton(
          "export_results",
          "Export review",
          width = "100%"
        ),
        
        verbatimTextOutput(
          "export_message"
        )
      )
      
    ),
    
    
    # ========================================================
    # MAIN DISPLAY
    # ========================================================
    
    column(
      
      width = 10,
      
      class = "main-fixed",
      
      
      fluidRow(
        
        class = "plot-row",
        
        
        # ----------------------------------------------------
        # ORTHOMOSAIC
        # ----------------------------------------------------
        
        column(
          
          width = 6,
          
          class = "plot-column",
          
          
          div(
            
            class = "plot-box",
            
            
            div(
              class = "plot-title",
              "Orthomosaic + predicted waste"
            ),
            
            
            div(
              
              class = "plot-container",
              
              plotOutput(
                "image_plot",
                width = "100%",
                height = "100%"
              )
            )
          )
        ),
        
        
        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------
        
        column(
          
          width = 6,
          
          class = "plot-column",
          
          
          div(
            
            class = "plot-box",
            
            
            div(
              class = "plot-title",
              "Waste probability"
            ),
            
            
            div(
              
              class = "plot-container",
              
              plotOutput(
                "probability_plot",
                width = "100%",
                height = "100%"
              )
            )
          )
        )
        
      )
      
    )
    
  )
  
)


# ============================================================
# 13. SERVER
# ============================================================

server <- function(
    input,
    output,
    session
) {
  
  
  current_index <- reactiveVal(
    1
  )
  
  
  reviews <- reactiveVal(
    review_table_initial
  )
  
  
  export_status <- reactiveVal(
    ""
  )
  
  
  # ==========================================================
  # CURRENT CHIP
  # ==========================================================
  
  current_chip <- reactive({
    
    chips[
      current_index(),
      ,
      drop = FALSE
    ]
    
  })
  
  
  # ==========================================================
  # CURRENT PREDICTIONS
  # ==========================================================
  
  current_predictions <- reactive({
    
    chip <- current_chip()
    
    
    idx <- st_intersects(
      chip,
      predictions
    )[[1]]
    
    
    if (
      length(idx) == 0
    ) {
      
      return(
        predictions[
          0,
          ,
          drop = FALSE
        ]
      )
    }
    
    
    predictions[
      idx,
      ,
      drop = FALSE
    ]
    
  })
  
  
  # ==========================================================
  # UPDATE NOTE WHEN CHIP CHANGES
  # ==========================================================
  
  observeEvent(
    
    current_index(),
    
    {
      
      chip_id <- current_chip()$chip_id
      
      
      review_table <- reviews()
      
      
      row_id <- match(
        chip_id,
        review_table$chip_id
      )
      
      
      updateTextAreaInput(
        session,
        "notes",
        value = review_table$notes[
          row_id
        ]
      )
    }
  )
  
  
  # ==========================================================
  # 14. ORTHOMOSAIC PANEL
  # ==========================================================
  
  output$image_plot <- renderPlot({
    
    
    chip <- current_chip()
    
    
    chip_v <- vect(
      chip
    )
    
    
    chip_ext <- ext(
      chip_v
    )
    
    
    # --------------------------------------------------------
    # Crop to review chip
    # --------------------------------------------------------
    
    image_chip <- crop(
      orthomosaic,
      chip_ext,
      snap = "out"
    )
    
    
    # --------------------------------------------------------
    # Restore full 50 m x 50 m extent.
    #
    # This is especially relevant for chips near imagery edges.
    # --------------------------------------------------------
    
    image_chip <- extend(
      image_chip,
      chip_ext,
      snap = "out",
      fill = NA
    )
    
    
    # --------------------------------------------------------
    # Exact chip limits
    # --------------------------------------------------------
    
    x_limits <- c(
      xmin(chip_ext),
      xmax(chip_ext)
    )
    
    
    y_limits <- c(
      ymin(chip_ext),
      ymax(chip_ext)
    )
    
    
    par(
      mar = c(
        0,
        0,
        1.5,
        0
      ),
      pty = "s",
      xaxs = "i",
      yaxs = "i"
    )
    
    
    plotRGB(
      
      image_chip,
      
      r = RGB_BANDS[1],
      
      g = RGB_BANDS[2],
      
      b = RGB_BANDS[3],
      
      stretch = "lin",
      
      axes = FALSE,
      
      asp = 1,
      
      xlim = x_limits,
      
      ylim = y_limits,
      
      colNA = "black",
      
      main = chip$chip_id
    )
    
    
    # --------------------------------------------------------
    # Add waste polygons as red outlines only
    # --------------------------------------------------------
    
    pred_here <- current_predictions()
    
    
    if (
      nrow(pred_here) > 0
    ) {
      
      plot(
        st_geometry(
          pred_here
        ),
        add = TRUE,
        border = "red",
        col = NA,
        lwd = 2
      )
    }
    
  })
  
  
  # ==========================================================
  # 15. PROBABILITY PANEL
  # ==========================================================
  
  output$probability_plot <- renderPlot({
    
    
    chip <- current_chip()
    
    
    chip_v <- vect(
      chip
    )
    
    
    chip_ext <- ext(
      chip_v
    )
    
    
    probability_chip <- crop(
      probability,
      chip_ext,
      snap = "out"
    )
    
    
    probability_chip <- extend(
      probability_chip,
      chip_ext,
      snap = "out",
      fill = NA
    )
    
    
    x_limits <- c(
      xmin(chip_ext),
      xmax(chip_ext)
    )
    
    
    y_limits <- c(
      ymin(chip_ext),
      ymax(chip_ext)
    )
    
    
    par(
      mar = c(
        0,
        0,
        1.5,
        3
      ),
      pty = "s",
      xaxs = "i",
      yaxs = "i"
    )
    
    
    plot(
      
      probability_chip,
      
      range = c(
        PROBABILITY_MIN,
        PROBABILITY_MAX
      ),
      
      col = hcl.colors(
        100,
        "YlOrRd"
      ),
      
      axes = FALSE,
      
      asp = 1,
      
      xlim = x_limits,
      
      ylim = y_limits,
      
      main = chip$chip_id,
      
      plg = list(
        title = "P(waste)"
      )
      
    )
    
  })
  
  
  # ==========================================================
  # 16. CHIP INFORMATION
  # ==========================================================
  
  output$chip_information <- renderUI({
    
    
    chip <- current_chip()
    
    
    review_table <- reviews()
    
    
    row_id <- match(
      chip$chip_id,
      review_table$chip_id
    )
    
    
    current_status <- review_table$status[
      row_id
    ]
    
    
    pred_count <- nrow(
      current_predictions()
    )
    
    
    tagList(
      
      
      div(
        
        class = "chip-title",
        
        paste0(
          chip$chip_id,
          "  (",
          current_index(),
          " / ",
          nrow(chips),
          ")"
        )
      ),
      
      
      p(
        
        strong(
          "Status: "
        ),
        
        current_status
      ),
      
      
      p(
        
        strong(
          "Objects in chip: "
        ),
        
        pred_count
      )
      
    )
    
  })
  
  
  # ==========================================================
  # 17. CLASSIFICATION FUNCTION
  # ==========================================================
  
  classify_current_chip <- function(
    new_status
  ) {
    
    
    chip_id <- current_chip()$chip_id
    
    
    review_table <- reviews()
    
    
    row_id <- match(
      chip_id,
      review_table$chip_id
    )
    
    
    review_table$status[
      row_id
    ] <- new_status
    
    
    review_table$notes[
      row_id
    ] <- input$notes
    
    
    review_table$reviewed_at[
      row_id
    ] <- format(
      Sys.time(),
      "%Y-%m-%d %H:%M:%S"
    )
    
    
    reviews(
      review_table
    )
    
    
    save_reviews(
      review_table
    )
    
    
    next_index <- find_next_unreviewed(
      current_index(),
      review_table
    )
    
    
    current_index(
      next_index
    )
  }
  
  
  observeEvent(
    input$mark_waste,
    {
      
      classify_current_chip(
        "waste"
      )
      
    }
  )
  
  
  observeEvent(
    input$mark_not_waste,
    {
      
      classify_current_chip(
        "not_waste"
      )
      
    }
  )
  
  
  observeEvent(
    input$mark_mixed,
    {
      
      classify_current_chip(
        "mixed"
      )
      
    }
  )
  
  
  # ==========================================================
  # 18. SAVE NOTE
  # ==========================================================
  
  observeEvent(
    
    input$save_note,
    
    {
      
      chip_id <- current_chip()$chip_id
      
      
      review_table <- reviews()
      
      
      row_id <- match(
        chip_id,
        review_table$chip_id
      )
      
      
      review_table$notes[
        row_id
      ] <- input$notes
      
      
      reviews(
        review_table
      )
      
      
      save_reviews(
        review_table
      )
      
    }
  )
  
  
  # ==========================================================
  # 19. NAVIGATION
  # ==========================================================
  
  observeEvent(
    
    input$previous_chip,
    
    {
      
      idx <- max(
        1,
        current_index() - 1
      )
      
      
      current_index(
        idx
      )
      
    }
  )
  
  
  observeEvent(
    
    input$next_chip,
    
    {
      
      idx <- min(
        nrow(chips),
        current_index() + 1
      )
      
      
      current_index(
        idx
      )
      
    }
  )
  
  
  observeEvent(
    
    input$next_unreviewed,
    
    {
      
      idx <- find_next_unreviewed(
        current_index(),
        reviews()
      )
      
      
      current_index(
        idx
      )
      
    }
  )
  
  
  # ==========================================================
  # 20. REVIEW SUMMARY
  # ==========================================================
  
  output$review_summary <- renderUI({
    
    
    review_table <- reviews()
    
    
    total <- nrow(
      review_table
    )
    
    
    reviewed <- sum(
      review_table$status != "unreviewed"
    )
    
    
    remaining <- sum(
      review_table$status == "unreviewed"
    )
    
    
    waste_n <- sum(
      review_table$status == "waste"
    )
    
    
    not_waste_n <- sum(
      review_table$status == "not_waste"
    )
    
    
    mixed_n <- sum(
      review_table$status == "mixed"
    )
    
    
    candidate_remaining <- get_remaining_candidate_count(
      review_table
    )
    
    
    tagList(
      
      
      div(
        class = "metric",
        paste(
          "Reviewed:",
          reviewed,
          "/",
          total
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
          waste_n
        )
      ),
      
      
      div(
        class = "metric",
        paste(
          "Not waste:",
          not_waste_n
        )
      ),
      
      
      div(
        class = "metric",
        paste(
          "Mixed:",
          mixed_n
        )
      ),
      
      
      hr(),
      
      
      div(
        
        class = "metric",
        
        strong(
          paste(
            "Candidate objects remaining:",
            format(
              candidate_remaining,
              big.mark = ","
            )
          )
        )
      )
      
    )
    
  })
  
  
  # ==========================================================
  # 21. EXPORT REVIEW
  # ==========================================================
  
  observeEvent(
    
    input$export_results,
    
    {
      
      
      export_status(
        "Exporting..."
      )
      
      
      review_table <- reviews()
      
      
      # ------------------------------------------------------
      # Review grid
      # ------------------------------------------------------
      
      review_grid <- chips
      
      
      review_grid$status <- review_table$status[
        match(
          review_grid$chip_id,
          review_table$chip_id
        )
      ]
      
      
      review_grid$notes <- review_table$notes[
        match(
          review_grid$chip_id,
          review_table$chip_id
        )
      ]
      
      
      review_grid$reviewed_at <- review_table$reviewed_at[
        match(
          review_grid$chip_id,
          review_table$chip_id
        )
      ]
      
      
      st_write(
        
        review_grid,
        
        REVIEW_GRID_GPKG,
        
        layer = "review_grid",
        
        delete_dsn = TRUE,
        
        quiet = TRUE
      )
      
      
      # ------------------------------------------------------
      # Review classes
      # ------------------------------------------------------
      
      rejected_chips <- review_grid[
        review_grid$status == "not_waste",
        ,
        drop = FALSE
      ]
      
      
      mixed_chips <- review_grid[
        review_grid$status == "mixed",
        ,
        drop = FALSE
      ]
      
      
      waste_chips <- review_grid[
        review_grid$status == "waste",
        ,
        drop = FALSE
      ]
      
      
      # ------------------------------------------------------
      # Initially retain all predictions
      # ------------------------------------------------------
      
      retained <- predictions
      
      
      rejected_fragments <- predictions[
        0,
        ,
        drop = FALSE
      ]
      
      
      # ------------------------------------------------------
      # Remove geometry within not-waste chips
      # ------------------------------------------------------
      
      if (
        nrow(rejected_chips) > 0
      ) {
        
        
        rejected_union <- st_union(
          rejected_chips
        )
        
        
        rejected_fragments <- suppressWarnings(
          st_intersection(
            predictions,
            rejected_union
          )
        )
        
        
        retained <- suppressWarnings(
          st_difference(
            predictions,
            rejected_union
          )
        )
        
      }
      
      
      # ------------------------------------------------------
      # Clean retained geometry
      # ------------------------------------------------------
      
      retained <- st_make_valid(
        retained
      )
      
      
      retained <- suppressWarnings(
        st_collection_extract(
          retained,
          "POLYGON"
        )
      )
      
      
      if (
        nrow(retained) > 0
      ) {
        
        
        retained$area_m2_reviewed <- as.numeric(
          st_area(
            retained
          )
        )
        
        
        retained <- retained[
          retained$area_m2_reviewed >=
            MIN_FINAL_AREA_M2,
          ,
          drop = FALSE
        ]
        
      }
      
      
      # ------------------------------------------------------
      # Mixed areas
      # ------------------------------------------------------
      
      mixed_fragments <- predictions[
        0,
        ,
        drop = FALSE
      ]
      
      
      if (
        nrow(mixed_chips) > 0
      ) {
        
        
        mixed_union <- st_union(
          mixed_chips
        )
        
        
        mixed_fragments <- suppressWarnings(
          st_intersection(
            retained,
            mixed_union
          )
        )
        
      }
      
      
      # ------------------------------------------------------
      # Waste-classified areas
      # ------------------------------------------------------
      
      waste_fragments <- predictions[
        0,
        ,
        drop = FALSE
      ]
      
      
      if (
        nrow(waste_chips) > 0
      ) {
        
        
        waste_union <- st_union(
          waste_chips
        )
        
        
        waste_fragments <- suppressWarnings(
          st_intersection(
            retained,
            waste_union
          )
        )
        
      }
      
      
      # ------------------------------------------------------
      # Write outputs
      # ------------------------------------------------------
      
      write_layer(
        retained,
        RETAINED_GPKG,
        "retained_candidates"
      )
      
      
      write_layer(
        rejected_fragments,
        REJECTED_GPKG,
        "hard_negative_fragments"
      )
      
      
      write_layer(
        mixed_fragments,
        MIXED_GPKG,
        "mixed_predictions"
      )
      
      
      write_layer(
        waste_fragments,
        WASTE_GPKG,
        "waste_predictions"
      )
      
      
      # ------------------------------------------------------
      # Provisional pile count
      # ------------------------------------------------------
      
      provisional_piles <- 0
      
      provisional_area_m2 <- 0
      
      
      if (
        nrow(retained) > 0
      ) {
        
        
        retained_parts <- suppressWarnings(
          st_cast(
            retained,
            "POLYGON"
          )
        )
        
        
        retained_parts$area_m2 <- as.numeric(
          st_area(
            retained_parts
          )
        )
        
        
        retained_parts <- retained_parts[
          retained_parts$area_m2 >=
            MIN_FINAL_AREA_M2,
          ,
          drop = FALSE
        ]
        
        
        provisional_piles <- nrow(
          retained_parts
        )
        
        
        provisional_area_m2 <- sum(
          retained_parts$area_m2,
          na.rm = TRUE
        )
        
      }
      
      
      # ------------------------------------------------------
      # Save summary
      # ------------------------------------------------------
      
      summary_table <- data.frame(
        
        total_review_chips =
          nrow(review_table),
        
        reviewed_chips =
          sum(
            review_table$status !=
              "unreviewed"
          ),
        
        waste_chips =
          sum(
            review_table$status ==
              "waste"
          ),
        
        not_waste_chips =
          sum(
            review_table$status ==
              "not_waste"
          ),
        
        mixed_chips =
          sum(
            review_table$status ==
              "mixed"
          ),
        
        original_candidate_objects =
          nrow(predictions),
        
        provisional_remaining_piles =
          provisional_piles,
        
        provisional_remaining_area_m2 =
          provisional_area_m2
      )
      
      
      write.csv(
        summary_table,
        SUMMARY_CSV,
        row.names = FALSE
      )
      
      
      export_status(
        
        paste0(
          
          "Export complete\n",
          
          "Piles: ",
          format(
            provisional_piles,
            big.mark = ","
          ),
          
          "\nArea: ",
          round(
            provisional_area_m2,
            1
          ),
          " m²"
        )
      )
      
    }
    
  )
  
  
  output$export_message <- renderText({
    
    export_status()
    
  })
  
}


# ============================================================
# 22. RUN APP
# ============================================================

shinyApp(
  ui = ui,
  server = server
)