# 1. Override Windows Temp directories to prevent missing directory/Rtmp crashes
local_tmp <- file.path(Sys.getenv("USERPROFILE"), "AppData", "Local", "Temp", "R_Shiny_Temp")
dir.create(local_tmp, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(TMP = local_tmp, TEMP = local_tmp, TMPDIR = local_tmp)

# 2. Load required libraries
pacman::p_load(shiny, terra, png, DBI, RSQLite)

#----------------------------------------------------------
# CONFIG & PATHS
#----------------------------------------------------------

BASE_DIR <- "C:/Users/Peter/Documents/GitHub/mapping-waste-urban-africa/data/tiled_data"
if (!dir.exists(BASE_DIR)) BASE_DIR <- getwd()

IMAGES_ROOT      <- file.path(BASE_DIR, "images")
MASKS_ROOT       <- file.path(BASE_DIR, "masks")
REDIGITIZED_ROOT <- file.path(BASE_DIR, "redigitized")
CACHE_DIR        <- file.path(BASE_DIR, "preview_cache")
DB_PATH          <- file.path(BASE_DIR, "annotation_log.sqlite")

# Ensure required directory trees exist
dir.create(file.path(REDIGITIZED_ROOT, "images"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(REDIGITIZED_ROOT, "masks"), recursive = TRUE, showWarnings = FALSE)
dir.create(CACHE_DIR, recursive = TRUE, showWarnings = FALSE)

stretch_probs <- c(0.02, 0.98)

#----------------------------------------------------------
# FILE CONVERSION HELPERS
#----------------------------------------------------------

# Render satellite image tile (RGB stretched)
img_to_png <- function(img_path) {
  out_path <- file.path(CACHE_DIR, "current_img.png")
  
  r <- terra::rast(img_path)
  nb <- terra::nlyr(r)
  r_disp <- if (nb >= 3) r[[1:3]] else r[[1]]
  arr <- terra::as.array(r_disp)
  
  stretch_band <- function(b) {
    rng <- stats::quantile(b, probs = stretch_probs, na.rm = TRUE)
    if (diff(rng) == 0) {
      b[] <- 0.5
    } else {
      b <- (b - rng[1]) / (rng[2] - rng[1])
      b[b < 0] <- 0
      b[b > 1] <- 1
    }
    b
  }
  
  if (nb >= 3) {
    for (k in 1:3) arr[, , k] <- stretch_band(arr[, , k])
  } else {
    band <- stretch_band(arr[, , 1])
    arr <- array(band, dim = c(dim(band)[1:2], 3))
  }
  arr[is.na(arr)] <- 0
  
  png::writePNG(arr, out_path)
  out_path
}

# Render waste mask tile (1 = Bright Red, 0 = Black)
mask_to_red_png <- function(mask_path) {
  out_path <- file.path(CACHE_DIR, "current_mask.png")
  
  r <- terra::rast(mask_path)
  mask_mat <- terra::as.matrix(r[[1]], wide = TRUE)
  mask_mat[is.na(mask_mat)] <- 0
  
  rows <- nrow(mask_mat)
  cols <- ncol(mask_mat)
  
  rgb_arr <- array(0, dim = c(rows, cols, 3))
  waste_pixels <- (mask_mat == 1)
  
  # Channel 1 = Red
  rgb_arr[,,1][waste_pixels] <- 1.0
  
  png::writePNG(rgb_arr, out_path)
  out_path
}

list_tifs <- function(dir_path) {
  if (!dir.exists(dir_path)) return(character(0))
  list.files(dir_path, pattern = "\\.tiff?$", ignore.case = TRUE, full.names = FALSE)
}

#----------------------------------------------------------
# DATABASE HELPERS
#----------------------------------------------------------

db_connect <- function() {
  con <- DBI::dbConnect(RSQLite::SQLite(), DB_PATH)
  DBI::dbExecute(con, "PRAGMA journal_mode=WAL;")
  DBI::dbExecute(con, "PRAGMA busy_timeout=5000;")
  con
}

db_init <- function() {
  con <- db_connect()
  on.exit(DBI::dbDisconnect(con))
  DBI::dbExecute(con, "
    CREATE TABLE IF NOT EXISTS review_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      filename TEXT NOT NULL,
      status TEXT NOT NULL,
      reviewer TEXT NOT NULL,
      timestamp TEXT NOT NULL
    );
  ")
}

log_action <- function(filename, status, reviewer) {
  con <- db_connect()
  on.exit(DBI::dbDisconnect(con))
  DBI::dbExecute(
    con,
    "INSERT INTO review_log (filename, status, reviewer, timestamp) VALUES (?, ?, ?, ?)",
    params = list(filename, status, reviewer, as.character(Sys.time()))
  )
}

db_init()

#----------------------------------------------------------
# UI
#----------------------------------------------------------

ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      html, body { height: 100%; }
      .image-container {
        display: flex;
        gap: 10px;
        height: calc(100vh - 80px);
        background-color: #f7f7f7;
        padding: 10px;
        border: 1px solid #ddd;
      }
      .img-box {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
      }
      .img-box img {
        width: 100%;
        height: 90%;
        object-fit: contain;
      }
      .progress-box {
        background: #f7f7f7;
        border: 1px solid #ddd;
        padding: 10px;
        margin-bottom: 10px;
      }
    "))
  ),
  
  titlePanel("Quality Control: Waste Digitization Review"),
  
  sidebarLayout(
    sidebarPanel(
      uiOutput("welcome_text"),
      hr(),
      p(em("Waste mask panel shows digitized waste (1 = Bright Red).")),
      br(),
      actionButton("flag_bad", "Redigitize (Move pair)", class = "btn-danger", icon = icon("trash"), style="width:100%"),
      br(), br(),
      actionButton("flag_good", "Keep (Good Digitization)", class = "btn-success", icon = icon("check"), style="width:100%"),
      br(), br(),
      actionButton("skip_btn", "Skip File", icon = icon("forward"), style="width:100%"),
      hr(),
      verbatimTextOutput("current_file_text"),
      div(class = "progress-box", uiOutput("progress_stats")),
      width = 3
    ),
    
    mainPanel(
      div(
        class = "image-container",
        div(class = "img-box", h4("Satellite Image"), imageOutput("img_view", height = "100%")),
        div(class = "img-box", h4("Waste Mask (1 = Bright Red)"), imageOutput("mask_view", height = "100%"))
      ),
      width = 9
    )
  )
)

#----------------------------------------------------------
# SERVER
#----------------------------------------------------------

server <- function(input, output, session) {
  
  annotator_name <- reactiveVal(NULL)
  
  showModal(modalDialog(
    title = "Reviewer Login",
    textInput("name_input", "Enter your name:"),
    footer = actionButton("name_submit", "Start Quality Control", class = "btn-primary"),
    easyClose = FALSE
  ))
  
  observeEvent(input$name_submit, {
    nm <- trimws(input$name_input)
    if (nzchar(nm)) {
      annotator_name(nm)
      removeModal()
    }
  })
  
  output$welcome_text <- renderUI({
    req(annotator_name())
    strong(paste0("Reviewer: ", annotator_name()))
  })
  
  state <- reactiveValues(current_file = NULL)
  
  pick_next_file <- function() {
    imgs <- list_tifs(IMAGES_ROOT)
    msks <- list_tifs(MASKS_ROOT)
    valid_pairs <- intersect(imgs, msks)
    if (length(valid_pairs) == 0) return(NULL)
    sample(valid_pairs, 1)
  }
  
  refresh_file <- function() {
    state$current_file <- pick_next_file()
  }
  
  observe({
    req(annotator_name())
    if (is.null(state$current_file)) refresh_file()
  })
  
  output$img_view <- renderImage({
    req(state$current_file)
    p <- img_to_png(file.path(IMAGES_ROOT, state$current_file))
    list(src = p, contentType = "image/png", style = "width:100%; height:100%; object-fit:contain;")
  }, deleteFile = FALSE)
  
  output$mask_view <- renderImage({
    req(state$current_file)
    p <- mask_to_red_png(file.path(MASKS_ROOT, state$current_file))
    list(src = p, contentType = "image/png", style = "width:100%; height:100%; object-fit:contain;")
  }, deleteFile = FALSE)
  
  output$current_file_text <- renderText({
    if (is.null(state$current_file)) "No matching pairs remaining!" else paste("Current file:\n", state$current_file)
  })
  
  output$progress_stats <- renderUI({
    rem <- length(intersect(list_tifs(IMAGES_ROOT), list_tifs(MASKS_ROOT)))
    redig <- length(list_tifs(file.path(REDIGITIZED_ROOT, "images")))
    div(
      p(strong("Remaining in queue: "), rem),
      p(strong("Moved to redigitized: "), redig)
    )
  })
  
  process_file <- function(action) {
    fname <- state$current_file
    req(fname, annotator_name())
    
    img_src  <- file.path(IMAGES_ROOT, fname)
    mask_src <- file.path(MASKS_ROOT, fname)
    
    if (action == "REDIGITIZE") {
      file.rename(img_src, file.path(REDIGITIZED_ROOT, "images", fname))
      file.rename(mask_src, file.path(REDIGITIZED_ROOT, "masks", fname))
      log_action(fname, "FLAGGED_REDIGITIZE", annotator_name())
      showNotification(paste("Moved", fname, "to redigitized/"), type = "warning", duration = 2)
    } else if (action == "KEEP") {
      log_action(fname, "APPROVED_GOOD", annotator_name())
      showNotification(paste("Approved", fname), type = "message", duration = 2)
    }
    
    refresh_file()
  }
  
  observeEvent(input$flag_bad, { process_file("REDIGITIZE") })
  observeEvent(input$flag_good, { process_file("KEEP") })
  observeEvent(input$skip_btn, { refresh_file() })
}

shinyApp(ui, server)