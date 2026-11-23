# =============================================================================
# Shared figure theme, paths and result loaders
# =============================================================================
# Sourced by every figure script in this folder so that all manuscript figures
# share one typographic style and one output convention.
#
#   source("07_figures/00_figure_theme.R")
#
# Style follows the existing manuscript figures: IEEE-ish, serif, greyscale-safe,
# vector PDF plus 600-dpi PNG at 7.5 x 4.3 inches.
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
})

# --- paths -------------------------------------------------------------------
# Resolve everything from the repository root so the scripts run from anywhere.
repo_root <- function() {
  p <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  while (!file.exists(file.path(p, "CITATION.cff")) && dirname(p) != p) {
    p <- dirname(p)
  }
  if (!file.exists(file.path(p, "CITATION.cff"))) {
    stop("Could not find the repository root. Run from inside the repository.")
  }
  p
}

ROOT        <- repo_root()
RESULTS_DIR <- file.path(ROOT, "results")
FIG_DIR     <- file.path(RESULTS_DIR, "figures", "manuscript")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# --- theme -------------------------------------------------------------------
paper_theme <- theme_classic(base_size = 10, base_family = "serif") + theme(
  axis.text.x     = element_text(size = 8.5),
  axis.title      = element_text(size = 10),
  legend.position = "bottom",
  legend.title    = element_blank(),
  legend.text     = element_text(size = 9),
  strip.background = element_blank(),
  strip.text      = element_text(size = 9, face = "bold"),
  plot.margin     = margin(12, 18, 10, 12)
)

# Greyscale-safe palette: prints legibly in black and white.
grey_palette <- c("grey10", "grey40", "grey62", "grey80", "black", "grey25")

# Point shapes that stay distinguishable without colour.
mono_shapes <- c(21, 22, 23, 24, 25, 1)

#' Save a figure as vector PDF and 600-dpi PNG, matching the existing figures.
save_figure <- function(plot, name, width = 7.5, height = 4.3, dir = FIG_DIR) {
  stem <- file.path(dir, name)
  ggsave(paste0(stem, ".pdf"), plot, width = width, height = height)
  ggsave(paste0(stem, ".png"), plot, width = width, height = height,
         dpi = 600, bg = "white")
  message("  wrote ", basename(stem), ".{pdf,png}")
  invisible(stem)
}

# --- result loaders ----------------------------------------------------------
# Every experiment family writes the same metrics_all_seeds.csv schema, so one
# loader serves them all.

EXPERIMENT_FAMILIES <- c(
  "02_geobia_pixel"           = "GeoBIA (pixel level)",
  "03_unet_pytorch"          = "PyTorch baseline",
  "04_label_budget_factorial" = "Label budget",
  "05_negative_weight_sweep"  = "Negative weight",
  "06_pretrained_backbones"   = "Pretrained backbone"
)

# Configurations excluded from every reported table and figure.
# geobia_oracle_rf uses polygons vectorised from the reference mask, so the
# polygon boundaries ARE the answer and precision is 1.000 by construction. It
# is retained on disk as the evidence for how much segmentation quality matters,
# but it is not a method and is never reported as a result.
EXCLUDED_CONFIGS <- c("geobia_oracle_rf")

#' Read every metrics_all_seeds.csv under results/ into one tidy data frame.
#'
#' @param split  "test" (default) or "val"
#' @param rule   "argmax" (default) or "thresh"
#' @return data frame with one row per configuration, averaged over seeds
load_all_results <- function(split = "test", rule = "argmax",
                             include_excluded = FALSE) {
  files <- list.files(RESULTS_DIR, pattern = "^metrics_all_seeds\\.csv$",
                      recursive = TRUE, full.names = TRUE)
  if (!length(files)) stop("No metrics_all_seeds.csv found under ", RESULTS_DIR)

  rows <- lapply(files, function(f) {
    d <- tryCatch(utils::read.csv(f, check.names = FALSE),
                  error = function(e) NULL)
    if (is.null(d) || !nrow(d)) return(NULL)
    if (!all(c("split", "rule", "f1") %in% names(d))) return(NULL)
    d <- d[d$split == split & d$rule == rule, , drop = FALSE]
    if (!nrow(d)) return(NULL)

    rel    <- sub(paste0("^", RESULTS_DIR, "/?"), "", dirname(f))
    parts  <- strsplit(gsub("\\\\", "/", rel), "/")[[1]]
    family <- parts[1]
    config <- parts[length(parts)]

    gapcol <- "required_fpr_for_P=0.5_at_p=0.00045"
    impcol <- "implied_precision_at_p=0.00045"

    data.frame(
      family      = family,
      family_label = unname(ifelse(family %in% names(EXPERIMENT_FAMILIES),
                                   EXPERIMENT_FAMILIES[family], family)),
      config      = config,
      arch        = if ("arch" %in% names(d)) as.character(d$arch[1]) else NA_character_,
      seeds       = nrow(d),
      f1          = mean(d$f1),
      f1_sd       = stats::sd(d$f1),
      precision   = mean(d$precision),
      recall      = mean(d$recall),
      iou         = mean(d$iou),
      fpr         = mean(d$fpr),
      prauc       = if ("average_precision" %in% names(d)) mean(d$average_precision) else NA_real_,
      implied_P   = if (impcol %in% names(d)) mean(d[[impcol]]) else NA_real_,
      gap_x       = if (gapcol %in% names(d)) mean(d$fpr / d[[gapcol]]) else NA_real_,
      minutes     = if ("minutes" %in% names(d)) mean(d$minutes) else NA_real_,
      stringsAsFactors = FALSE
    )
  })

  out <- do.call(rbind, Filter(Negate(is.null), rows))
  if (!include_excluded) {
    drop <- out$config %in% EXCLUDED_CONFIGS
    if (any(drop))
      message("  excluded from reporting: ",
              paste(out$config[drop], collapse = ", "))
    out <- out[!drop, , drop = FALSE]
  }
  out <- out[order(-out$f1), ]
  rownames(out) <- NULL
  out
}

#' Read one of the pre-summarised comparison tables.
read_result_table <- function(relpath) {
  f <- file.path(RESULTS_DIR, relpath)
  if (!file.exists(f)) {
    stop("Missing result table: ", f,
         "\nGenerate it first with the corresponding script in 06_evaluation/ or 07_figures/.")
  }
  utils::read.csv(f, check.names = FALSE)
}

#' Tidy a backbone key such as "v6_dinov3vitl_frozen" into a readable label.
pretty_backbone <- function(x) {
  x <- sub("^v6_", "", x)
  x <- sub("^fx_", "", x)
  mode <- ifelse(grepl("_frozen$", x), "frozen",
                 ifelse(grepl("_finetuned$", x), "fine-tuned", NA))
  base <- sub("_(frozen|finetuned)$", "", x)
  base <- c(
    "efficientnetb4imagenet" = "EfficientNet-B4 (ImageNet)",
    "efficientnetb4oem"      = "EfficientNet-B4 (OpenEarthMap)",
    "dinov3vitl"             = "DINOv3 ViT-L (web)",
    "dinov3satvitl"          = "DINOv3 ViT-L (SAT-493M)",
    "satlasaerial"           = "SatlasPretrain (aerial)",
    "dinov3vits"             = "DINOv3 ViT-S",
    "dinov3convnext"         = "DINOv3 ConvNeXt",
    "dinov3sat"              = "DINOv3-SAT",
    "efficientnetb0"         = "EfficientNet-B0 (ImageNet)",
    "openearthmap"           = "OpenEarthMap",
    "resnet34"               = "ResNet-34 (ImageNet)",
    "scratch"                = "U-Net from scratch",
    "geobia-otb-rf"          = "GeoBIA, OTB segments (RF)",
    "geobia-oracle-rf"       = "GeoBIA, oracle segments (RF)"
  )[base] |> unname() |> (\(v) ifelse(is.na(v), base, v))()
  list(backbone = base, mode = mode)
}


#' Label points, using ggrepel when it is installed and geom_text otherwise.
#' @param data data frame containing the aesthetics already mapped by the plot
#' @param label_col column holding the text
label_points <- function(data, label_col = "backbone", size = 2.7) {
  if (requireNamespace("ggrepel", quietly = TRUE)) {
    ggrepel::geom_text_repel(
      data = data, ggplot2::aes(label = .data[[label_col]]),
      family = "serif", size = size, colour = "grey20",
      min.segment.length = 0, segment.colour = "grey70",
      box.padding = 0.45, max.overlaps = Inf, seed = 42)
  } else {
    ggplot2::geom_text(
      data = data, ggplot2::aes(label = .data[[label_col]]),
      family = "serif", size = size, colour = "grey20",
      hjust = -0.08, vjust = -0.6, check_overlap = TRUE)
  }
}

# Landscape prevalence used throughout the study.
P_LANDSCAPE <- 0.00045

message("figure theme loaded | root: ", ROOT)
message("figures will be written to: ", FIG_DIR)
