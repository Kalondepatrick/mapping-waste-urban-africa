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

# --- IEEE figure specification -----------------------------------------------
# IEEE Transactions: single-column figures are 3.5 in (88.9 mm) wide, double
# column 7.16 in (181.9 mm). Body text is 10 pt Times, so figure lettering is
# set at 8 pt and never smaller than 6 pt after any scaling. Line art is
# supplied at 600 dpi, and every figure must remain legible in greyscale.
IEEE_1COL <- 3.5
IEEE_2COL <- 7.16
IEEE_BASE <- 8      # pt, lettering
IEEE_FAMILY <- "serif"   # maps to Times in the pdf and png devices

ieee_theme <- theme_classic(base_size = IEEE_BASE, base_family = IEEE_FAMILY) +
  theme(
    axis.text        = element_text(size = IEEE_BASE - 1, colour = "black"),
    axis.title       = element_text(size = IEEE_BASE, colour = "black"),
    axis.line        = element_line(linewidth = 0.3, colour = "black"),
    axis.ticks       = element_line(linewidth = 0.3, colour = "black"),
    axis.ticks.length = unit(1.6, "pt"),
    legend.position  = "bottom",
    legend.title     = element_blank(),
    legend.text      = element_text(size = IEEE_BASE - 1),
    legend.key.size  = unit(8, "pt"),
    legend.margin    = margin(0, 0, 0, 0),
    legend.box.margin = margin(-4, 0, 0, 0),
    legend.background = element_blank(),
    strip.background = element_blank(),
    strip.text       = element_text(size = IEEE_BASE - 1, face = "bold"),
    plot.title       = element_blank(),   # IEEE titles live in the caption
    plot.subtitle    = element_blank(),
    plot.margin      = margin(2, 4, 2, 2)
  )

# Backward-compatible alias for scripts written against the earlier theme.
paper_theme <- ieee_theme

# Greyscale-safe palette: prints legibly in black and white.
grey_palette <- c("grey10", "grey40", "grey62", "grey80", "black", "grey25")

# Point shapes that stay distinguishable without colour.
mono_shapes <- c(21, 22, 23, 24, 25, 1)

#' Save a figure at IEEE specification.
#'
#' Writes a vector PDF (for the submission), a 600-dpi PNG (for the Word
#' manuscript) and, when `tiff = TRUE`, a 600-dpi LZW TIFF, which is what IEEE
#' asks for in the final package.
#'
#' @param width inches; defaults to the IEEE single-column measure
save_figure <- function(plot, name, width = IEEE_1COL, height = 2.6,
                        dir = FIG_DIR, tiff = FALSE) {
  stem <- file.path(dir, name)
  ggsave(paste0(stem, ".pdf"), plot, width = width, height = height,
         device = grDevices::cairo_pdf)
  ggsave(paste0(stem, ".png"), plot, width = width, height = height,
         dpi = 600, bg = "white")
  if (tiff) {
    ggsave(paste0(stem, ".tiff"), plot, width = width, height = height,
           dpi = 600, bg = "white", compression = "lzw")
  }
  message("  wrote ", basename(stem), sprintf("  [%.2f x %.2f in]", width, height))
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

#' Test-split results with the matching validation F1 attached.
#'
#' Model selection must be made on the validation split and reported on the
#' held-out split. Carrying both in one frame makes the selection auditable:
#' `val_f1` is what the choice was made on, everything else is what is reported.
load_with_validation <- function(rule = "argmax", include_excluded = FALSE) {
  te <- load_all_results("test", rule, include_excluded)
  va <- load_all_results("val",  rule, include_excluded)
  te$val_f1 <- va$f1[match(te$config, va$config)]
  te$val_rank <- rank(-te$val_f1, na.last = "keep", ties.method = "min")
  te
}

#' The configuration with the highest validation F1 within a subset.
#'
#' @param d    frame from load_with_validation()
#' @param rows logical index restricting the candidate set
select_on_validation <- function(d, rows = TRUE) {
  s <- d[rows & !is.na(d$val_f1), , drop = FALSE]
  if (!nrow(s)) return(NULL)
  s[which.max(s$val_f1), , drop = FALSE]
}

#' Pareto frontier over (recall maximised, deployability gap minimised).
#'
#' A configuration is on the frontier when no other configuration achieves both
#' a higher recall and a lower gap. This replaces an arbitrary recall cut-off
#' when asking which model is operationally preferable.
pareto_frontier <- function(d) {
  s <- d[is.finite(d$gap_x) & is.finite(d$recall), , drop = FALSE]
  keep <- vapply(seq_len(nrow(s)), function(i) {
    !any(s$recall >= s$recall[i] & s$gap_x <= s$gap_x[i] &
           (s$recall > s$recall[i] | s$gap_x < s$gap_x[i]))
  }, logical(1))
  s[keep, , drop = FALSE][order(-s$recall[keep]), , drop = FALSE]
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


#' Turn a configuration directory name into a reader-facing label.
#'
#' Configuration names encode the experiment family in a prefix and the varied
#' factor in a suffix: `fx_<backbone>_n<budget>` for the annotation-budget
#' factorial, `nw_<backbone>_w<weight>` for the loss sweep, `v6_<backbone>_
#' <mode>` for the backbone comparison and a trailing `_hn` for hard-negative
#' fine-tuning. This renders all of them in one readable form.
config_label <- function(x) {
  vapply(x, function(cfg) {
    if (identical(cfg, "geobia_otb_rf"))    return("GeoBIA, OTB segments (RF)")
    if (identical(cfg, "geobia_oracle_rf")) return("GeoBIA, oracle segments (RF)")
    s <- sub("^(fx|v6|nw|torch)_", "", cfg)
    extra <- character(0)
    if (grepl("_n[0-9]+$", s)) {
      extra <- c(extra, paste0(sub(".*_n", "", s), " tiles"))
      s <- sub("_n[0-9]+$", "", s)
    }
    if (grepl("_w[0-9.]+$", s)) {
      extra <- c(extra, paste0("not-waste weight ", sub(".*_w", "", s)))
      s <- sub("_w[0-9.]+$", "", s)
    }
    if (grepl("_hn$", s)) {
      extra <- c(extra, "hard negatives")
      s <- sub("_hn$", "", s)
    }
    pb <- pretty_backbone(s)
    if (!is.na(pb$mode)) extra <- c(pb$mode, extra)
    base <- pb$backbone
    if (identical(base, "dinov2")) base <- "DINOv2 (web)"
    if (identical(base, "dinov3satvitl")) base <- "DINOv3 ViT-L (SAT-493M)"
    if (identical(base, "dinov3vitl")) base <- "DINOv3 ViT-L (web)"
    if (identical(base, "satlasaerial")) base <- "SatlasPretrain (aerial)"
    if (identical(base, "efficientnetb4imagenet")) base <- "EfficientNet-B4 (ImageNet)"
    if (identical(base, "unet_f16_clean")) base <- "U-Net from scratch (TensorFlow)"
    if (identical(base, "scratch")) base <- "U-Net from scratch"
    if (!length(extra)) base else paste0(base, ", ", paste(extra, collapse = ", "))
  }, character(1), USE.NAMES = FALSE)
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
