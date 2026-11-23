# =============================================================================
# Build every manuscript table from the saved result files
# =============================================================================
# One script, one source of truth. Each table is written as CSV (for the record)
# and as Markdown (to paste into the manuscript), so no number is ever typed by
# hand.
#
#   Rscript 06_evaluation/06_build_manuscript_tables.R
#
# Tables
#   T1  headline comparison: GeoBIA vs U-Net vs pretrained backbones
#   T2  pretrained and foundation backbones, frozen vs fine-tuned
#   T3  label budget x pretraining domain
#   T4  complementary not-waste loss weight sweep
#   T5  deployability: every reported configuration, ranked by gap
#
# Excluded from all of them: the oracle-segment GeoBIA condition, whose
# polygons were vectorised from the reference mask and which therefore cannot
# produce a false positive. See EXCLUDED_CONFIGS in 07_figures/00_figure_theme.R
#
# Writes results/tables/*.csv and results/tables/*.md
# =============================================================================

local({
  p <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  while (!file.exists(file.path(p, "CITATION.cff")) && dirname(p) != p) p <- dirname(p)
  if (!file.exists(file.path(p, "CITATION.cff")))
    stop("Run this from inside the repository.")
  setwd(p)
})
source("07_figures/00_figure_theme.R")

TAB_DIR <- file.path(RESULTS_DIR, "tables")
dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)

# --- helpers ------------------------------------------------------------------
write_table <- function(df, name, caption, digits = 4) {
  num <- vapply(df, is.numeric, logical(1))
  out <- df
  out[num] <- lapply(out[num], function(x) round(x, digits))
  utils::write.csv(out, file.path(TAB_DIR, paste0(name, ".csv")), row.names = FALSE)

  hdr <- paste0("| ", paste(names(out), collapse = " | "), " |")
  sep <- paste0("|", paste(rep("---", ncol(out)), collapse = "|"), "|")
  body <- apply(out, 1, function(r)
    paste0("| ", paste(format(r, trim = TRUE), collapse = " | "), " |"))
  writeLines(c(paste0("**", caption, "**"), "", hdr, sep, body),
             file.path(TAB_DIR, paste0(name, ".md")))
  message(sprintf("  %-28s %d rows", paste0(name, ".{csv,md}"), nrow(out)))
  invisible(out)
}

fmt_pm <- function(m, s) sprintf("%.4f (%.4f)", m, s)

d <- load_all_results("test", "argmax")
cat(sprintf("\nreported configurations: %d\n", nrow(d)))

# =============================================================================
# T1 headline comparison
# =============================================================================
pick <- function(cfg) d[d$config == cfg, , drop = FALSE]
best_of <- function(fam) {
  s <- d[d$family == fam, ]
  if (!nrow(s)) return(NULL)
  s[which.max(s$f1), ]
}

rows <- list()
add <- function(r, label, note) {
  if (is.null(r) || !nrow(r)) return(invisible(NULL))
  rows[[length(rows) + 1]] <<- data.frame(
    Approach = label,
    `Segmentation required` = note,
    F1 = fmt_pm(r$f1, r$f1_sd),
    Precision = round(r$precision, 4), Recall = round(r$recall, 4),
    IoU = round(r$iou, 4), FPR = round(r$fpr, 5),
    `Deployability gap` = sprintf("%.1f x", r$gap_x),
    check.names = FALSE, stringsAsFactors = FALSE)
}

add(pick("geobia_otb_rf"), "GeoBIA, random forest", "yes, automated (OTB meanshift)")
add(pick("unet_f16_clean"), "U-Net, trained from scratch", "no")
add(best_of("06_pretrained_backbones"), "U-Net with pretrained backbone (best)", "no")

t1 <- do.call(rbind, rows)
write_table(t1, "T1_headline_comparison",
            paste("Table 1. Waste-class accuracy on the spatially clean held-out",
                  "partition (216 tiles, label 99 excluded, 3 seeds, argmax rule).",
                  "F1 reported as mean (SD). The deployability gap is the achieved",
                  "false-positive rate divided by the rate required for precision",
                  "0.5 at landscape prevalence (0.045%)."))

# =============================================================================
# T2 pretrained and foundation backbones
# =============================================================================
v6 <- d[d$family == "06_pretrained_backbones", ]
if (nrow(v6)) {
  meta <- lapply(v6$config, pretty_backbone)
  t2 <- data.frame(
    Backbone = vapply(meta, `[[`, character(1), "backbone"),
    Encoder  = vapply(meta, `[[`, character(1), "mode"),
    F1 = fmt_pm(v6$f1, v6$f1_sd),
    Precision = round(v6$precision, 4), Recall = round(v6$recall, 4),
    IoU = round(v6$iou, 4), `PR-AUC` = round(v6$prauc, 4),
    FPR = round(v6$fpr, 5),
    `Implied precision` = round(v6$implied_P, 4),
    `Gap` = sprintf("%.1f x", v6$gap_x),
    `Minutes` = round(v6$minutes, 1),
    check.names = FALSE, stringsAsFactors = FALSE)
  t2 <- t2[order(-v6$f1), ]
  write_table(t2, "T2_pretrained_backbones",
              paste("Table 2. Pretrained and foundation-model backbones, frozen",
                    "versus fine-tuned encoders. All ten configurations loaded",
                    "100% of their checkpoint tensors (verified in",
                    "preflight_report.json). Implied precision is the Bayes",
                    "projection of each model's recall and FPR to landscape",
                    "prevalence."))
}

# =============================================================================
# T3 label budget x pretraining domain
# =============================================================================
lb <- d[d$family == "04_label_budget_factorial", ]
if (nrow(lb)) {
  lb$budget <- as.integer(sub(".*_n", "", lb$config))
  lb$key    <- sub("^fx_", "", sub("_n[0-9]+$", "", lb$config))
  lb$Backbone <- vapply(lb$key, function(k) pretty_backbone(k)$backbone, character(1))
  budgets <- sort(unique(lb$budget))
  wide <- data.frame(Backbone = sort(unique(lb$Backbone)),
                     check.names = FALSE, stringsAsFactors = FALSE)
  for (b in budgets) {
    col <- vapply(wide$Backbone, function(bb) {
      r <- lb[lb$Backbone == bb & lb$budget == b, ]
      if (nrow(r)) sprintf("%.4f", r$f1[1]) else "--"
    }, character(1))
    wide[[paste0("n = ", b)]] <- col
  }
  ord <- vapply(wide$Backbone, function(bb) {
    r <- lb[lb$Backbone == bb & lb$budget == min(budgets), ]
    if (nrow(r)) r$f1[1] else NA_real_
  }, numeric(1))
  wide <- wide[order(-ord), ]
  write_table(wide, "T3_label_budget",
              paste("Table 3. Waste-class F1 on held-out data as a function of",
                    "the number of annotated training tiles, by pretraining",
                    "domain. Mean of 3 seeds."))
}

# =============================================================================
# T4 negative-weight sweep
# =============================================================================
nw <- load_all_results("test", "argmax", include_excluded = TRUE)
nw <- nw[nw$family == "05_negative_weight_sweep", ]
if (nrow(nw)) {
  nw$w <- as.numeric(sub(".*_w", "", nw$config))
  nw <- nw[order(nw$w), ]
  t4 <- data.frame(
    `Not-waste loss weight` = nw$w,
    F1 = round(nw$f1, 4), Precision = round(nw$precision, 4),
    Recall = round(nw$recall, 4), FPR = round(nw$fpr, 6),
    `Implied precision` = round(nw$implied_P, 4),
    Gap = ifelse(is.finite(nw$gap_x), sprintf("%.1f x", nw$gap_x), "undefined"),
    check.names = FALSE)
  write_table(t4, "T4_negative_weight_sweep",
              paste("Table 4. Effect of the complementary not-waste loss weight.",
                    "Raising it lowers the false-positive rate but collapses",
                    "recall; at weights of 1.0 and above the gap is undefined",
                    "because recall falls below 1%."))
}

# =============================================================================
# T5 deployability across every reported configuration
# =============================================================================
u <- d[is.finite(d$gap_x), ]
t5 <- data.frame(
  Configuration = u$config,
  Family = u$family_label,
  F1 = round(u$f1, 4), Recall = round(u$recall, 4),
  FPR = round(u$fpr, 5),
  `Implied precision at 0.045%` = round(u$implied_P, 4),
  `Gap` = round(u$gap_x, 1),
  check.names = FALSE)
t5 <- t5[order(t5$Gap), ]
write_table(t5, "T5_deployability_all",
            paste("Table 5. Every reported configuration ranked by deployability",
                  "gap. No configuration approaches 1.0, which is the value a",
                  "directly usable map would require."))

# =============================================================================
# summary for the text
# =============================================================================
cat("\n--- numbers for the manuscript text ---\n")
usable <- d[is.finite(d$gap_x) & d$recall >= 0.35, ]
cat(sprintf("reported configurations            : %d\n", nrow(d)))
cat(sprintf("usable (recall >= 0.35)            : %d\n", nrow(usable)))
cat(sprintf("F1 range over usable configurations: %.4f to %.4f\n",
            min(usable$f1), max(usable$f1)))
cat(sprintf("deployability gap over the same    : %.1fx to %.1fx\n",
            min(usable$gap_x), max(usable$gap_x)))
cat(sprintf("best gap                           : %.1fx (%s)\n",
            min(usable$gap_x), usable$config[which.min(usable$gap_x)]))
if (nrow(v6))
  cat(sprintf("best gap among foundation models   : %.1fx (%s)\n",
              min(v6$gap_x), v6$config[which.min(v6$gap_x)]))

g <- d[d$config == "geobia_otb_rf", ]
b <- d[which.max(d$f1), ]
if (nrow(g))
  cat(sprintf("\nGeoBIA (OTB) %.4f vs best model %.4f: difference %+.4f\n",
              g$f1, b$f1, b$f1 - g$f1))

if (nrow(lb)) {
  s64 <- lb[lb$budget == min(lb$budget), ]
  bb  <- s64[which.max(s64$f1), ]
  sc  <- lb[lb$key == "scratch", ]
  if (nrow(sc)) {
    cat(sprintf("\n%s at n=%d (F1 %.4f) matches scratch at n=%d (F1 %.4f): %.0fx fewer labels\n",
                bb$Backbone, bb$budget, bb$f1, max(sc$budget),
                sc$f1[sc$budget == max(sc$budget)], max(sc$budget) / bb$budget))
  }
}

prev <- utils::read.csv(file.path(RESULTS_DIR, "03_unet_pytorch", "unet_f16_clean",
                                  "metrics_all_seeds.csv"), check.names = FALSE)
cat(sprintf("\ntest prevalence %.3f%%, validation %.3f%%, landscape %.3f%% (%.0fx)\n",
            100 * unique(prev$prevalence[prev$split == "test"])[1],
            100 * unique(prev$prevalence[prev$split == "val"])[1],
            100 * P_LANDSCAPE,
            unique(prev$prevalence[prev$split == "test"])[1] / P_LANDSCAPE))

cat(sprintf("\ntables written to %s\n", TAB_DIR))
