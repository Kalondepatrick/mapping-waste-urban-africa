# =============================================================================
# Build every manuscript table from the saved result files
# =============================================================================
# One script, one source of truth. Each table is written as CSV (for the
# record) and as Markdown (to read), together with results/tables/manifest.csv,
# which carries the IEEE table number, title and note so that the Word build
# never has to parse a caption out of prose.
#
#   Rscript 06_evaluation/06_build_manuscript_tables.R
#
# Tables are numbered in order of first citation in docs/RESULTS.md, as IEEE
# requires. The sections below follow the order of the analysis, so the roman
# numerals are not in file order.
#   I    comparison of mapping approaches                     cited in 2
#   II   GeoBIA model development, segment level              cited in 2
#   III  annotation budget x pretraining corpus               cited in 3.1
#   IV   pretrained and foundation backbones                  cited in 3.2
#   V    attainable operating set (recall-gap frontier)       cited in 4.2
#   VI   complementary not-waste loss weight                  cited in 4.3
#   VII  spatial distribution of false positives              07_error_location.R
#   VIII spatial resolution robustness                        09_robustness_tables.R
#   IX   brightness robustness                                09_robustness_tables.R
#   X    every configuration ranked by gap                    appendix, uncited
#
# MODEL SELECTION. Every "best" reported here is chosen on the VALIDATION
# split and only then evaluated on the held-out split, using the one-standard-
# error rule: take the highest validation F1, and among configurations whose
# validation F1 lies within one standard error of it, prefer the one with the
# lowest deployability gap. Choosing on the held-out split would make the
# reported accuracy optimistically biased.
#
# Excluded throughout: the oracle-segment GeoBIA condition, whose polygons were
# vectorised from the reference mask and which therefore cannot produce a false
# positive. See EXCLUDED_CONFIGS in 07_figures/00_figure_theme.R
# =============================================================================

local({
  p <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  while (!file.exists(file.path(p, "CITATION.cff")) && dirname(p) != p) p <- dirname(p)
  if (!file.exists(file.path(p, "CITATION.cff")))
    stop("Run this from inside the repository.")
  setwd(p)
})
suppressMessages(source("07_figures/00_figure_theme.R"))

TAB_DIR <- file.path(RESULTS_DIR, "tables")
dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)

MANIFEST <- data.frame(file = character(), roman = character(),
                       title = character(), note = character(),
                       placement = character(), stringsAsFactors = FALSE)

# --- helpers ------------------------------------------------------------------
write_table <- function(df, name, roman, title, digits = 4, note = "",
                        placement = "body") {
  out <- df
  out[] <- lapply(out, function(x) trimws(as.character(x)))
  # right-align anything that reads as a number, left-align the rest
  num <- vapply(out, function(x)
    all(grepl("^[-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?$", x) | x %in% c("--", "n.d.")),
    logical(1))

  utils::write.csv(out, file.path(TAB_DIR, paste0(name, ".csv")), row.names = FALSE)

  align <- ifelse(num, "---:", ":---")
  md <- c(
    paste0("**TABLE ", roman, "**"),
    paste0("**", title, "**"), "",
    paste0("| ", paste(names(out), collapse = " | "), " |"),
    paste0("| ", paste(align, collapse = " | "), " |"),
    apply(out, 1, function(r) paste0("| ", paste(r, collapse = " | "), " |"))
  )
  if (nzchar(note)) md <- c(md, "", paste0("*", note, "*"))
  writeLines(md, file.path(TAB_DIR, paste0(name, ".md")))

  MANIFEST[nrow(MANIFEST) + 1, ] <<- list(paste0(name, ".csv"), roman, title,
                                          note, placement)
  message(sprintf("  TABLE %-4s %-34s %2d rows x %d cols", roman,
                  paste0(name, ".csv"), nrow(out), ncol(out)))
  invisible(out)
}

fmt_pm  <- function(m, s) sprintf("%.3f (%.3f)", m, s)
fmt_gap <- function(x) ifelse(is.finite(x), sprintf("%.1f", x), "n.d.")
# Fixed decimal places so that columns align on the point, as IEEE requires.
f1n <- function(x) sprintf("%.1f", x)
f3  <- function(x) sprintf("%.3f", x)
f4  <- function(x) sprintf("%.4f", x)
f5  <- function(x) sprintf("%.5f", x)

# --- load, with validation attached so every choice is auditable --------------
d  <- load_with_validation("argmax")
va <- load_all_results("val", "argmax")
d$val_sd <- va$f1_sd[match(d$config, va$config)]

cat(sprintf("\nreported configurations: %d\n", nrow(d)))

# --- the selection rule, applied once and used everywhere ---------------------
vbest  <- d[which.max(d$val_f1), ]
val_se <- vbest$val_sd / sqrt(vbest$seeds)
tied   <- d[!is.na(d$val_f1) & d$val_f1 >= vbest$val_f1 - val_se, ]
chosen <- tied[which.min(tied$gap_x), ]

cat(sprintf("selection: validation F1 max %.4f (%s), SE %.4f\n",
            vbest$val_f1, vbest$config, val_se))
cat(sprintf("           %d configuration(s) within one SE; selected %s\n",
            nrow(tied), chosen$config))

SELECTION_NOTE <- sprintf(paste(
  "Models are selected on the validation split by the one-standard-error rule",
  "(highest validation F1, then lowest deployability gap among configurations",
  "within one standard error of it) and only then evaluated on the held-out",
  "partition. The rule selects %s. Val. F1 is shown so that the selection can",
  "be checked; it is not a performance claim."),
  pretty_backbone(chosen$config)$backbone)

# =============================================================================
# TABLE I  comparison of mapping approaches
# =============================================================================
pick <- function(cfg) d[d$config == cfg, , drop = FALSE]

rows <- list()
add <- function(r, label, seg) {
  if (is.null(r) || !nrow(r)) return(invisible(NULL))
  rows[[length(rows) + 1]] <<- data.frame(
    Approach = label,
    `Segmentation` = seg,
    `Val. F1` = f3(r$val_f1),
    `F1` = fmt_pm(r$f1, r$f1_sd),
    Precision = f3(r$precision), Recall = f3(r$recall),
    IoU = f3(r$iou), FPR = f4(r$fpr),
    `Gap` = fmt_gap(r$gap_x),
    check.names = FALSE, stringsAsFactors = FALSE)
}

add(pick("geobia_otb_rf"),  "GeoBIA, random forest",         "OTB meanshift")
add(pick("unet_f16_clean"), "U-Net, from scratch",           "none")
add(chosen,                 paste0(pretty_backbone(chosen$config)$backbone,
                                   ", ", pretty_backbone(chosen$config)$mode),
    "none")

t1 <- do.call(rbind, rows)
write_table(t1, "T1_approach_comparison", "I",
  "Comparison of Mapping Approaches on the Held-Out Partition",
  note = paste(
    "Waste class, 216 spatially clean held-out tiles, label 99 excluded, mean",
    "of three seeds with the standard deviation in parentheses, argmax decision",
    "rule. FPR is the false-positive rate. Gap is the achieved FPR divided by",
    "the FPR that precision 0.5 would require at the landscape prevalence of",
    "0.045%; a directly usable map would have a gap of 1.0.",
    SELECTION_NOTE))

# =============================================================================
# TABLE II  pretrained and foundation backbones
# =============================================================================
v6 <- d[d$family == "06_pretrained_backbones", ]
v6 <- v6[order(-v6$val_f1), ]
if (nrow(v6)) {
  meta <- lapply(v6$config, pretty_backbone)
  t2 <- data.frame(
    Backbone = vapply(meta, `[[`, character(1), "backbone"),
    Encoder  = vapply(meta, `[[`, character(1), "mode"),
    `Val. F1` = f3(v6$val_f1),
    F1 = fmt_pm(v6$f1, v6$f1_sd),
    Precision = f3(v6$precision), Recall = f3(v6$recall),
    IoU = f3(v6$iou), `PR-AUC` = f3(v6$prauc),
    FPR = f4(v6$fpr),
    `Gap` = fmt_gap(v6$gap_x),
    `Min.` = f1n(v6$minutes),
    check.names = FALSE, stringsAsFactors = FALSE)
  write_table(t2, "T4_pretrained_backbones", "IV",
    "Pretrained and Foundation-Model Backbones, Frozen Versus Fine-Tuned",
    note = paste(
      "Ordered by validation F1, which is the quantity selection is made on.",
      "All ten configurations transferred 100% of their checkpoint tensors,",
      "verified against a freshly initialised model of the same specification",
      "(preflight_report.json). PR-AUC is threshold independent. Min. is",
      "training minutes per seed on one GPU. Other columns as in Table I."))
}

# =============================================================================
# TABLE III  annotation budget x pretraining corpus
# =============================================================================
lb <- d[d$family == "04_label_budget_factorial", ]
if (nrow(lb)) {
  lb$budget   <- as.integer(sub(".*_n", "", lb$config))
  lb$key      <- sub("^fx_", "", sub("_n[0-9]+$", "", lb$config))
  lb$Backbone <- vapply(lb$key, function(k) pretty_backbone(k)$backbone, character(1))
  budgets <- sort(unique(lb$budget))
  wide <- data.frame(Backbone = sort(unique(lb$Backbone)),
                     check.names = FALSE, stringsAsFactors = FALSE)
  for (b in budgets) {
    wide[[paste0("n = ", b)]] <- vapply(wide$Backbone, function(bb) {
      r <- lb[lb$Backbone == bb & lb$budget == b, ]
      if (nrow(r)) sprintf("%.3f", r$f1[1]) else "--"
    }, character(1))
  }
  sc64 <- lb$f1[lb$key == "scratch" & lb$budget == min(budgets)]
  wide[["Gain at n = 64"]] <- vapply(wide$Backbone, function(bb) {
    r <- lb[lb$Backbone == bb & lb$budget == min(budgets), ]
    if (!nrow(r) || !length(sc64)) return("--")
    if (identical(bb, "U-Net from scratch")) return("--")
    sprintf("%+.3f", r$f1[1] - sc64[1])
  }, character(1))

  ord <- vapply(wide$Backbone, function(bb) {
    r <- lb[lb$Backbone == bb & lb$budget == min(budgets), ]
    if (nrow(r)) r$f1[1] else NA_real_
  }, numeric(1))
  wide <- wide[order(-ord), ]
  write_table(wide, "T3_label_budget", "III",
    "Waste-Class F1 by Annotation Budget and Pretraining Corpus",
    note = paste(
      "Held-out F1, mean of three seeds, argmax rule. n is the number of",
      "annotated training tiles. The same tile subset is drawn for every seed",
      "at a given budget, so the spread across seeds reflects training",
      "stochasticity only and not the variance of the subset itself; the",
      "budget effect is therefore reported as an order of magnitude and not",
      "as a point estimate. Rows ordered by F1 at the smallest budget."))
}

# =============================================================================
# TABLE IV  complementary not-waste loss weight
# =============================================================================
nw <- d[d$family == "05_negative_weight_sweep", ]
if (nrow(nw)) {
  nw$w <- as.numeric(sub(".*_w", "", nw$config))
  nw <- nw[order(nw$w), ]
  t4 <- data.frame(
    `Weight` = nw$w,
    F1 = f3(nw$f1), Precision = f3(nw$precision),
    Recall = f4(nw$recall), FPR = f5(nw$fpr),
    `Implied precision` = f3(nw$implied_P),
    Gap = fmt_gap(nw$gap_x),
    check.names = FALSE)
  write_table(t4, "T6_negative_weight_sweep", "VI",
    "Effect of the Complementary Not-Waste Loss Weight",
    note = paste(
      "Raising the weight suppresses false positives but removes recall with",
      "them. Implied precision is the Bayes projection of the measured recall",
      "and FPR to the landscape prevalence of 0.045%. n.d. means the gap is",
      "not determined: at least one seed returned zero recall, so the required",
      "FPR is zero and the ratio is undefined. Scratch architecture throughout."))
}

# =============================================================================
# TABLE V  attainable operating set (recall-gap Pareto frontier)
# =============================================================================
fr <- pareto_frontier(d)
if (nrow(fr)) {
  t5 <- data.frame(
    Configuration = config_label(fr$config),
    Recall = f3(fr$recall),
    Precision = f3(fr$precision),
    FPR = f5(fr$fpr),
    `Implied precision` = f3(fr$implied_P),
    Gap = f1n(fr$gap_x),
    check.names = FALSE)
  write_table(t5, "T5_operating_frontier", "V",
    "Attainable Operating Set: the Recall-Gap Pareto Frontier",
    note = paste(
      "A configuration is on the frontier when no other configuration attains",
      "both a higher recall and a lower gap, so this is the complete set of",
      "non-dominated operating points over all 54 configurations. Trading",
      "recall for precision moves along the frontier; it does not leave it.",
      "The gap exceeds 11 at every attainable point, including those that",
      "discard more than nine tenths of the waste."))
}

# =============================================================================
# TABLE VI  every configuration ranked by gap   [appendix]
# =============================================================================
u <- d[is.finite(d$gap_x), ]
u <- u[order(u$gap_x), ]
t6 <- data.frame(
  Configuration = config_label(u$config),
  Family = u$family_label,
  `Val. F1` = f3(u$val_f1),
  F1 = f3(u$f1), Recall = f3(u$recall),
  FPR = f5(u$fpr),
  `Implied precision` = f3(u$implied_P),
  `Gap` = f1n(u$gap_x),
  `Usable recall` = ifelse(u$recall >= 0.35, "yes", "no"),
  check.names = FALSE)
write_table(t6, "T11_deployability_all", "XI",
  "Every Reported Configuration, Ranked by Deployability Gap",
  note = paste(
    "No configuration approaches a gap of 1.0. Configurations marked 'no'",
    "under usable recall return less than 35% of the waste and reach a low",
    "gap only by declining to detect; they are retained for completeness and",
    "excluded from the ranges quoted in the text."),
  placement = "appendix")

# =============================================================================
# TABLE VII  GeoBIA model development, segment level   [appendix]
# =============================================================================
g <- read_result_table("01_geobia/model_performance_comparison.csv")
g <- g[order(-g$F1_Score), ]
cl <- c(rf = "Random forest", nnet = "Neural network",
        nb = "Naive Bayes", svmLinear = "Linear SVM")
scn <- c(Optical_Only = "Optical bands only",
         Optimal_RFE  = "Optimal feature set (RFE)")
s1 <- data.frame(
  `Feature set` = unname(ifelse(g$Scenario %in% names(scn), scn[g$Scenario], g$Scenario)),
  Classifier    = unname(ifelse(g$Model %in% names(cl), cl[g$Model], g$Model)),
  Accuracy  = f3(g$Accuracy),
  Precision = f3(g$Precision),
  Recall    = f3(g$Sensitivity),
  F1        = f3(g$F1_Score),
  IoU       = f3(g$IoU_Waste),
  check.names = FALSE)
write_table(s1, "T2_geobia_model_development", "II",
  "GeoBIA Model Development, Evaluated at Segment Level",
  note = paste(
    "Segment-level accuracy measures how well segments are labelled once they",
    "exist. It is not comparable with the pixel-level results in Tables I and",
    "III to XI, which measure how well the resulting map delineates waste on",
    "the ground, and it is reported only to document how the GeoBIA classifier",
    "was chosen. The pixel-level GeoBIA result is in Table I."),
  placement = "body")

utils::write.csv(MANIFEST, file.path(TAB_DIR, "manifest.csv"), row.names = FALSE)

# =============================================================================
# numbers quoted in the Results text
# =============================================================================
cat("\n--- numbers for the manuscript text ---\n")
usable <- d[is.finite(d$gap_x) & d$recall >= 0.35, ]
cat(sprintf("configurations reported            : %d\n", nrow(d)))
cat(sprintf("usable (recall >= 0.35)            : %d\n", nrow(usable)))
cat(sprintf("F1 range over usable configurations: %.4f to %.4f\n",
            min(usable$f1), max(usable$f1)))
cat(sprintf("gap over usable configurations     : %.1f to %.1f\n",
            min(usable$gap_x), max(usable$gap_x)))
cat(sprintf("gap over the whole frontier        : %.1f to %.1f\n",
            min(fr$gap_x), max(fr$gap_x)))
cat(sprintf("selected model                     : %s\n", chosen$config))
cat(sprintf("  val F1 %.4f | test F1 %.4f (%.4f) | P %.4f R %.4f | FPR %.4f | gap %.1f\n",
            chosen$val_f1, chosen$f1, chosen$f1_sd, chosen$precision,
            chosen$recall, chosen$fpr, chosen$gap_x))
cat(sprintf("  rank on held-out F1              : %d of %d\n",
            match(chosen$config, d$config[order(-d$f1)]), nrow(d)))
bt <- d[which.max(d$f1), ]
cat(sprintf("best held-out F1 (not selected)    : %s %.4f, gap %.1f\n",
            bt$config, bt$f1, bt$gap_x))
lowgap <- usable[which.min(usable$gap_x), ]
cat(sprintf("lowest gap at usable recall        : %s gap %.1f, recall %.3f\n",
            lowgap$config, lowgap$gap_x, lowgap$recall))
hi <- usable[usable$recall >= 0.70, ]
cat(sprintf("lowest gap at recall >= 0.70       : %s gap %.1f, recall %.3f, F1 %.4f\n",
            hi$config[which.min(hi$gap_x)], min(hi$gap_x),
            hi$recall[which.min(hi$gap_x)], hi$f1[which.min(hi$gap_x)]))

g1 <- d[d$config == "geobia_otb_rf", ]
if (nrow(g1))
  cat(sprintf("GeoBIA %.4f vs selected %.4f       : %+.4f\n",
              g1$f1, chosen$f1, chosen$f1 - g1$f1))

if (exists("lb") && nrow(lb)) {
  s64 <- lb[lb$budget == min(lb$budget), ]
  bb  <- s64[which.max(s64$f1), ]
  scr <- lb[lb$key == "scratch", ]
  if (nrow(scr))
    cat(sprintf("%s at n=%d (%.4f) matches scratch at n=%d (%.4f): %.0fx fewer labels\n",
                bb$Backbone, bb$budget, bb$f1, max(scr$budget),
                scr$f1[scr$budget == max(scr$budget)], max(scr$budget) / bb$budget))
  cat(sprintf("spread across corpora: n=%d %.3f -> n=%d %.3f\n",
              min(budgets), diff(range(lb$f1[lb$budget == min(budgets)])),
              max(budgets), diff(range(lb$f1[lb$budget == max(budgets)]))))
}

fin <- d[d$family == "06_pretrained_backbones" & grepl("_finetuned$", d$config), ]
frz <- d[d$family == "06_pretrained_backbones" & grepl("_frozen$", d$config), ]
cat(sprintf("fine-tuned mean F1 %.4f vs frozen %.4f (%+.4f); FPR %.4f vs %.4f\n",
            mean(fin$f1), mean(frz$f1), mean(fin$f1) - mean(frz$f1),
            mean(fin$fpr), mean(frz$fpr)))

prev <- utils::read.csv(file.path(RESULTS_DIR, "03_unet_pytorch", "unet_f16_clean",
                                  "metrics_all_seeds.csv"), check.names = FALSE)
cat(sprintf("prevalence: test %.3f%%, val %.3f%%, landscape %.3f%% (%.0fx)\n",
            100 * unique(prev$prevalence[prev$split == "test"])[1],
            100 * unique(prev$prevalence[prev$split == "val"])[1],
            100 * P_LANDSCAPE,
            unique(prev$prevalence[prev$split == "test"])[1] / P_LANDSCAPE))
cat(sprintf("\ntables written to %s\n", TAB_DIR))
