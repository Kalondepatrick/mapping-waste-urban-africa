# =============================================================================
# Figure: GeoBIA, and what segmentation quality costs
# =============================================================================
# Two figures:
#
#   geobia_models          the eight GeoBIA model-development configurations
#                          (2 feature scenarios x 4 classifiers), segment level
#   geobia_vs_deep         GeoBIA with automated segmentation against the
#                          semantic-segmentation models, ALL on the same
#                          pixel-level basis (same split, same 216 clean tiles)
#
# The second figure is the one that matters. Everything downstream of
# 01_mask_vectorization.R used polygons vectorised from the reference MASK, so
# the polygon boundaries ARE the answer and precision is 1.000 by construction.
# That condition is excluded from all reported results (see EXCLUDED_CONFIGS in
# 00_figure_theme.R). 01_geobia/10_pixel_level_evaluation.R re-runs GeoBIA with
# the real OTB segments, which is what an automated method can achieve.
#
#   Rscript 07_figures/07_plot_geobia_comparison.R
#
# Writes results/figures/manuscript/geobia_models.{pdf,png}
#        results/figures/manuscript/geobia_vs_deep.{pdf,png}
#        results/geobia_segmentation_comparison.csv
# =============================================================================

local({
  p <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  while (!file.exists(file.path(p, "CITATION.cff")) && dirname(p) != p) p <- dirname(p)
  if (!file.exists(file.path(p, "CITATION.cff")))
    stop("Run this from inside the repository.")
  setwd(p)
})
source("07_figures/00_figure_theme.R")

# --- Figure 1: the GeoBIA model-development grid ------------------------------
g <- read_result_table("01_geobia/model_performance_comparison.csv")

classifier_label <- c(rf = "Random forest", nnet = "Neural network",
                      nb = "Naive Bayes", svmLinear = "Linear SVM")
scenario_label   <- c(Optical_Only = "Optical bands only",
                      Optimal_RFE  = "Optimal feature set (RFE)")
g$classifier <- unname(ifelse(g$Model %in% names(classifier_label),
                              classifier_label[g$Model], g$Model))
g$scenario   <- factor(unname(ifelse(g$Scenario %in% names(scenario_label),
                                     scenario_label[g$Scenario], g$Scenario)),
                       levels = unname(scenario_label))
ord <- tapply(g$F1_Score, g$classifier, max)
g$classifier <- factor(g$classifier, levels = names(sort(ord)))

p1 <- ggplot(g, aes(x = classifier, y = F1_Score, fill = scenario)) +
  geom_col(position = position_dodge(width = 0.74), width = 0.68,
           colour = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.3f", F1_Score)),
            position = position_dodge(width = 0.74), hjust = -0.18,
            size = 2.7, family = "serif") +
  scale_fill_manual(values = c("grey85", "grey40")) +
  coord_flip(ylim = c(0, 1.0)) +
  labs(x = NULL, y = expression(F[1]~"score, waste class (segment level)"),
       caption = paste("Segment-level accuracy during model development.",
                       "Not comparable with the pixel-level results.")) +
  paper_theme +
  theme(axis.text.y = element_text(size = 9),
        plot.caption = element_text(size = 8, hjust = 0, colour = "grey30"))

save_figure(p1, "geobia_models", height = 3.8)

# --- Figure 2: what segmentation costs, on the deep-learning basis -----------
d <- load_all_results("test", "argmax")
d <- d[is.finite(d$f1), ]

geo <- d[d$family == "02_geobia_pixel", ]
if (!nrow(geo))
  stop("No pixel-level GeoBIA results.\nRun: Rscript 01_geobia/10_pixel_level_evaluation.R --segments otb\n",
       "                        and: --segments oracle")

lab <- c(geobia_oracle_rf = "GeoBIA\noracle segments",
         geobia_otb_rf    = "GeoBIA\nOTB segments")
geo$label <- unname(ifelse(geo$config %in% names(lab), lab[geo$config], geo$config))

deep <- d[d$family != "02_geobia_pixel", ]
best <- deep[which.max(deep$f1), ]
best$label <- "Best deep model\n(no segmentation)"
scratch <- deep[grepl("^fx_scratch_n639$|^unet_f16_clean$", deep$config), ]
if (nrow(scratch)) {
  scratch <- scratch[which.max(scratch$f1), ]
  scratch$label <- "U-Net from scratch\n(no segmentation)"
}

cmp <- rbind(geo[, c("label", "f1", "f1_sd", "precision", "recall", "fpr", "gap_x")],
             best[, c("label", "f1", "f1_sd", "precision", "recall", "fpr", "gap_x")])
if (nrow(scratch))
  cmp <- rbind(cmp, scratch[, c("label", "f1", "f1_sd", "precision", "recall",
                                "fpr", "gap_x")])
cmp <- cmp[order(cmp$f1), ]
cmp$label <- factor(cmp$label, levels = cmp$label)
cmp$kind <- ifelse(grepl("GeoBIA", cmp$label), "GeoBIA", "Deep model")

utils::write.csv(cmp, file.path(RESULTS_DIR, "geobia_segmentation_comparison.csv"),
                 row.names = FALSE)

p2 <- ggplot(cmp, aes(x = label, y = f1, fill = kind)) +
  geom_col(width = 0.62, colour = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = f1 - f1_sd, ymax = f1 + f1_sd),
                width = 0.16, linewidth = 0.35) +
  geom_text(aes(label = sprintf("%.3f", f1)), vjust = -0.9, hjust = 0.5,
            size = 2.9, family = "serif") +
  scale_fill_manual(values = c("GeoBIA" = "grey82", "Deep model" = "grey38")) +
  scale_y_continuous(limits = c(0, 0.92)) +
  labs(x = NULL, y = expression(F[1]~"score, waste class (held-out test, pixel level)"),
       caption = paste(
         "All bars share one basis: the same spatial split, the same 216 clean",
         "test tiles, label 99 ignored, 3 seeds.\nOracle segments are polygons",
         "vectorised from the reference mask, so their boundaries are the",
         "answer. They are an upper bound, not a method.")) +
  paper_theme +
  theme(axis.text.x = element_text(size = 8, lineheight = 0.95),
        plot.caption = element_text(size = 7.5, hjust = 0, colour = "grey30"))

save_figure(p2, "geobia_vs_deep", height = 4.4)

# --- summary ------------------------------------------------------------------
cat("\n--- GeoBIA on the deep-learning basis (test, argmax, 3 seeds) ---\n")
for (i in seq_len(nrow(cmp))) {
  cat(sprintf("%-34s F1 %.4f +/- %.4f  P %.4f  R %.4f  FPR %.5f  gap %5.1fx\n",
              gsub("\n", " ", cmp$label[i]), cmp$f1[i], cmp$f1_sd[i],
              cmp$precision[i], cmp$recall[i], cmp$fpr[i], cmp$gap_x[i]))
}

t <- geo[geo$config == "geobia_otb_rf", ]
if (nrow(t))
  cat(sprintf("
Best deep model %.4f vs automated GeoBIA %.4f: difference %+.4f
",
              best$f1, t$f1, best$f1 - t$f1))

cat("
Methods note: a further condition was run in which the GeoBIA polygons
")
cat("were vectorised from the reference mask. It is excluded from all reported
")
cat("results because polygons cut from the reference cannot produce a false
")
cat("positive, so the condition measures nothing about the method. The run is
")
cat("retained at results/02_geobia_pixel/geobia_oracle_rf/ as the record.
")
