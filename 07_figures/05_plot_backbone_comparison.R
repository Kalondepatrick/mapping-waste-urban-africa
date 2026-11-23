# =============================================================================
# Figure: pretrained and foundation-model backbones, frozen vs fine-tuned
# =============================================================================
# The ten configurations of the RS-backbone experiment, all of which loaded
# 100% of their checkpoint tensors (verified in preflight_report.json).
#
#   Rscript 07_figures/05_plot_backbone_comparison.R
#
# Reads  results/06_pretrained_backbones/*/metrics_all_seeds.csv
# Writes results/figures/manuscript/backbone_comparison.{pdf,png}
#        results/figures/manuscript/backbone_precision_recall.{pdf,png}
#        results/06_pretrained_backbones/backbone_table.csv
# =============================================================================

local({
  p <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  while (!file.exists(file.path(p, "CITATION.cff")) && dirname(p) != p) p <- dirname(p)
  if (!file.exists(file.path(p, "CITATION.cff")))
    stop("Run this from inside the repository.")
  setwd(p)
})
source("07_figures/00_figure_theme.R")

d <- load_all_results("test", "argmax")
d <- d[d$family == "06_pretrained_backbones", ]
if (!nrow(d)) stop("No pretrained-backbone results found.")

meta        <- lapply(d$config, pretty_backbone)
d$backbone  <- vapply(meta, `[[`, character(1), "backbone")
d$mode      <- vapply(meta, `[[`, character(1), "mode")
d$mode      <- factor(d$mode, levels = c("frozen", "fine-tuned"))

# Order backbones by their best F1 so the figure reads top-down.
best <- tapply(d$f1, d$backbone, max)
d$backbone <- factor(d$backbone, levels = names(sort(best)))

utils::write.csv(
  d[order(-d$f1), c("backbone", "mode", "f1", "f1_sd", "precision", "recall",
                    "fpr", "iou", "prauc", "implied_P", "gap_x", "minutes")],
  file.path(RESULTS_DIR, "06_pretrained_backbones", "backbone_table.csv"),
  row.names = FALSE)

# --- Figure 1: F1 by backbone and training mode ------------------------------
p1 <- ggplot(d, aes(x = backbone, y = f1, fill = mode)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.66,
           colour = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = f1 - f1_sd, ymax = f1 + f1_sd),
                position = position_dodge(width = 0.72), width = 0.18,
                linewidth = 0.35) +
  scale_fill_manual(values = c("frozen" = "grey82", "fine-tuned" = "grey35")) +
  coord_flip(ylim = c(0.6, 0.78)) +
  labs(x = NULL, y = expression(F[1]~"score, waste class (held-out test)")) +
  paper_theme + theme(axis.text.y = element_text(size = 9))

save_figure(p1, "backbone_comparison", height = 4.0)

# --- Figure 2: the deployability view ----------------------------------------
# Accuracy on the development distribution against what it implies at the
# prevalence the map is actually deployed at.
p2 <- ggplot(d, aes(x = f1, y = gap_x)) +
  geom_hline(yintercept = min(d$gap_x), linetype = "dotted", colour = "grey45") +
  geom_point(aes(shape = mode, fill = mode), size = 3, colour = "black") +
  label_points(d, "backbone") +
  scale_shape_manual(values = c("frozen" = 21, "fine-tuned" = 24)) +
  scale_fill_manual(values = c("frozen" = "white", "fine-tuned" = "grey30")) +
  scale_y_continuous(limits = c(0, max(d$gap_x) * 1.12)) +
  labs(x = expression(F[1]~"score on held-out test data"),
       y = expression("Deployability gap ("*times*" too high an FPR)")) +
  paper_theme

save_figure(p2, "backbone_precision_recall")

# --- summary for the manuscript ----------------------------------------------
cat("\n--- pretrained backbones (test, argmax, 3 seeds) ---\n")
s <- d[order(-d$f1), ]
for (i in seq_len(nrow(s))) {
  cat(sprintf("%-32s %-11s F1 %.4f  FPR %.4f  gap %5.1fx\n",
              s$backbone[i], as.character(s$mode[i]),
              s$f1[i], s$fpr[i], s$gap_x[i]))
}
ft <- d[d$mode == "fine-tuned", ]; fz <- d[d$mode == "frozen", ]
cat(sprintf("\nfine-tuning gains on average %.4f F1 over a frozen encoder\n",
            mean(ft$f1) - mean(fz$f1)))
cat(sprintf("deployability gap across all ten: %.1fx to %.1fx (best %.1fx)\n",
            min(d$gap_x), max(d$gap_x), min(d$gap_x)))
cat("Accuracy varies by backbone. The gap does not close for any of them.\n")
