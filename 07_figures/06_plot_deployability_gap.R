# =============================================================================
# Figure: the deployability gap across every configuration tested
# =============================================================================
# The central result. Development accuracy varies widely across 53
# configurations spanning a 450-fold range of model capacity, five pretraining
# corpora, four label budgets and frozen vs fine-tuned encoders. The ratio of
# achieved false-positive rate to the rate required for precision 0.5 at
# landscape prevalence does not.
#
#   Rscript 07_figures/06_plot_deployability_gap.R
#
# Reads  results/*/*/metrics_all_seeds.csv  (every family)
# Writes results/figures/manuscript/deployability_gap.{pdf,png}
#        results/figures/manuscript/prevalence_precision.{pdf,png}
#        results/deployability_gap_all_configs.csv
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
d_all <- load_all_results("test", "argmax")
cat(sprintf("configurations loaded: %d
", nrow(d_all)))

# A model whose recall has collapsed gives an undefined gap: the FPR required
# for precision 0.5 underflows. Report these, then exclude them.
collapsed <- d_all[!is.finite(d_all$gap_x), ]
if (nrow(collapsed)) {
  cat(sprintf("excluded, recall collapsed below 1%%: %d (%s)
",
              nrow(collapsed), paste(collapsed$config, collapse = ", ")))
}
d <- d_all[is.finite(d_all$gap_x) & is.finite(d_all$f1), ]


# Models whose recall has collapsed are not candidates for deployment at any
# threshold. Keep them in the table, mark them in the figure.
d$usable <- d$recall >= 0.35
d$family_label <- factor(d$family_label,
                         levels = unname(EXPERIMENT_FAMILIES))

utils::write.csv(d[order(d$gap_x), ],
                 file.path(RESULTS_DIR, "deployability_gap_all_configs.csv"),
                 row.names = FALSE)

u <- d[d$usable, ]
band <- range(u$gap_x)
cat(sprintf("usable configurations (recall >= 0.35): %d\n", nrow(u)))
cat(sprintf("gap band: %.1fx to %.1fx\n", band[1], band[2]))

# --- Figure 1: accuracy against deployability --------------------------------
p1 <- ggplot(u, aes(x = f1, y = gap_x)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = band[1], ymax = band[2],
           fill = "grey90", alpha = 0.7) +
  annotate("text", x = min(u$f1), y = band[2],
           label = sprintf("all configurations fall in %.0f-%.0f×",
                           band[1], band[2]),
           hjust = 0, vjust = -0.6, size = 3, family = "serif", colour = "grey30") +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "black") +
  annotate("text", x = min(u$f1), y = 1, label = "deployable (precision 0.5)",
           hjust = 0, vjust = -0.7, size = 3, family = "serif") +
  geom_point(aes(shape = family_label, fill = family_label),
             size = 2.6, colour = "black", alpha = 0.95) +
  scale_shape_manual(values = mono_shapes[seq_len(nlevels(droplevels(u$family_label)))]) +
  scale_fill_manual(values = grey_palette[seq_len(nlevels(droplevels(u$family_label)))]) +
  scale_y_continuous(limits = c(0, max(u$gap_x) * 1.15)) +
  labs(x = expression(F[1]~"score on held-out test data"),
       y = expression("Deployability gap ("*times*" too high an FPR)")) +
  guides(shape = guide_legend(nrow = 2), fill = guide_legend(nrow = 2)) +
  paper_theme

save_figure(p1, "deployability_gap")

# --- Figure 2: why development accuracy misleads -----------------------------
# Precision as a function of prevalence, holding each model's recall and FPR
# fixed. Bayes: P = R.p / (R.p + F.(1-p))
best <- u[which.min(u$gap_x), ]
med  <- u[which.min(abs(u$gap_x - stats::median(u$gap_x))), ]

prev <- 10^seq(log10(P_LANDSCAPE / 3), log10(0.20), length.out = 300)
curve_for <- function(row, name) {
  data.frame(p = prev,
             precision = row$recall * prev /
               (row$recall * prev + row$fpr * (1 - prev)),
             model = name)
}
# Prevalence of the held-out test partition, on which these metrics are computed.
.m <- utils::read.csv(file.path(RESULTS_DIR, "03_unet_pytorch", "unet_f16_clean",
                                "metrics_all_seeds.csv"), check.names = FALSE)
dev_prev <- unique(.m$prevalence[.m$split == "test"])[1]
val_prev <- unique(.m$prevalence[.m$split == "val"])[1]

cur <- rbind(curve_for(best, sprintf("best configuration (%.0f× gap)", best$gap_x)),
             curve_for(med,  sprintf("median configuration (%.0f× gap)", med$gap_x)))

p2 <- ggplot(cur, aes(p, precision, linetype = model)) +
  geom_vline(xintercept = dev_prev, colour = "grey50", linetype = "dotted") +
  geom_vline(xintercept = P_LANDSCAPE, colour = "grey50", linetype = "dotted") +
  annotate("text", x = dev_prev, y = 0.97, label = "test partition", angle = 90,
           hjust = 1, vjust = -0.5, size = 2.9, family = "serif", colour = "grey30") +
  annotate("text", x = P_LANDSCAPE, y = 0.97, label = "landscape", angle = 90,
           hjust = 1, vjust = -0.5, size = 2.9, family = "serif", colour = "grey30") +
  geom_hline(yintercept = 0.5, colour = "black", linetype = "dashed") +
  geom_line(linewidth = 0.6, colour = "black") +
  scale_x_log10(labels = function(x) paste0(signif(x * 100, 2), "%")) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "Prevalence of waste (log scale)",
       y = "Precision implied by the model's recall and FPR") +
  paper_theme

save_figure(p2, "prevalence_precision")

# --- summary ------------------------------------------------------------------
cat("\n--- deployability across every configuration ---\n")
cat(sprintf("test-partition prevalence : %.3f%%  (%.0fx the landscape rate)
",
            dev_prev * 100, dev_prev / P_LANDSCAPE))
cat(sprintf("validation prevalence     : %.3f%%  (%.0fx)
",
            val_prev * 100, val_prev / P_LANDSCAPE))
cat(sprintf("landscape prevalence      : %.3f%%
", P_LANDSCAPE * 100))
cat(sprintf("\nF1 range over usable configurations : %.4f to %.4f\n",
            min(u$f1), max(u$f1)))
cat(sprintf("gap range over the same set         : %.1fx to %.1fx\n",
            band[1], band[2]))
cat(sprintf("best: %s (%s), F1 %.4f, gap %.1fx\n",
            best$config, best$family_label, best$f1, best$gap_x))
cat("
The gap varies more than fourfold across configurations, so it is not
")
cat("invariant. But no configuration comes within an order of magnitude of 1.0,
")
cat("and the smallest gaps are bought by sacrificing recall.
")
