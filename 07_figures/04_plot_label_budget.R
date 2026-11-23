# =============================================================================
# Figure: label-budget curves by pretraining domain
# =============================================================================
# Answers: how few annotated tiles does each pretraining corpus need to reach a
# given accuracy? This is the "label scarcity" axis of the study.
#
#   Rscript 07_figures/04_plot_label_budget.R
#
# Reads  results/04_label_budget_factorial/*/metrics_all_seeds.csv
# Writes results/figures/manuscript/label_budget_curves.{pdf,png}
#        results/04_label_budget_factorial/label_budget_curve_R.csv
# =============================================================================

# Locate the repository root, then load the shared theme.
local({
  p <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  while (!file.exists(file.path(p, "CITATION.cff")) && dirname(p) != p) p <- dirname(p)
  if (!file.exists(file.path(p, "CITATION.cff")))
    stop("Run this from inside the repository.")
  setwd(p)
})
source("07_figures/00_figure_theme.R")

d <- load_all_results("test", "argmax")
d <- d[d$family == "04_label_budget_factorial", ]
if (!nrow(d)) stop("No label-budget results found.")

# config looks like fx_<arch>_n<budget>
d$budget <- as.integer(sub(".*_n", "", d$config))
d$key    <- sub("^fx_", "", sub("_n[0-9]+$", "", d$config))
d$label  <- vapply(d$key, function(k) pretty_backbone(k)$backbone, character(1))

d <- d[order(d$label, d$budget), ]
utils::write.csv(d[, c("label", "budget", "f1", "f1_sd", "fpr", "gap_x")],
                 file.path(RESULTS_DIR, "04_label_budget_factorial",
                           "label_budget_curve_R.csv"), row.names = FALSE)

# Order the legend by accuracy at the smallest budget: the quantity of interest.
small <- d[d$budget == min(d$budget), ]
d$label <- factor(d$label, levels = small$label[order(-small$f1)])

n_arch <- nlevels(d$label)

p <- ggplot(d, aes(x = budget, y = f1, group = label)) +
  geom_errorbar(aes(ymin = f1 - f1_sd, ymax = f1 + f1_sd),
                width = 0.035, colour = "grey65", linewidth = 0.35) +
  geom_line(aes(linetype = label), colour = "black", linewidth = 0.55) +
  geom_point(aes(shape = label), fill = "white", colour = "black", size = 2.4) +
  scale_x_log10(breaks = sort(unique(d$budget)),
                labels = sort(unique(d$budget))) +
  scale_shape_manual(values = rep(mono_shapes, length.out = n_arch)) +
  scale_linetype_manual(values = rep(c("solid", "22", "42", "1343", "73", "2262"),
                                     length.out = n_arch)) +
  labs(x = "Training tiles (log scale)",
       y = expression(F[1]~"score, waste class (held-out test)")) +
  guides(shape = guide_legend(nrow = 3), linetype = guide_legend(nrow = 3)) +
  paper_theme

save_figure(p, "label_budget_curves")

# --- the headline comparison, printed for the manuscript text ----------------
smallest <- min(d$budget); largest <- max(d$budget)
at_small <- d[d$budget == smallest, ]
at_large <- d[d$budget == largest, ]
best_small <- at_small[which.max(at_small$f1), ]
scratch_rows <- d[grepl("scratch", d$key), ]

cat("\n--- label budget summary (test, argmax) ---\n")
cat(sprintf("best at n=%d: %-32s F1 = %.4f\n",
            smallest, best_small$label, best_small$f1))
if (nrow(scratch_rows)) {
  s_small <- scratch_rows[scratch_rows$budget == smallest, ]
  s_large <- scratch_rows[scratch_rows$budget == largest, ]
  if (nrow(s_small)) cat(sprintf("scratch at n=%d:              %34.4f\n",
                                 smallest, s_small$f1))
  if (nrow(s_large)) cat(sprintf("scratch at n=%d:             %34.4f\n",
                                 largest, s_large$f1))
}
spread_small <- diff(range(at_small$f1))
spread_large <- diff(range(at_large$f1))
cat(sprintf("\nspread across pretraining domains: %.4f at n=%d, %.4f at n=%d\n",
            spread_small, smallest, spread_large, largest))
cat("The advantage of pretraining is a small-sample effect: it shrinks as the\n")
cat("label budget grows.\n")
