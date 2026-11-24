# =============================================================================
# Manuscript figures 1 to 5
# =============================================================================
#   Rscript 07_figures/10_manuscript_figures.R
#
#   1  accuracy of the three mapping approaches
#   2  accuracy against annotation budget, by pretraining corpus
#   3  frozen against fine-tuned encoders
#   4  precision against prevalence for the selected model
#   5  the attainable operating set
#
# Numbered in order of first citation in docs/RESULTS.md. Model selection
# follows 06_evaluation/06_build_manuscript_tables.R: highest validation F1,
# then lowest deployability gap within one standard error of it.
# =============================================================================

local({
  p <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  while (!file.exists(file.path(p, "CITATION.cff")) && dirname(p) != p) p <- dirname(p)
  if (!file.exists(file.path(p, "CITATION.cff")))
    stop("Run this from inside the repository.")
  setwd(p)
})
suppressMessages(source("07_figures/00_figure_theme.R"))

d  <- load_with_validation("argmax")
va <- load_all_results("val", "argmax")
d$val_sd <- va$f1_sd[match(d$config, va$config)]

vbest  <- d[which.max(d$val_f1), ]
val_se <- vbest$val_sd / sqrt(vbest$seeds)
tied   <- d[!is.na(d$val_f1) & d$val_f1 >= vbest$val_f1 - val_se, ]
CHOSEN <- tied[which.min(tied$gap_x), ]
message("selected: ", CHOSEN$config)

# =============================================================================
# Fig. 1  accuracy of the three mapping approaches
# =============================================================================
f1_rows <- rbind(
  d[d$config == "geobia_otb_rf", ],
  d[d$config == "unet_f16_clean", ],
  d[d$config == "torch_scratch", ],
  CHOSEN)
f1_rows$label <- c("GeoBIA\n(OTB segments, RF)",
                   "U-Net from scratch\n(TensorFlow)",
                   "U-Net from scratch\n(PyTorch)",
                   paste0(pretty_backbone(CHOSEN$config)$backbone, "\nfine-tuned"))
f1_rows$label <- factor(f1_rows$label, levels = rev(f1_rows$label))

p1 <- ggplot(f1_rows, aes(x = f1, y = label)) +
  geom_errorbarh(aes(xmin = f1 - f1_sd, xmax = f1 + f1_sd),
                 height = 0.16, linewidth = 0.35, colour = "grey30") +
  geom_point(size = 2.1, shape = 21, fill = "grey20", colour = "black",
             stroke = 0.35) +
  geom_text(aes(label = sprintf("%.3f", f1)), hjust = -0.45, vjust = -0.9,
            size = 2.3, family = IEEE_FAMILY, colour = "grey20") +
  scale_x_continuous("Waste-class F1 on the held-out partition",
                     limits = c(0.35, 0.85), breaks = seq(0.4, 0.8, 0.1)) +
  ylab(NULL) +
  ieee_theme + theme(panel.grid.major.y = element_line(linewidth = 0.2,
                                                       colour = "grey92"))
save_figure(p1, "fig1_approach_comparison", width = IEEE_1COL, height = 2.2)

# =============================================================================
# Fig. 2  accuracy against annotation budget
# =============================================================================
lb <- d[d$family == "04_label_budget_factorial", ]
lb$budget   <- as.integer(sub(".*_n", "", lb$config))
lb$key      <- sub("^fx_", "", sub("_n[0-9]+$", "", lb$config))
lb$Backbone <- vapply(lb$key, function(k) {
  b <- pretty_backbone(k)$backbone
  if (identical(b, "scratch")) "U-Net from scratch" else b
}, character(1))

full_scratch <- lb$f1[lb$key == "scratch" & lb$budget == max(lb$budget)]
ord <- tapply(lb$f1[lb$budget == min(lb$budget)],
              lb$Backbone[lb$budget == min(lb$budget)], mean)
lb$Backbone <- factor(lb$Backbone, levels = names(sort(ord, decreasing = TRUE)))

p2 <- ggplot(lb, aes(x = budget, y = f1, group = Backbone)) +
  geom_hline(yintercept = full_scratch, linetype = "dotted",
             linewidth = 0.35, colour = "grey45") +
  geom_line(aes(linetype = Backbone), linewidth = 0.4, colour = "grey25") +
  geom_point(aes(shape = Backbone), size = 1.5, fill = "white",
             colour = "grey15", stroke = 0.35) +
  annotate("text", x = 64, y = full_scratch, vjust = -0.7, hjust = 0,
           label = "scratch, full budget", size = 2.1, family = IEEE_FAMILY,
           colour = "grey35") +
  scale_x_continuous("Annotated training tiles", trans = "log2",
                     breaks = sort(unique(lb$budget))) +
  scale_y_continuous("Waste-class F1", limits = c(0.54, 0.77),
                     breaks = seq(0.55, 0.75, 0.05)) +
  scale_shape_manual(values = rep(c(21, 22, 23, 24, 25, 1, 4), 2)) +
  scale_linetype_manual(values = rep(c("solid", "22", "42", "1343", "73",
                                       "2262", "12"), 2)) +
  guides(shape = guide_legend(ncol = 2, byrow = TRUE),
         linetype = guide_legend(ncol = 2, byrow = TRUE)) +
  ieee_theme +
  theme(legend.position = "bottom",
        legend.key.width = unit(16, "pt"),
        legend.key.height = unit(7, "pt"),
        legend.text = element_text(size = 6.0),
        legend.spacing.y = unit(0, "pt"))
save_figure(p2, "fig2_label_budget", width = IEEE_1COL, height = 2.9)

# =============================================================================
# Fig. 3  frozen against fine-tuned encoders
# =============================================================================
bb <- d[d$family == "06_pretrained_backbones", ]
bb$mode     <- ifelse(grepl("_frozen$", bb$config), "frozen", "fine-tuned")
bb$Backbone <- vapply(bb$config, function(k) pretty_backbone(k)$backbone, character(1))
wide <- reshape(bb[, c("Backbone", "mode", "f1")], idvar = "Backbone",
                timevar = "mode", direction = "wide")
names(wide) <- c("Backbone", "f1_frozen", "f1_ft")
wide <- wide[order(wide$f1_ft), ]
wide$Backbone <- factor(wide$Backbone, levels = wide$Backbone)
bb$Backbone <- factor(bb$Backbone, levels = levels(wide$Backbone))

p3 <- ggplot() +
  geom_segment(data = wide,
               aes(y = Backbone, yend = Backbone, x = f1_frozen, xend = f1_ft),
               linewidth = 0.4, colour = "grey55",
               arrow = arrow(length = unit(3, "pt"), type = "closed")) +
  geom_point(data = bb, aes(x = f1, y = Backbone, shape = mode, fill = mode),
             size = 1.9, colour = "black", stroke = 0.35) +
  scale_shape_manual(values = c(frozen = 21, `fine-tuned` = 23)) +
  scale_fill_manual(values = c(frozen = "white", `fine-tuned` = "grey15")) +
  scale_x_continuous("Waste-class F1 on the held-out partition",
                     limits = c(0.65, 0.78)) +
  ylab(NULL) +
  ieee_theme + theme(axis.text.y = element_text(size = 6.4))
save_figure(p3, "fig3_frozen_vs_finetuned", width = IEEE_1COL, height = 2.1)

# =============================================================================
# Fig. 5  the attainable operating set
# =============================================================================
fr <- pareto_frontier(d)
pts <- d[is.finite(d$gap_x) & is.finite(d$recall), ]
pts$on_front <- pts$config %in% fr$config
sel <- pts[pts$config == CHOSEN$config, ]

p4 <- ggplot(pts, aes(x = recall, y = gap_x)) +
  geom_hline(yintercept = 1, linetype = "solid", linewidth = 0.45,
             colour = "grey15") +
  annotate("text", x = 0.02, y = 1, vjust = -0.75, hjust = 0,
           label = "directly usable map (gap = 1)", size = 2.2,
           family = IEEE_FAMILY, colour = "grey15") +
  geom_point(data = pts[!pts$on_front, ], size = 1.1, shape = 1,
             colour = "grey65", stroke = 0.3) +
  geom_line(data = fr[order(fr$recall), ], linewidth = 0.4, colour = "grey25") +
  geom_point(data = fr, size = 1.7, shape = 21, fill = "grey25",
             colour = "black", stroke = 0.35) +
  geom_point(data = sel, size = 2.6, shape = 23, fill = "white",
             colour = "black", stroke = 0.5) +
  annotate("text", x = sel$recall, y = sel$gap_x, label = "selected model",
           hjust = 1.12, vjust = -0.9, size = 2.2, family = IEEE_FAMILY) +
  scale_x_continuous("Recall on the held-out partition", limits = c(0, 0.88),
                     breaks = seq(0, 0.8, 0.2)) +
  scale_y_continuous("Deployability gap",
                     trans = "log10", breaks = c(1, 3, 10, 30, 100),
                     limits = c(0.8, 100)) +
  ieee_theme +
  theme(panel.grid.major = element_line(linewidth = 0.18, colour = "grey93"))
save_figure(p4, "fig5_operating_frontier", width = IEEE_1COL, height = 2.6)

# =============================================================================
# Fig. 4  precision against prevalence
# =============================================================================
best_gap <- {
  u <- d[is.finite(d$gap_x) & d$recall >= 0.70, ]
  u[which.min(u$gap_x), ]
}
curve_for <- function(r, label) {
  p <- 10^seq(log10(2e-5), log10(0.1), length.out = 400)
  data.frame(p = p, model = label,
             precision = r$recall * p / (r$recall * p + r$fpr * (1 - p)))
}
cv <- rbind(curve_for(CHOSEN, "selected model"),
            curve_for(best_gap, "lowest gap at recall > 0.70"))

prev_test <- 0.03301
p5 <- ggplot(cv, aes(x = p, y = precision, linetype = model)) +
  geom_vline(xintercept = P_LANDSCAPE, linewidth = 0.3, colour = "grey45") +
  geom_vline(xintercept = prev_test, linewidth = 0.3, colour = "grey45") +
  geom_hline(yintercept = 0.5, linewidth = 0.45, colour = "grey15") +
  geom_line(linewidth = 0.45, colour = "grey15") +
  annotate("text", x = P_LANDSCAPE, y = 0.97, angle = 90, hjust = 1, vjust = -0.5,
           label = "landscape 0.045%", size = 2.1, family = IEEE_FAMILY,
           colour = "grey30") +
  annotate("text", x = prev_test, y = 0.97, angle = 90, hjust = 1, vjust = 1.3,
           label = "annotated tiles 3.3%", size = 2.1, family = IEEE_FAMILY,
           colour = "grey30") +
  annotate("text", x = 2.2e-5, y = 0.5, vjust = -0.6, hjust = 0,
           label = "precision 0.5", size = 2.1, family = IEEE_FAMILY,
           colour = "grey15") +
  scale_x_continuous("Prevalence of waste (% of pixels)",
                     trans = "log10",
                     breaks = c(1e-4, 1e-3, 1e-2, 1e-1),
                     labels = c("0.01%", "0.1%", "1%", "10%")) +
  scale_y_continuous("Precision implied by the Bayes projection",
                     limits = c(0, 1)) +
  scale_linetype_manual(values = c("solid", "42")) +
  ieee_theme + theme(legend.position = "bottom")
save_figure(p5, "fig4_prevalence_precision", width = IEEE_1COL, height = 2.5)

message("\nall manuscript figures written to ", FIG_DIR)
