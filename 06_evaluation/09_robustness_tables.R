# =============================================================================
# Tables and figures for the acquisition-robustness experiment
# =============================================================================
#   python 06_evaluation/08_robustness_perturbation.py   # produces the metrics
#   Rscript 06_evaluation/09_robustness_tables.R         # this script
#
# Reports the fraction of native accuracy retained under each perturbation,
# which is the comparable quantity across models of differing native accuracy.
#
# Writes T9_resolution_robustness, T10_brightness_robustness and
# fig8_resolution_robustness, fig10_brightness_robustness
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
# Two sources: the locally evaluated 03_unet_pytorch checkpoints, and the
# pretrained-backbone checkpoints evaluated on the GPU cluster. Same script,
# same perturbations, same reference masks.
SRCS <- file.path(RESULTS_DIR, "09_robustness",
                  c("robustness_metrics.csv", "robustness_v6.csv"))
SRCS <- SRCS[file.exists(SRCS)]
if (!length(SRCS))
  stop("Run 06_evaluation/08_robustness_perturbation.py first.")

GSD_CM <- 3.81
r <- do.call(rbind, lapply(SRCS, utils::read.csv, stringsAsFactors = FALSE))

# Report one row per distinct encoder treatment. The hard-negative variants are
# a different training regime rather than a different encoder and are kept out
# of the body tables; they remain in the metrics file.
r <- r[!grepl("_hn$", r$config), ]
message("configurations reported: ", length(unique(r$config)))

# average over seeds
agg <- aggregate(cbind(f1, precision, recall, fpr, gap) ~ config + axis + level + condition,
                 data = r, FUN = mean)
agg$label <- config_label(agg$config)

base <- agg[agg$axis == "baseline", c("config", "f1", "fpr", "recall")]
names(base) <- c("config", "f1_base", "fpr_base", "recall_base")
agg <- merge(agg, base, by = "config")
agg$retained <- agg$f1 / agg$f1_base

# order configurations by how much they retain at twice the pixel size
ord_key <- agg[agg$axis == "resolution" & abs(agg$level - 2) < 1e-9,
               c("config", "retained")]
ord_key <- ord_key[order(-ord_key$retained), ]
CFG_ORDER <- ord_key$config
if (!length(CFG_ORDER)) CFG_ORDER <- unique(agg$config)

fmt3 <- function(x) ifelse(is.na(x), "--", sprintf("%.3f", x))

wide_for <- function(axis, levels_col_fmt) {
  s <- agg[agg$axis %in% c("baseline", axis), ]
  lv <- sort(unique(s$level[s$axis == axis]))
  base_lv <- if (axis == "resolution") 1 else 1
  out <- data.frame(Configuration = config_label(CFG_ORDER),
                    check.names = FALSE, stringsAsFactors = FALSE)
  out[[levels_col_fmt(base_lv)]] <- fmt3(
    agg$f1_base[match(CFG_ORDER, agg$config)])
  for (l in lv) {
    out[[levels_col_fmt(l)]] <- fmt3(vapply(CFG_ORDER, function(c0) {
      v <- s$f1[s$config == c0 & s$axis == axis & abs(s$level - l) < 1e-9]
      if (length(v)) v[1] else NA_real_
    }, numeric(1)))
  }
  out
}

# --- Table: spatial resolution -----------------------------------------------
res_lab <- function(l) sprintf("%.1f cm", GSD_CM * l)
t_res <- wide_for("resolution", res_lab)
keep2 <- agg[agg$axis == "resolution" & abs(agg$level - 2) < 1e-9, ]
t_res[["Retained at 7.6 cm"]] <- sprintf(
  "%.0f%%", 100 * keep2$retained[match(CFG_ORDER, keep2$config)])

# --- Table: brightness --------------------------------------------------------
br_lab <- function(l) sprintf("x%.2f", l)
t_bri <- wide_for("brightness", br_lab)
worst <- do.call(rbind, lapply(CFG_ORDER, function(c0) {
  s <- agg[agg$config == c0 & agg$axis == "brightness", ]
  if (!nrow(s)) return(data.frame(config = c0, w = NA_real_))
  data.frame(config = c0, w = min(s$retained))
}))
t_bri[["Worst case retained"]] <- sprintf(
  "%.0f%%", 100 * worst$w[match(CFG_ORDER, worst$config)])

# --- write -------------------------------------------------------------------
write_rob <- function(df, name, roman, title, note) {
  utils::write.csv(df, file.path(TAB_DIR, paste0(name, ".csv")), row.names = FALSE)
  align <- c(":---", rep("---:", ncol(df) - 1))
  md <- c(paste0("**TABLE ", roman, "**"), paste0("**", title, "**"), "",
          paste0("| ", paste(names(df), collapse = " | "), " |"),
          paste0("| ", paste(align, collapse = " | "), " |"),
          apply(df, 1, function(x) paste0("| ", paste(trimws(x), collapse = " | "), " |")),
          "", paste0("*", note, "*"))
  writeLines(md, file.path(TAB_DIR, paste0(name, ".md")))

  mf <- file.path(TAB_DIR, "manifest.csv")
  m <- utils::read.csv(mf, stringsAsFactors = FALSE)
  m <- m[m$file != paste0(name, ".csv"), ]
  m[nrow(m) + 1, ] <- list(paste0(name, ".csv"), roman, title, note, "body")
  utils::write.csv(m, mf, row.names = FALSE)
  message(sprintf("  TABLE %-5s %-34s %d rows x %d cols", roman,
                  paste0(name, ".csv"), nrow(df), ncol(df)))
}

n_seeds <- length(unique(r$seed))
write_rob(t_res, "T9_resolution_robustness", "IX",
  "Waste-Class F1 Under Simulated Loss of Spatial Resolution",
  sprintf(paste(
    "Held-out imagery reduced by area averaging and restored to the 256 x 256",
    "evaluation grid by bilinear interpolation, so every value is measured",
    "against the same reference mask. The first column is the native ground",
    "sample distance of %.2f cm. Mean of %d seeds. Rows ordered by the fraction",
    "of accuracy retained at twice the pixel size. Masks are never perturbed."),
    GSD_CM, n_seeds))

write_rob(t_bri, "T10_brightness_robustness", "X",
  "Waste-Class F1 Under a Change of Scene Brightness",
  sprintf(paste(
    "Held-out imagery multiplied by a gain and clipped to the 8-bit range,",
    "simulating the exposure and illumination differences between flights.",
    "x1.00 is the unperturbed image. Mean of %d seeds. Row order as in Table",
    "IX. Worst case retained is the lowest fraction of native-resolution F1",
    "over all gains tested."), n_seeds))

# --- figures -----------------------------------------------------------------
# Twelve curves on one panel is unreadable. The tables carry every
# configuration; the figures carry six that span the design: no pretraining,
# supervised ImageNet, remote-sensing pretraining, and web-scale
# self-supervised pretraining, each at the encoder treatment that matters.
FEATURED <- c(
  "torch_scratch",
  "v6_efficientnetb4imagenet_finetuned",
  "v6_satlasaerial_finetuned",
  "v6_dinov3satvitl_finetuned",
  "v6_dinov3satvitl_frozen",
  "v6_dinov3vitl_frozen")
FEATURED <- FEATURED[FEATURED %in% agg$config]

panel_data <- function(axis_name) {
  s <- agg[agg$config %in% FEATURED & agg$axis %in% c("baseline", axis_name), ]
  s$level[s$axis == "baseline"] <- 1
  s$Configuration <- factor(config_label(s$config),
                            levels = config_label(FEATURED))
  s
}

rob_scales <- list(
  scale_shape_manual(values = c(21, 24, 4, 22, 3, 23)),
  scale_linetype_manual(values = c("solid", "1343", "42", "22", "1333", "2262")),
  scale_y_continuous("Waste-class F1", limits = c(0, 0.8),
                     breaks = seq(0, 0.8, 0.2)),
  guides(shape = guide_legend(ncol = 1), linetype = guide_legend(ncol = 1)),
  ieee_theme,
  theme(legend.position = "bottom",
        legend.text = element_text(size = 5.6),
        legend.key.width = unit(16, "pt"),
        legend.key.height = unit(6.5, "pt"))
)

pr <- panel_data("resolution")
pr$gsd <- GSD_CM * pr$level
p6 <- ggplot(pr, aes(x = gsd, y = f1, group = Configuration)) +
  geom_line(aes(linetype = Configuration), linewidth = 0.4, colour = "grey25") +
  geom_point(aes(shape = Configuration), size = 1.4, fill = "white",
             colour = "grey15", stroke = 0.35) +
  scale_x_continuous("Effective ground sample distance (cm)", trans = "log2",
                     breaks = round(GSD_CM * sort(unique(pr$level)), 1)) +
  rob_scales
save_figure(p6, "fig8_resolution_robustness", width = IEEE_1COL, height = 3.3)

pb <- panel_data("brightness")
p8 <- ggplot(pb, aes(x = level, y = f1, group = Configuration)) +
  geom_vline(xintercept = 1, linewidth = 0.3, colour = "grey70") +
  geom_line(aes(linetype = Configuration), linewidth = 0.4, colour = "grey25") +
  geom_point(aes(shape = Configuration), size = 1.4, fill = "white",
             colour = "grey15", stroke = 0.35) +
  scale_x_continuous("Brightness gain applied to the held-out imagery",
                     breaks = sort(unique(pb$level))) +
  rob_scales
save_figure(p8, "fig10_brightness_robustness", width = IEEE_1COL, height = 3.3)

# --- numbers for the text ----------------------------------------------------
cat("\n--- robustness summary ---\n")
r2 <- agg[agg$axis == "resolution" & abs(agg$level - 2) < 1e-9, ]
r2 <- r2[order(-r2$retained), ]
for (i in seq_len(nrow(r2)))
  cat(sprintf("%-46s %.3f -> %.3f  (%.0f%% retained at 7.6 cm)\n",
              r2$label[i], r2$f1_base[i], r2$f1[i], 100 * r2$retained[i]))
cat("\nbrightness, worst case over all gains:\n")
for (c0 in CFG_ORDER) {
  s <- agg[agg$config == c0 & agg$axis == "brightness", ]
  if (!nrow(s)) next
  k <- which.min(s$retained)
  cat(sprintf("%-46s %.3f at gain %.2f  (%.0f%% retained)\n",
              s$label[k], s$f1[k], s$level[k], 100 * s$retained[k]))
}
cat(sprintf("\nfigure and tables written; seeds per point: %d\n", n_seeds))
