# =============================================================================
# Land cover confused with waste
# =============================================================================
#   python 06_evaluation/10_confusion_by_landcover.py   # produces the counts
#   Rscript 06_evaluation/11_landcover_tables.R         # this script
#
# Commission rate is intrinsic confusability; share of errors depends on class
# area. A class can be highly confusable and contribute few errors, or barely
# confusable and dominate the error budget, so both are reported.
#
# Writes results/tables/T8_landcover_confusion.{csv,md} and
# results/figures/manuscript/fig6_landcover_confusion.{pdf,png}
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
SRCS <- file.path(RESULTS_DIR, "10_landcover_confusion",
                  c("landcover_confusion.csv", "landcover_v6.csv"))
SRCS <- SRCS[file.exists(SRCS)]
if (!length(SRCS))
  stop("Run 06_evaluation/10_confusion_by_landcover.py first.")

r <- do.call(rbind, lapply(SRCS, utils::read.csv, stringsAsFactors = FALSE))
r <- r[!grepl("_hn$", r$config), ]          # hard negatives are a separate regime
n_cfg <- length(unique(r$config))
message("configurations: ", n_cfg, "   seeds per configuration: ",
        length(unique(r$seed)))

# mean over seeds first, so each configuration counts once
# na.action = na.pass: the waste row carries NA for share_of_false_positives
# by construction, and the formula interface would otherwise drop it silently.
per_cfg <- aggregate(cbind(rate, share_of_false_positives, class_pixels) ~
                       config + raw_label + class_name, data = r,
                     FUN = function(x) mean(x, na.rm = TRUE),
                     na.action = stats::na.pass)

scene <- unique(per_cfg[, c("raw_label", "class_name", "class_pixels")])
scene$pct <- 100 * scene$class_pixels / sum(scene$class_pixels)

agg <- do.call(rbind, lapply(split(per_cfg, per_cfg$raw_label), function(s) {
  data.frame(
    raw_label = s$raw_label[1], class_name = s$class_name[1],
    rate_mean = mean(s$rate), rate_min = min(s$rate), rate_max = max(s$rate),
    share_mean = mean(s$share_of_false_positives),
    share_min = min(s$share_of_false_positives),
    share_max = max(s$share_of_false_positives),
    stringsAsFactors = FALSE)
}))
agg <- merge(agg, scene[, c("raw_label", "pct")], by = "raw_label")

waste <- agg[agg$raw_label == 1, ]
other <- agg[agg$raw_label != 1, ]
other <- other[order(-other$rate_mean), ]

pc <- function(x) sprintf("%.2f", 100 * x)
out <- data.frame(
  `Land-cover class` = other$class_name,
  `Share of scene`   = sprintf("%.2f", other$pct),
  `Commission rate`  = pc(other$rate_mean),
  `Range`            = paste0(pc(other$rate_min), "-", pc(other$rate_max)),
  `Share of errors`  = sprintf("%.1f", 100 * other$share_mean),
  `Range `           = paste0(sprintf("%.1f", 100 * other$share_min), "-",
                              sprintf("%.1f", 100 * other$share_max)),
  check.names = FALSE, stringsAsFactors = FALSE)

title <- "Land Cover Confused With Waste on the Held-Out Partition"
note <- sprintf(paste(
  "All figures are percentages, computed on the 216 spatially clean held-out",
  "tiles with label 99 excluded, under the argmax rule. Share of scene is the",
  "fraction of assessed pixels belonging to the class. Commission rate is the",
  "fraction of that class predicted waste, and is independent of how much of",
  "the class the scene contains. Share of errors is the fraction of all",
  "false-positive pixels whose true class is the one named. Means over %d",
  "configurations, each itself a mean over three seeds, with the range across",
  "configurations. For reference, recall on the waste class itself averages",
  "%.1f%%. Rows ordered by commission rate."),
  n_cfg, 100 * waste$rate_mean)

utils::write.csv(out, file.path(TAB_DIR, "T8_landcover_confusion.csv"),
                 row.names = FALSE)
align <- c(":---", rep("---:", ncol(out) - 1))
writeLines(c("**TABLE VIII**", paste0("**", title, "**"), "",
             paste0("| ", paste(names(out), collapse = " | "), " |"),
             paste0("| ", paste(align, collapse = " | "), " |"),
             apply(out, 1, function(x) paste0("| ", paste(trimws(x), collapse = " | "), " |")),
             "", paste0("*", note, "*")),
           file.path(TAB_DIR, "T8_landcover_confusion.md"))

mf <- file.path(TAB_DIR, "manifest.csv")
m <- utils::read.csv(mf, stringsAsFactors = FALSE)
m <- m[m$file != "T8_landcover_confusion.csv", ]
m[nrow(m) + 1, ] <- list("T8_landcover_confusion.csv", "VIII", title, note, "body")
utils::write.csv(m, mf, row.names = FALSE)
message("  TABLE VIII  T8_landcover_confusion.csv   ", nrow(out), " rows")

# --- figure ------------------------------------------------------------------
# Rate against area, so that the two kinds of problem separate visually: high
# rate and small area (intrinsically confusable, few pixels) against low rate
# and large area (barely confusable, dominates the error budget by extent).
pl <- other
pl$label <- sub(" surface", "", pl$class_name)
p8 <- ggplot(pl, aes(x = pct, y = 100 * rate_mean)) +
  geom_linerange(aes(ymin = 100 * rate_min, ymax = 100 * rate_max),
                 linewidth = 0.35, colour = "grey60") +
  geom_point(aes(size = 100 * share_mean), shape = 21, fill = "grey25",
             colour = "black", stroke = 0.35) +
  label_points(pl, "label", size = 2.1) +
  scale_x_continuous("Share of the assessed scene (%)", trans = "log10") +
  scale_y_continuous("Pixels of the class predicted waste (%)", trans = "log10") +
  scale_size_area("share of all false positives (%)", max_size = 4.2,
                  breaks = c(5, 15, 35)) +
  ieee_theme +
  theme(legend.position = "bottom",
        legend.title = element_text(size = IEEE_BASE - 2),
        legend.text = element_text(size = IEEE_BASE - 2.5))
save_figure(p8, "fig6_landcover_confusion", width = IEEE_1COL, height = 3.0)

cat("\n--- land cover confused with waste ---\n")
cat(sprintf("%-26s %8s %12s %14s\n", "class", "scene %", "commission %", "share of FP %"))
for (i in seq_len(nrow(other)))
  cat(sprintf("%-26s %8.2f %12.2f %14.1f\n", other$class_name[i], other$pct[i],
              100 * other$rate_mean[i], 100 * other$share_mean[i]))
cat(sprintf("\nwaste recall for reference: %.1f%%\n", 100 * waste$rate_mean))
