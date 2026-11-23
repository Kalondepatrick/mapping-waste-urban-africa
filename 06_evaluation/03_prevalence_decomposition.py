"""Does the precision collapse follow from the base rate alone?

Uses only per-tile pixel confusion counts already written by training, so it
needs no model and no GPU.

Method: tiles are split into waste-bearing and pure-background. Re-weighting
the background tiles by w sweeps the waste prevalence of the pooled sample
from its development value down towards landscape values, without changing
the classifier at all. At each prevalence we compare

  observed  precision = TP / (TP + FP)
  Bayes     precision = R*p / (R*p + F*(1-p))   using the development R and F

Agreement means the collapse is arithmetic (base-rate effect). A shortfall of
observed below Bayes is the extra commission the landscape causes because it
contains confusers the development data did not (shift effect).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/03_unet_pytorch/unet_f16_clean/seed_1"
RULE = sys.argv[2] if len(sys.argv) > 2 else "argmax"
LAND = 0.00045

rows = []
for split in ("val", "test"):
    f = os.path.join(RUN, f"per_tile_counts_{split}_{RULE}.csv")
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    bg = d["n_waste"] == 0
    tp, fn = d["tp"].sum(), d["fn"].sum()
    fp_w, tn_w = d.loc[~bg, "fp"].sum(), d.loc[~bg, "tn"].sum()
    fp_b, tn_b = d.loc[bg, "fp"].sum(), d.loc[bg, "tn"].sum()

    R = tp / max(tp + fn, 1)
    F0 = (fp_w + fp_b) / max(fp_w + fp_b + tn_w + tn_b, 1)
    Fb = fp_b / max(fp_b + tn_b, 1)
    Fw = fp_w / max(fp_w + tn_w, 1)
    print(f"\n=== {split} ({RULE}) : {len(d)} tiles, "
          f"{int((~bg).sum())} with waste, {int(bg.sum())} background ===")
    print(f"  recall                         {R:.4f}")
    print(f"  FPR, all negatives             {F0:.5f}")
    print(f"  FPR, negatives in waste tiles  {Fw:.5f}")
    print(f"  FPR, negatives in background   {Fb:.5f}")

    for w in np.logspace(0, 4.5, 40):
        FP = fp_w + w * fp_b
        TN = tn_w + w * tn_b
        n = tp + fn + FP + TN
        p = (tp + fn) / n
        obs = tp / max(tp + FP, 1e-12)
        bayes = R * p / (R * p + F0 * (1 - p))
        rows.append({"split": split, "w": w, "prevalence": p,
                     "precision_observed": obs, "precision_bayes": bayes,
                     "recall": R, "fpr_dev": F0, "fpr_bg": Fb})

df = pd.DataFrame(rows)
out = os.path.join(RUN, f"prevalence_sweep_{RULE}.csv")
df.to_csv(out, index=False)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for split, m in (("val", "o"), ("test", "s")):
    s = df[df.split == split]
    if s.empty:
        continue
    ax.plot(s.prevalence, s.precision_observed, m + "-", ms=3, lw=1.5,
            label=f"{split}: observed")
    ax.plot(s.prevalence, s.precision_bayes, "--", lw=1.2,
            label=f"{split}: Bayes from dev recall/FPR")
ax.axvline(LAND, color="grey", ls=":", lw=1.2)
ax.text(LAND * 1.15, 0.9, f"landscape\np = {LAND:g}", fontsize=8, color="grey")
ax.axhline(0.020, color="crimson", ls=":", lw=1.2)
ax.text(1.5e-4, 0.035, "observed landscape precision 0.020",
        fontsize=8, color="crimson")
ax.set(xscale="log", xlabel="Waste prevalence in the evaluated sample",
       ylabel="Precision", ylim=(0, 1),
       title="Precision falls with prevalence, with the classifier unchanged")
ax.grid(True, ls="--", alpha=0.4)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
png = os.path.join(RUN, f"prevalence_sweep_{RULE}.png")
fig.savefig(png, dpi=300)

print(f"\nwrote {out}")
print(f"wrote {png}")
print("\n--- precision at selected prevalences ---")
for split in df.split.unique():
    s = df[df.split == split]
    print(f"  {split}:")
    for target in (0.05, 0.01, 0.002, LAND):
        i = (s.prevalence - target).abs().idxmin()
        r = s.loc[i]
        print(f"    p={r.prevalence:.5f}   observed {r.precision_observed:.4f}"
              f"   Bayes {r.precision_bayes:.4f}")
