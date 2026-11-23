"""Collect every finished run into the model-comparison table.

Columns answer the question the paper asks: does a stronger model reach the
false-positive rate that landscape-scale use actually requires?
"""
import glob
import os

import pandas as pd

LAND = 0.00045
rows = []
for f in sorted(glob.glob("results/03_unet_pytorch/*/metrics_all_seeds.csv")):
    name = os.path.basename(os.path.dirname(f))
    d = pd.read_csv(f)
    ip = [c for c in d.columns if c.startswith("implied")][0]
    rq = [c for c in d.columns if c.startswith("required")][0]
    for (split, rule), g in d.groupby(["split", "rule"]):
        rows.append({
            "model": name, "split": split, "rule": rule, "seeds": len(g),
            "precision": g.precision.mean(), "recall": g.recall.mean(),
            "f1": g.f1.mean(), "f1_sd": g.f1.std(),
            "fpr": g.fpr.mean(),
            "implied_P_landscape": g[ip].mean(),
            "required_fpr_P50": g[rq].mean(),
            "fpr_gap_x": g.fpr.mean() / g[rq].mean(),
            "minutes": g.minutes.mean() if "minutes" in g else float("nan"),
        })
t = pd.DataFrame(rows)
t.to_csv("results/03_unet_pytorch/model_comparison.csv", index=False)

for split in ("test", "val"):
    s = t[(t.split == split) & (t.rule == "argmax")]
    if s.empty:
        continue
    print(f"\n===== {split.upper()} (argmax) =====")
    print(s[["model", "seeds", "f1", "f1_sd", "precision", "recall", "fpr",
             "implied_P_landscape", "fpr_gap_x"]]
          .sort_values("f1", ascending=False).round(4).to_string(index=False))
print("\nfpr_gap_x = how many times too high the false-positive rate is for "
      f"precision 0.5 at landscape prevalence p={LAND:g}")
print("\nwrote results/03_unet_pytorch/model_comparison.csv")
