"""Label-budget curves: F1 and deployability gap vs number of labelled tiles."""
import glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

rows = []
for f in sorted(glob.glob("results/04_label_budget_factorial/*/metrics_all_seeds.csv")):
    n = os.path.basename(os.path.dirname(f)); d = pd.read_csv(f)
    ip = [c for c in d.columns if c.startswith("implied")][0]
    rq = [c for c in d.columns if c.startswith("required")][0]
    g = d[(d.split == "test") & (d.rule == "argmax")]
    if g.empty: continue
    a, b = n.replace("fx_", "").rsplit("_n", 1)
    rows.append(dict(arch=a, n=int(b), f1=g.f1.mean(), sd=g.f1.std(),
                     gap=g.fpr.mean() / g[rq].mean()))
t = pd.DataFrame(rows)

LAB = {"scratch": "U-Net scratch", "efficientnetb0": "EfficientNet-B0 (ImageNet)",
       "resnet34": "ResNet-34 (ImageNet)", "openearthmap": "EfficientNet-B4 (OpenEarthMap)",
       "dinov3vits": "DINOv3 ViT-S (web)", "dinov3sat": "DINOv3 ViT-L (SAT-493M)",
       "dinov3convnext": "DINOv3 ConvNeXt (web)"}
ORD = ["scratch", "efficientnetb0", "resnet34", "openearthmap",
       "dinov3vits", "dinov3sat", "dinov3convnext"]
C = dict(zip(ORD, ["#444444", "#1f77b4", "#17becf", "#2ca02c",
                   "#ff7f0e", "#d62728", "#9467bd"]))

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for a in ORD:
    s = t[t.arch == a].sort_values("n")
    if s.empty: continue
    ls = "--" if a == "scratch" else "-"
    ax[0].errorbar(s.n, s.f1, yerr=s.sd, marker="o", ms=5, lw=2, ls=ls,
                   color=C[a], label=LAB[a], capsize=3)
    ax[1].plot(s.n, s.gap, marker="o", ms=5, lw=2, ls=ls, color=C[a], label=LAB[a])

sc = t[t.arch == "scratch"].sort_values("n")
if not sc.empty:
    ax[0].axhline(sc.f1.iloc[-1], color="#444444", ls=":", lw=1.2)
    ax[0].text(70, sc.f1.iloc[-1] + 0.008,
               "scratch with ALL 639 tiles", fontsize=8, color="#444444")

ax[0].set(xscale="log", xlabel="Labelled training tiles", ylabel="F1 (test, argmax)",
          title="Accuracy: foundation models need ~10x fewer labels")
ax[1].set(xscale="log", xlabel="Labelled training tiles",
          ylabel="FPR / FPR required for precision 0.5",
          title="Deployability: no model closes the gap")
ax[1].axhline(1.0, color="crimson", ls=":", lw=1.5)
ax[1].text(70, 1.6, "deployable (gap = 1)", fontsize=8, color="crimson")
for a_ in ax:
    a_.grid(True, ls="--", alpha=0.4)
    a_.set_xticks([64, 128, 320, 639]); a_.set_xticklabels([64, 128, 320, 639])
ax[0].legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig("results/04_label_budget_factorial/label_budget_curves.png", dpi=200)
t.to_csv("results/04_label_budget_factorial/label_budget_curve.csv", index=False)
print("wrote results/04_label_budget_factorial/label_budget_curves.png")
print(f"\nscratch @639 F1 = {sc.f1.iloc[-1]:.4f}")
for a in ["dinov3vits", "dinov3sat", "dinov3convnext"]:
    s = t[(t.arch == a) & (t.n == 64)]
    if not s.empty:
        print(f"{a:16s} @64  F1 = {s.f1.iloc[0]:.4f}")
