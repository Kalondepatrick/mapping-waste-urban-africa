"""Landscape-prevalence operating frontier: what you can actually buy."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

t = pd.read_csv("results/05_negative_weight_sweep/frontier.csv").sort_values("neg_weight")
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t.rec, t.implied, "o-", lw=2, ms=7, color="#1f77b4")
for _, r in t.iterrows():
    ax.annotate(f"w={r.neg_weight:g}", (r.rec, r.implied),
                textcoords="offset points", xytext=(8, 6), fontsize=8)

ax.axhline(0.5, color="crimson", ls="--", lw=1.5)
ax.text(0.02, 0.52, "precision 0.5 — a usable map", color="crimson", fontsize=9)
ax.axvspan(0.35, 1.0, color="green", alpha=0.05)
ax.text(0.45, 0.30, "useful coverage\n(recall > 0.35)", fontsize=9,
        color="green", ha="center")

ax.set(xlabel="Recall (share of real waste pixels found)",
       ylabel="Precision at landscape prevalence (p = 0.00045)",
       title="No operating point is both accurate and useful",
       xlim=(-0.03, 0.78), ylim=(0, 0.6))
ax.grid(True, ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig("results/05_negative_weight_sweep/frontier.png", dpi=200)
print("wrote results/05_negative_weight_sweep/frontier.png")
best = t[t.rec > 0.35]
print(f"\nbest implied precision at recall > 0.35: {best.implied.max():.4f}")
print(f"best implied precision at ANY recall    : {t.implied.max():.4f} "
      f"(at recall {t.loc[t.implied.idxmax(),'rec']:.4f})")
