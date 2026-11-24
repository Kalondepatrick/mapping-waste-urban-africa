"""Qualitative figures, from the spatially partitioned checkpoints.

The manuscript's earlier qualitative figures were produced by Keras models
trained on the unpartitioned tile pool, which contains the validation tiles
they are shown predicting. These replacements use the same checkpoints as the
Results tables, on tiles never seen in training.

    Fig. 7   one row per confusable land-cover class; tiles chosen as those
             carrying the most false-positive pixels of that class
    Fig. 9   one row per ground sample distance, two models side by side

Outcomes are colour-separated (found / invented / missed) rather than shown as
a single mask.

    python 07_figures/12_qualitative_panels.py [--model ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.patches import Patch
from PIL import Image


def repo_root() -> str:
    p = os.path.abspath(os.getcwd())
    while not os.path.exists(os.path.join(p, "CITATION.cff")) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    if not os.path.exists(os.path.join(p, "CITATION.cff")):
        raise SystemExit("Run this from inside the repository.")
    return p


ROOT = repo_root()
os.chdir(ROOT)
TRAINER_DIR = os.path.join(ROOT, "03_unet_pytorch")
RESULTS = os.path.join(ROOT, "results")
CKPT_ROOT = os.path.join(RESULTS, "03_unet_pytorch")
FIG_DIR = os.path.join(RESULTS, "figures", "manuscript")
TEST_DIR = os.path.join(ROOT, "data", "tiled_data", "partitioned", "Validation")
CLEAN = os.path.join(RESULTS, "03_unet_pytorch", "test_tiles_clean.csv")

sys.path.insert(0, TRAINER_DIR)
_spec = importlib.util.spec_from_file_location(
    "trainer", os.path.join(TRAINER_DIR, "01_train_unet_v5_torch.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

GSD_CM = 3.81
CLASS_NAMES = {1: "Waste", 2: "Soil", 3: "Vegetation", 4: "Built-up surface",
               5: "Built-up surface (paved)", 6: "Water body",
               7: "Natural surface, rocks", 8: "Non-persistent object",
               9: "Ancillary class"}
# Ordered as Table VIII orders them: by commission rate.
PANEL_CLASSES = [7, 6, 2, 3]

TP_COL = np.array([0.15, 0.62, 0.25])    # green  - found
FP_COL = np.array([0.85, 0.20, 0.15])    # red    - invented
FN_COL = np.array([0.20, 0.42, 0.85])    # blue   - missed
REF_COL = np.array([0.95, 0.75, 0.10])   # gold   - reference, so that it
                                         # is never confused with an error

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7, "axes.titlesize": 7, "figure.dpi": 600,
})


def load_tiles():
    keep = set(pd.read_csv(CLEAN)["tile"].astype(str))
    keep |= {os.path.splitext(k)[0] for k in keep}
    img_dir, msk_dir = (os.path.join(TEST_DIR, d) for d in ("images", "masks"))
    names = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith(".tif")
                   and (f in keep or os.path.splitext(f)[0] in keep))
    imgs, raw = [], []
    for n in names:
        with rasterio.open(os.path.join(img_dir, n)) as s:
            imgs.append(s.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8))
        with rasterio.open(os.path.join(msk_dir, n)) as s:
            raw.append(s.read(1).astype(np.int16))
    return names, np.stack(imgs), np.stack(raw)


def load_model(cfg, seed, device):
    meta = json.load(open(os.path.join(CKPT_ROOT, cfg, "config.json"),
                          encoding="utf-8"))
    arch = meta.get("arch") or meta.get("backbone")
    model, _ = T.build(arch, 9, 256, pretrained=True, freeze_backbone=False,
                       oem_weights=meta.get("oem_weights"))
    ck = os.path.join(CKPT_ROOT, cfg, f"seed_{seed}", "best.pt")
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True),
                          strict=True)
    return model.to(device).eval(), arch


def reference(img, truth, alpha=0.55):
    """Greyscale base with the reference waste polygons in gold."""
    grey = img.astype(np.float32).mean(2) / 255.0
    out = np.repeat(grey[:, :, None], 3, axis=2) * 0.85 + 0.08
    out[truth] = (1 - alpha) * out[truth] + alpha * REF_COL
    return np.clip(out, 0, 1)


def overlay(img, truth, pred, valid, alpha=0.55):
    """Greyscale base with true/false positives and false negatives coloured."""
    grey = img.astype(np.float32).mean(2) / 255.0
    out = np.repeat(grey[:, :, None], 3, axis=2) * 0.85 + 0.08
    for mask, col in ((truth & pred & valid, TP_COL),
                      (~truth & pred & valid, FP_COL),
                      (truth & ~pred & valid, FN_COL)):
        out[mask] = (1 - alpha) * out[mask] + alpha * col
    return np.clip(out, 0, 1)


def degrade(img, factor):
    if factor == 1:
        return img
    h, w = img.shape[:2]
    small = Image.fromarray(img).resize((max(1, round(w / factor)),
                                         max(1, round(h / factor))), Image.BOX)
    return np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.uint8)


def legend(fig, y=0.055, with_ref=False):
    handles = [Patch(facecolor=REF_COL, label="reference waste")] if with_ref else []
    fig.legend(handles=handles + [Patch(facecolor=TP_COL, label="waste found"),
                        Patch(facecolor=FP_COL, label="predicted, not waste"),
                        Patch(facecolor=FN_COL, label="waste missed")],
               loc="lower center", ncol=4 if with_ref else 3, frameon=False,
               bbox_to_anchor=(0.5, y), fontsize=6.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="torch_scratch")
    ap.add_argument("--compare", default="torch_dinov2",
                    help="second model for the degradation panel")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, imgs, raw = load_tiles()
    valid = raw != 99
    truth = raw == 1
    print(f"{len(names)} clean held-out tiles; device {device}")

    model, arch = load_model(a.model, a.seed, device)
    pred, _ = T.predict_waste(model, imgs, 0, 8, device, amp=False)

    # ---------------------------------------------------------------- panel 1
    fp = (~truth) & pred & valid
    rows = []
    used = set()
    for lab in PANEL_CLASSES:
        counts = ((raw == lab) & fp).reshape(len(names), -1).sum(1)
        order = np.argsort(-counts)
        pick = next((i for i in order if i not in used and counts[i] > 0), None)
        if pick is None:
            continue
        used.add(pick)
        rows.append((lab, int(pick), int(counts[pick])))

    fig, axes = plt.subplots(len(rows), 3, figsize=(3.5, 1.24 * len(rows) + 0.5))
    axes = np.atleast_2d(axes)
    for r, (lab, i, n_fp) in enumerate(rows):
        axes[r, 0].imshow(imgs[i])
        axes[r, 1].imshow(reference(imgs[i], truth[i]))
        axes[r, 2].imshow(overlay(imgs[i], truth[i], pred[i], valid[i]))
        axes[r, 0].set_ylabel(f"({chr(97+r)}) {CLASS_NAMES[lab]}", fontsize=6.3)
        if r == 0:
            for c, t in enumerate(("image", "reference", "prediction")):
                axes[r, c].set_title(t, fontsize=7)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    legend(fig, y=0.012, with_ref=True)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.075,
                        wspace=0.04, hspace=0.04)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig7_qualitative_landcover.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig7_qualitative_landcover")

    # ---------------------------------------------------------------- panel 2
    waste_px = truth.reshape(len(names), -1).sum(1)
    tile = int(np.argmax(waste_px))
    factors = [1, 2, 4]
    model2, _ = load_model(a.compare, a.seed, device)
    fig, axes = plt.subplots(len(factors), 3, figsize=(3.5, 1.24 * len(factors) + 0.5))
    for r, f in enumerate(factors):
        dim = degrade(imgs[tile], f)
        p1, _ = T.predict_waste(model, dim[None], 0, 1, device, amp=False)
        p2, _ = T.predict_waste(model2, dim[None], 0, 1, device, amp=False)
        axes[r, 0].imshow(dim)
        axes[r, 1].imshow(overlay(dim, truth[tile], p1[0], valid[tile]))
        axes[r, 2].imshow(overlay(dim, truth[tile], p2[0], valid[tile]))
        axes[r, 0].set_ylabel(f"({chr(97+r)}) {GSD_CM*f:.1f} cm", fontsize=6.3)
        if r == 0:
            for c, t in enumerate(("image", "from scratch", "DINOv2 encoder")):
                axes[r, c].set_title(t, fontsize=7)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    legend(fig, y=0.012)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.075,
                        wspace=0.04, hspace=0.04)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig9_qualitative_resolution.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig9_qualitative_resolution")
    print(f"\npanel 1 tiles: " +
          ", ".join(f"{CLASS_NAMES[l]}={names[i]} ({n:,} FP px)" for l, i, n in rows))
    print(f"panel 2 tile : {names[tile]} ({waste_px[tile]:,} waste px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
