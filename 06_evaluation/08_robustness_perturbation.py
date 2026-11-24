"""Re-evaluate trained checkpoints on perturbed held-out imagery.

Every reported metric is measured on tiles cut from the same orthomosaic as the
training tiles, so the spatial split controls for location but holds radiometry
and scale fixed. This script varies both and re-scores the same checkpoints.

    resolution  area-average down by a factor, bilinear back to 256 x 256.
                Factor s corresponds to an effective GSD of 3.81 cm x s.
    brightness  multiply the 8-bit values by a gain and clip.

Masks are never perturbed, so results stay comparable with the headline tables.
Paths default to this repository and are overridable by WASTE_* environment
variables so the same file runs against the HPC checkout.

    python 06_evaluation/08_robustness_perturbation.py [--smoke] [--configs ...]

Writes results/09_robustness/robustness_metrics.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image


def repo_root() -> str:
    p = os.path.abspath(os.getcwd())
    while not os.path.exists(os.path.join(p, "CITATION.cff")) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    if not os.path.exists(os.path.join(p, "CITATION.cff")):
        raise SystemExit("Run this from inside the repository.")
    return p


def _env(name, default):
    return os.environ.get(name) or default


# Paths. The defaults are the layout of this repository; the environment
# variables let the same script run against the HPC checkout, whose directory
# names differ, without forking it.
ROOT = _env("WASTE_ROOT", None) or repo_root()
os.chdir(ROOT)
TRAINER_DIR = _env("WASTE_TRAINER_DIR", os.path.join(ROOT, "03_unet_pytorch"))
RESULTS = _env("WASTE_RESULTS_DIR", os.path.join(ROOT, "results"))
CKPT_ROOT = _env("WASTE_CKPT_ROOT", os.path.join(RESULTS, "03_unet_pytorch"))
OUT_DIR = _env("WASTE_OUT_DIR", os.path.join(RESULTS, "09_robustness"))
TEST_DIR = _env("WASTE_TEST_DIR",
                os.path.join(ROOT, "data", "tiled_data", "partitioned", "Validation"))
CLEAN = _env("WASTE_CLEAN_LIST",
             os.path.join(RESULTS, "03_unet_pytorch", "test_tiles_clean.csv"))

# The trainer module name starts with a digit, so it is loaded by path.
sys.path.insert(0, TRAINER_DIR)
_spec = importlib.util.spec_from_file_location(
    "trainer", os.path.join(TRAINER_DIR, "01_train_unet_v5_torch.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

GSD_CM = 3.81
P_LANDSCAPE = 0.00045
TARGET_P = 0.5

RESOLUTION_FACTORS = [1, 1.5, 2, 3, 4, 8]
BRIGHTNESS_GAINS = [0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50]


# ------------------------------------------------------------------ perturbations
def degrade_resolution(img: np.ndarray, factor: float) -> np.ndarray:
    """Area-average down to 1/factor, then bilinear back to the original grid."""
    if factor == 1:
        return img
    h, w = img.shape[:2]
    sw, sh = max(1, round(w / factor)), max(1, round(h / factor))
    small = Image.fromarray(img).resize((sw, sh), Image.BOX)
    return np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.uint8)


def change_brightness(img: np.ndarray, gain: float) -> np.ndarray:
    if gain == 1.0:
        return img
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def conditions(resolutions, gains):
    """(axis, level, label, transform). The unperturbed case appears once."""
    out = [("baseline", 1.0, "unperturbed", lambda a: a)]
    for f in resolutions:
        if f == 1:
            continue
        out.append(("resolution", float(f), f"{GSD_CM * f:.1f} cm GSD",
                    lambda a, f=f: degrade_resolution(a, f)))
    for g in gains:
        if g == 1.0:
            continue
        out.append(("brightness", g, f"gain {g:.2f}",
                    lambda a, g=g: change_brightness(a, g)))
    return out


# ------------------------------------------------------------------------- data
def load_test_tiles(img_size=256):
    keep = set(pd.read_csv(CLEAN)["tile"].astype(str))
    keep |= {os.path.splitext(k)[0] for k in keep}
    img_dir = os.path.join(TEST_DIR, "images")
    msk_dir = os.path.join(TEST_DIR, "masks")
    names = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith(".tif")
                   and (f in keep or os.path.splitext(f)[0] in keep))
    if not names:
        raise SystemExit(f"No clean test tiles found under {img_dir}")

    lut = np.full(100, -9, dtype=np.int16)
    for raw, idx in {r: i for i, r in enumerate(range(1, 10))}.items():
        lut[raw] = idx
    lut[99] = T.IGNORE

    images, masks = [], []
    for n in names:
        with rasterio.open(os.path.join(img_dir, n)) as src:
            images.append(src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8))
        with rasterio.open(os.path.join(msk_dir, n)) as src:
            masks.append(lut[src.read(1)].astype(np.int16))
    images = np.stack(images)
    masks = np.stack(masks)
    if (masks == -9).any():
        raise SystemExit("A mask carries a label outside 1-9 and 99.")
    print(f"  {len(names)} clean held-out tiles, {images.shape[1]}x{images.shape[2]} px")
    return names, images, masks


# ----------------------------------------------------------------------- models
def checkpoints(configs, seeds):
    base = CKPT_ROOT
    found = []
    for cfg in sorted(os.listdir(base)):
        d = os.path.join(base, cfg)
        if not os.path.isdir(d) or (configs and cfg not in configs):
            continue
        cfg_json = os.path.join(d, "config.json")
        if not os.path.exists(cfg_json):
            continue
        meta = json.load(open(cfg_json, encoding="utf-8"))
        arch = meta.get("arch") or meta.get("backbone")
        if not arch:
            continue
        for s in sorted(os.listdir(d)):
            if not s.startswith("seed_"):
                continue
            k = int(s.split("_")[1])
            if seeds and k not in seeds:
                continue
            ck = os.path.join(d, s, "best.pt")
            if os.path.exists(ck):
                found.append((cfg, arch, k, ck))
    return found


def load_model(arch, ckpt, device, img_size=256, n_classes=9):
    model, _ = T.build(arch, n_classes, img_size, pretrained=True,
                       freeze_backbone=False, oem_weights=None)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--smoke", action="store_true",
                    help="one checkpoint, one condition, for timing")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "robustness_metrics.csv"))
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    names, images, masks = load_test_tiles()
    valid = masks != T.IGNORE
    truth = masks == 0            # waste is raw label 1 -> index 0

    cks = checkpoints(a.configs, a.seeds)
    conds = conditions(RESOLUTION_FACTORS, BRIGHTNESS_GAINS)
    if a.smoke:
        cks, conds = cks[:1], conds[:2]
    if not cks:
        raise SystemExit("No checkpoints matched.")
    print(f"{len(cks)} checkpoints x {len(conds)} conditions "
          f"= {len(cks) * len(conds)} evaluation passes")

    # Perturbed copies are built once and reused for every checkpoint.
    print("building perturbed copies ...", flush=True)
    perturbed = {}
    for axis, level, label, fn in conds:
        perturbed[label] = np.stack([fn(im) for im in images])

    rows, done, t0 = [], 0, time.time()
    for cfg, arch, seed, ck in cks:
        try:
            model = load_model(arch, ck, device)
        except Exception as e:                       # noqa: BLE001
            print(f"  SKIP {cfg} seed {seed}: {type(e).__name__}: {e}")
            continue
        for axis, level, label, _ in conds:
            pred, _p = T.predict_waste(model, perturbed[label], 0,
                                       a.batch_size, device, amp=False)
            counts = T.per_tile_counts(truth, pred, np.array(names), valid=valid)
            m = T.pooled_metrics(counts, [P_LANDSCAPE], TARGET_P)
            req = m[f"required_fpr_for_P={TARGET_P:g}_at_p={P_LANDSCAPE:g}"]
            rows.append({
                "config": cfg, "arch": arch, "seed": seed,
                "axis": axis, "level": level, "condition": label,
                "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                "iou": m["iou"], "fpr": m["fpr"],
                "implied_precision": m[f"implied_precision_at_p={P_LANDSCAPE:g}"],
                "gap": m["fpr"] / req if req > 0 else float("inf"),
            })
            done += 1
            el = time.time() - t0
            print(f"  [{done:3d}/{len(cks) * len(conds)}] {cfg} s{seed} "
                  f"{label:<16} F1 {m['f1']:.4f}  FPR {m['fpr']:.5f} "
                  f"({el / done:.1f} s/pass)", flush=True)
        del model

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(df)} rows, {time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
