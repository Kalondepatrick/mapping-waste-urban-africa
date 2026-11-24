"""Tabulate which land-cover classes are predicted as waste.

The reference masks carry nine classes, so every false-positive pixel has a
true class. Two quantities are reported because they rank the classes
differently:

    commission rate   fraction of a class predicted waste; independent of how
                      much of the class the scene contains, so it transfers.
    share of errors   fraction of all false positives belonging to a class;
                      depends on class area, so it sets verification effort.

Inference runs once per checkpoint on the unperturbed held-out imagery, under
the same argmax rule and the same 216 clean tiles as every other table.

    python 06_evaluation/10_confusion_by_landcover.py [--configs ...]

Writes results/10_landcover_confusion/landcover_confusion.csv
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


def repo_root() -> str:
    p = os.path.abspath(os.getcwd())
    while not os.path.exists(os.path.join(p, "CITATION.cff")) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    if not os.path.exists(os.path.join(p, "CITATION.cff")):
        raise SystemExit("Run this from inside the repository.")
    return p


def _env(name, default):
    return os.environ.get(name) or default


ROOT = _env("WASTE_ROOT", None) or repo_root()
os.chdir(ROOT)
TRAINER_DIR = _env("WASTE_TRAINER_DIR", os.path.join(ROOT, "03_unet_pytorch"))
RESULTS = _env("WASTE_RESULTS_DIR", os.path.join(ROOT, "results"))
CKPT_ROOT = _env("WASTE_CKPT_ROOT", os.path.join(RESULTS, "03_unet_pytorch"))
OUT_DIR = _env("WASTE_OUT_DIR", os.path.join(RESULTS, "10_landcover_confusion"))
TEST_DIR = _env("WASTE_TEST_DIR",
                os.path.join(ROOT, "data", "tiled_data", "partitioned", "Validation"))
CLEAN = _env("WASTE_CLEAN_LIST",
             os.path.join(RESULTS, "03_unet_pytorch", "test_tiles_clean.csv"))

sys.path.insert(0, TRAINER_DIR)
_spec = importlib.util.spec_from_file_location(
    "trainer", os.path.join(TRAINER_DIR, "01_train_unet_v5_torch.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

# Legend from data/tiled_data/finetuning_hard_negatives/class_lookup.csv
CLASS_NAMES = {
    1: "Waste",
    2: "Soil",
    3: "Vegetation",
    4: "Built-up surface",
    5: "Built-up surface (paved)",
    6: "Water body",
    7: "Natural surface, rocks",
    8: "Non-persistent object",
    9: "Ancillary class",
}
N_CLASSES = 9


def load_test_tiles():
    keep = set(pd.read_csv(CLEAN)["tile"].astype(str))
    keep |= {os.path.splitext(k)[0] for k in keep}
    img_dir = os.path.join(TEST_DIR, "images")
    msk_dir = os.path.join(TEST_DIR, "masks")
    names = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith(".tif")
                   and (f in keep or os.path.splitext(f)[0] in keep))
    if not names:
        raise SystemExit(f"No clean test tiles under {img_dir}")
    images, raw = [], []
    for n in names:
        with rasterio.open(os.path.join(img_dir, n)) as src:
            images.append(src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8))
        with rasterio.open(os.path.join(msk_dir, n)) as src:
            raw.append(src.read(1).astype(np.int16))
    print(f"  {len(names)} clean held-out tiles")
    return names, np.stack(images), np.stack(raw)


def checkpoints(configs, seeds):
    found = []
    for cfg in sorted(os.listdir(CKPT_ROOT)):
        d = os.path.join(CKPT_ROOT, cfg)
        cfg_json = os.path.join(d, "config.json")
        if not os.path.isdir(d) or not os.path.exists(cfg_json):
            continue
        if configs and cfg not in configs:
            continue
        meta = json.load(open(cfg_json, encoding="utf-8"))
        arch = meta.get("arch") or meta.get("backbone")
        oem = meta.get("oem_weights")
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
                found.append((cfg, arch, oem, k, ck))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "landcover_confusion.csv"))
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    names, images, raw = load_test_tiles()
    # 99 is a no-data fill, excluded here exactly as it is from every metric.
    valid = (raw != 99)
    for lab in np.unique(raw[valid]):
        if lab not in CLASS_NAMES:
            raise SystemExit(f"Mask carries unexpected label {lab}")

    area = {lab: int(((raw == lab) & valid).sum()) for lab in CLASS_NAMES}
    print("  class areas (pixels):")
    for lab, n in area.items():
        print(f"    {lab} {CLASS_NAMES[lab]:<26} {n:>12,}  {100*n/valid.sum():6.3f}%")

    cks = checkpoints(a.configs, a.seeds)
    if not cks:
        raise SystemExit("No checkpoints matched.")
    print(f"{len(cks)} checkpoints")

    rows, t0 = [], time.time()
    for i, (cfg, arch, oem, seed, ck) in enumerate(cks, 1):
        try:
            model, _ = T.build(arch, N_CLASSES, 256, pretrained=True,
                               freeze_backbone=False, oem_weights=oem)
            model.load_state_dict(torch.load(ck, map_location="cpu",
                                             weights_only=True), strict=True)
            model = model.to(device).eval()
        except Exception as e:                        # noqa: BLE001
            print(f"  SKIP {cfg} seed {seed}: {type(e).__name__}: {e}")
            continue

        pred, _ = T.predict_waste(model, images, 0, a.batch_size, device, amp=False)
        del model

        pred_waste = pred & valid
        fp_total = int((pred_waste & (raw != 1)).sum())
        for lab, nm in CLASS_NAMES.items():
            in_class = (raw == lab) & valid
            n_cls = int(in_class.sum())
            n_hit = int((in_class & pred_waste).sum())
            rows.append({
                "config": cfg, "arch": arch, "seed": seed,
                "raw_label": lab, "class_name": nm,
                "class_pixels": n_cls,
                "predicted_waste": n_hit,
                # for the waste row this is recall; for the rest, commission
                "rate": n_hit / n_cls if n_cls else float("nan"),
                "share_of_false_positives":
                    (n_hit / fp_total if (fp_total and lab != 1) else
                     (float("nan") if lab == 1 else 0.0)),
            })
        el = time.time() - t0
        print(f"  [{i:3d}/{len(cks)}] {cfg} s{seed}  FP {fp_total:,}  "
              f"({el/i:.1f} s/model)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(df)} rows, {time.time()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
