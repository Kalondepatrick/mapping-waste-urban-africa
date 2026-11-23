"""
U-Net waste segmentation: training + prevalence-aware evaluation (v2).

What changed vs 01_train_unet.py
--------------------------------
1. Reproducible: every run is seeded; the train/val split is saved to CSV.
2. Spatial split: internal train/val split is by spatial block (tile centroid
   on a grid), not by random tile, so neighbouring tiles of one pile cannot
   sit on both sides of the split.
3. Clean roles for each dataset:
     - train      -> fitting
     - val        -> early stopping, LR schedule, checkpoint, threshold choice
     - test       -> the independent 222-tile partition; never used for any
                     decision, only for reporting
4. One evaluation function used everywhere, for BOTH decision rules:
     - "argmax"   -> same rule as the wide-area map
     - "thresh"   -> waste if p(waste) >= t*, with t* chosen on val only
5. Prevalence-aware outputs: pooled TP/FP/FN/TN, prevalence, false-positive
   rate, specificity, and the precision IMPLIED at a landscape prevalence you
   supply, plus the false-positive rate REQUIRED to reach a target precision.
6. Per-tile confusion counts are saved, so prevalence re-sampling experiments
   can be done later without re-running the model.
7. Multiple seeds in one call, with a mean/SD summary across seeds.
8. Optional extra training-only folders (for hard negatives later).
9. Architecture is UNCHANGED (16 filters, 3 levels), so differences in results
   come from the evaluation protocol, not the model.

Example (Windows):
    python 01_train_unet_v2.py ^
        --dev-dir  "C:\\...\\partitioned\\modelDevelopment" ^
        --test-dir "C:\\...\\partitioned\\validation" ^
        --seeds 1 2 3 --landscape-prevalence 0.00045
"""

import argparse
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import rotate
from sklearn.metrics import average_precision_score, precision_recall_curve

import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence, to_categorical


# =============================================================================
# Arguments
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dev-dir", default=(
        r"C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa\data"
        r"\tiled_data\partitioned\modelDevelopment"),
        help="Model-development folder containing images/ and masks/.")
    p.add_argument("--test-dir", default=None,
                   help="Independent test folder (images/, masks/). "
                        "Never used for model or threshold selection.")
    p.add_argument("--extra-train-dirs", nargs="*", default=[],
                   help="Extra folders added to TRAINING only (e.g. hard negatives).")
    p.add_argument("--results-dir", default="./results_v2")
    p.add_argument("--exp-name", default="unet_f16")

    # Match results/class_label_mapping.csv from the original training run:
    # raw labels 1..9 map to channels 0..8; raw label 99 maps to channel 9.
    p.add_argument("--labels", nargs="+", type=int, default=[*range(1, 10), 99],
                   help="Raw raster labels, fixed so N_CLASSES never changes "
                        "between runs (default 1..9 and 99).")
    p.add_argument("--waste-label", type=int, default=1)

    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--val-block-size", type=float, default=250.0,
                   help="Grid size in map units (m) for the spatial train/val "
                        "split. Smaller than the 500 m test blocks so the "
                        "development area has enough blocks to split.")
    p.add_argument("--deterministic", action="store_true",
                   help="Enable TF op determinism (slower, fully repeatable).")

    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--init-filters", type=int, default=16)
    p.add_argument("--small-rotation", action="store_true",
                   help="Also apply random +-15 degree rotation (original "
                        "augmentation). Flips and 90-degree rotations are "
                        "always applied.")

    p.add_argument("--landscape-prevalence", type=float, nargs="*",
                   default=[0.00045],
                   help="Waste pixel prevalence(s) at which to report implied "
                        "precision (e.g. whole-settlement prevalence).")
    p.add_argument("--target-precision", type=float, default=0.5,
                   help="Precision for the 'required false-positive rate' output.")
    return p.parse_args()


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


# =============================================================================
# Data loading (cached in memory as uint8 / int)
# =============================================================================
def list_pairs(folder):
    img_dir = os.path.join(folder, "images")
    msk_dir = os.path.join(folder, "masks")

    def tifs(d):
        return {f for f in os.listdir(d) if f.lower().endswith(".tif")}

    imgs, msks = tifs(img_dir), tifs(msk_dir)
    common = sorted(imgs & msks)
    if not common:
        raise FileNotFoundError(f"No matching image/mask .tif names in {folder}")
    if imgs - msks:
        print(f"  Warning: {len(imgs - msks)} images without masks in {folder}")
    if msks - imgs:
        print(f"  Warning: {len(msks - imgs)} masks without images in {folder}")
    return img_dir, msk_dir, common


def load_folder(folder, label_map, img_size, tag):
    """Load all tiles of a folder. Returns dict with arrays and metadata."""
    img_dir, msk_dir, names = list_pairs(folder)
    images, masks, cx, cy = [], [], [], []
    warned_dtype = False
    warned_crs = False

    lut = np.full(max(label_map) + 1, -1, dtype=np.int16)
    for raw, idx in label_map.items():
        lut[raw] = idx

    for name in names:
        with rasterio.open(os.path.join(img_dir, name)) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
            if img.dtype != np.uint8 and not warned_dtype:
                print(f"  Warning: imagery dtype is {img.dtype}, not uint8. "
                      f"Scaling assumes 0-255; check radiometry.")
                warned_dtype = True
            if src.crs is not None and not src.crs.is_projected and not warned_crs:
                print("  Warning: CRS is not projected; spatial blocks will be "
                      "in degrees. Set --val-block-size accordingly.")
                warned_crs = True
            t = src.transform
            cx.append(t.c + t.a * src.width / 2.0)
            cy.append(t.f + t.e * src.height / 2.0)

        with rasterio.open(os.path.join(msk_dir, name)) as src:
            m = src.read(1)

        if img.shape[:2] != (img_size, img_size) or m.shape != (img_size, img_size):
            raise ValueError(f"{name}: expected {img_size}x{img_size}, got "
                             f"image {img.shape[:2]}, mask {m.shape}")

        if m.min() < 0 or m.max() >= len(lut) or np.any(lut[m] < 0):
            bad = sorted(set(np.unique(m).tolist()) - set(label_map))
            raise ValueError(f"{name}: mask has labels not in --labels: {bad}")

        images.append(np.clip(img, 0, 255).astype(np.uint8))
        masks.append(lut[m].astype(np.int16))

    print(f"  Loaded {len(names)} tiles from {tag}: {folder}")
    return {
        "names": np.array(names),
        "images": np.stack(images),
        "masks": np.stack(masks),
        "cx": np.array(cx),
        "cy": np.array(cy),
        "source": np.array([tag] * len(names)),
    }


def subset(data, idx):
    return {k: v[idx] for k, v in data.items()}


def concat(*datasets):
    return {k: np.concatenate([d[k] for d in datasets]) for k in datasets[0]}


# =============================================================================
# Spatial train/val split
# =============================================================================
def spatial_split(data, block_size, val_frac, seed, waste_idx):
    bx = np.floor(data["cx"] / block_size).astype(np.int64)
    by = np.floor(data["cy"] / block_size).astype(np.int64)
    groups = np.array([f"{a}_{b}" for a, b in zip(bx, by)])
    unique = np.array(sorted(set(groups)))

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    n_target = val_frac * len(groups)
    val_groups, n_val = [], 0
    for g in unique:
        if n_val >= n_target:
            break
        val_groups.append(g)
        n_val += np.sum(groups == g)

    is_val = np.isin(groups, val_groups)
    if is_val.all() or (~is_val).all():
        raise ValueError("Spatial split produced an empty set; "
                         "reduce --val-block-size.")
    if not np.any(data["masks"][is_val] == waste_idx):
        raise ValueError("Validation blocks contain no waste; try another "
                         "seed or a smaller --val-block-size.")

    print(f"  Spatial split: {len(unique)} blocks, "
          f"{len(val_groups)} to val ({is_val.sum()} tiles), "
          f"{(~is_val).sum()} tiles to train")
    return np.where(~is_val)[0], np.where(is_val)[0], groups


# =============================================================================
# Generator
# =============================================================================
class TileGenerator(Sequence):
    def __init__(self, images, masks, n_classes, batch_size, augment,
                 small_rotation, seed, shuffle, **kwargs):
        super().__init__(**kwargs)
        self.images, self.masks = images, masks
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.augment = augment
        self.small_rotation = small_rotation
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(images))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _augment(self, img, msk):
        if self.rng.random() < 0.5:
            img, msk = img[::-1], msk[::-1]
        if self.rng.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        k = int(self.rng.integers(0, 4))
        img, msk = np.rot90(img, k), np.rot90(msk, k)
        if self.small_rotation and self.rng.random() < 0.5:
            angle = self.rng.uniform(-15, 15)
            img = rotate(img, angle, reshape=False, order=1, mode="reflect")
            msk = rotate(msk, angle, reshape=False, order=0, mode="nearest")
        return img, msk

    def __getitem__(self, i):
        idx = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
        xs, ys = [], []
        for j in idx:
            img = self.images[j].astype(np.float32)
            msk = self.masks[j]
            if self.augment:
                img, msk = self._augment(img, msk)
            xs.append(img / 255.0)
            ys.append(to_categorical(np.rint(msk).astype(np.int32),
                                     num_classes=self.n_classes))
        return np.array(xs), np.array(ys)


# =============================================================================
# Model (unchanged from v1)
# =============================================================================
def build_model(n_classes, img_size, init_filters=16):
    inputs = Input((img_size, img_size, 3))

    def conv_block(x, f):
        x = Conv2D(f, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(f, 3, padding="same", activation="relu")(x)
        return BatchNormalization()(x)

    s1 = conv_block(inputs, init_filters)
    s2 = conv_block(MaxPooling2D()(s1), init_filters * 2)
    s3 = conv_block(MaxPooling2D()(s2), init_filters * 4)
    b1 = conv_block(MaxPooling2D()(s3), init_filters * 8)

    d1 = conv_block(Concatenate()([
        Conv2DTranspose(init_filters * 4, 2, strides=2, padding="same")(b1), s3]),
        init_filters * 4)
    d2 = conv_block(Concatenate()([
        Conv2DTranspose(init_filters * 2, 2, strides=2, padding="same")(d1), s2]),
        init_filters * 2)
    d3 = conv_block(Concatenate()([
        Conv2DTranspose(init_filters, 2, strides=2, padding="same")(d2), s1]),
        init_filters)

    outputs = Conv2D(n_classes, 1, activation="softmax")(d3)
    return Model(inputs, outputs, name=f"UNet_Filters_{init_filters}")


# =============================================================================
# Evaluation
# =============================================================================
def predict_waste(model, images, waste_idx, batch_size):
    """Return (argmax==waste boolean, p(waste) float16), shape (N, H, W)."""
    is_waste, p_waste = [], []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].astype(np.float32) / 255.0
        probs = model.predict(x, verbose=0)
        is_waste.append(np.argmax(probs, axis=-1) == waste_idx)
        p_waste.append(probs[..., waste_idx].astype(np.float16))
    return np.concatenate(is_waste), np.concatenate(p_waste)


def per_tile_counts(truth, pred, names, groups=None):
    """Pixel confusion counts per tile for the waste class."""
    t = truth.reshape(len(truth), -1)
    p = pred.reshape(len(pred), -1)
    df = pd.DataFrame({
        "tile": names,
        "tp": np.sum(t & p, axis=1),
        "fp": np.sum(~t & p, axis=1),
        "fn": np.sum(t & ~p, axis=1),
        "tn": np.sum(~t & ~p, axis=1),
    })
    if groups is not None:
        df.insert(1, "block", groups)
    df["n_waste"] = df["tp"] + df["fn"]
    return df


def pooled_metrics(counts, landscape_prev, target_precision):
    tp, fp, fn, tn = (float(counts[c].sum()) for c in ("tp", "fp", "fn", "tn"))
    n = tp + fp + fn + tn
    eps = 1e-12
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    fpr = fp / (fp + tn + eps)
    out = {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "prevalence": (tp + fn) / n,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec + eps),
        "iou": tp / (tp + fp + fn + eps),
        "fpr": fpr,
        "specificity": 1 - fpr,
    }
    # Precision implied at another prevalence, holding recall and FPR fixed
    # (Bayes' rule): PPV = rec*p / (rec*p + fpr*(1-p))
    for p in landscape_prev:
        out[f"implied_precision_at_p={p:g}"] = (
            rec * p / (rec * p + fpr * (1 - p) + eps))
        # FPR needed to reach target precision at that prevalence
        P = target_precision
        out[f"required_fpr_for_P={P:g}_at_p={p:g}"] = (
            rec * p * (1 - P) / (P * (1 - p) + eps))
    return out


def plot_pr(y_true, y_prob, dev_prev, landscape_prev, path, title):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, lw=2, label=f"Waste PR (AP = {ap:.3f})")
    ax.axhline(dev_prev, ls="--", lw=1.2,
               label=f"No-skill, this set (p = {dev_prev:.4f})")
    for p in landscape_prev:
        ax.axhline(p, ls=":", lw=1.2, color="grey",
                   label=f"No-skill, landscape (p = {p:g})")
    ax.set(xlabel="Recall", ylabel="Precision", title=title,
           xlim=(0, 1), ylim=(0, 1))
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return ap


def choose_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[np.argmax(f1)])


# =============================================================================
# One run (one seed)
# =============================================================================
def run_seed(seed, args, dev, test, extra, label_map, waste_idx, n_classes):
    out_dir = os.path.join(args.results_dir, args.exp_name, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n==================== seed {seed} ====================")
    set_seed(seed, args.deterministic)
    tf.keras.backend.clear_session()

    tr_idx, va_idx, groups = spatial_split(
        dev, args.val_block_size, args.val_frac, seed, waste_idx)
    train = subset(dev, tr_idx)
    val = subset(dev, va_idx)
    if extra is not None:
        train = concat(train, extra)

    pd.DataFrame({
        "tile": np.concatenate([dev["names"][tr_idx], dev["names"][va_idx]]),
        "block": np.concatenate([groups[tr_idx], groups[va_idx]]),
        "split": ["train"] * len(tr_idx) + ["val"] * len(va_idx),
    }).to_csv(os.path.join(out_dir, "split.csv"), index=False)

    train_gen = TileGenerator(train["images"], train["masks"], n_classes,
                              args.batch_size, augment=True,
                              small_rotation=args.small_rotation,
                              seed=seed, shuffle=True)
    val_gen = TileGenerator(val["images"], val["masks"], n_classes,
                            args.batch_size, augment=False,
                            small_rotation=False, seed=seed, shuffle=False)

    model = build_model(n_classes, args.img_size, args.init_filters)
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])

    best_path = os.path.join(out_dir, "best.keras")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10,
                              min_lr=1e-6, verbose=1),
            ModelCheckpoint(best_path, monitor="val_loss",
                            save_best_only=True),
            CSVLogger(os.path.join(out_dir, "history.csv")),
        ],
        verbose=2,
    )

    # ---- Threshold chosen on VAL only ----
    va_arg, va_prob = predict_waste(model, val["images"], waste_idx,
                                    args.batch_size)
    va_truth = val["masks"] == waste_idx
    t_star = choose_threshold(va_truth.ravel(),
                              va_prob.ravel().astype(np.float32))
    print(f"  Threshold chosen on val: t* = {t_star:.4f}")

    sets = {"val": (val, va_arg, va_prob, groups[va_idx])}
    if test is not None:
        te_arg, te_prob = predict_waste(model, test["images"], waste_idx,
                                        args.batch_size)
        sets["test"] = (test, te_arg, te_prob, None)

    rows = []
    for split_name, (data, pred_arg, prob, grp) in sets.items():
        truth = data["masks"] == waste_idx
        prev = float(truth.mean())
        ap = plot_pr(truth.ravel(), prob.ravel().astype(np.float32), prev,
                     args.landscape_prevalence,
                     os.path.join(out_dir, f"pr_curve_{split_name}.png"),
                     f"Waste PR curve ({split_name}, seed {seed})")

        for rule, pred in (("argmax", pred_arg),
                           ("thresh", prob.astype(np.float32) >= t_star)):
            counts = per_tile_counts(truth, pred, data["names"], grp)
            counts.to_csv(os.path.join(
                out_dir, f"per_tile_counts_{split_name}_{rule}.csv"),
                index=False)
            m = pooled_metrics(counts, args.landscape_prevalence,
                               args.target_precision)
            rows.append({"seed": seed, "split": split_name, "rule": rule,
                         "threshold": t_star if rule == "thresh" else None,
                         "average_precision": ap, **m})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print(metrics[["split", "rule", "prevalence", "precision", "recall",
                   "f1", "iou", "fpr"]].to_string(index=False))
    return metrics


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    labels = sorted(args.labels)
    if args.waste_label not in labels:
        raise ValueError("--waste-label must be in --labels")
    label_map = {raw: i for i, raw in enumerate(labels)}
    waste_idx = label_map[args.waste_label]
    n_classes = len(labels)

    exp_dir = os.path.join(args.results_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "label_map": label_map,
                   "waste_model_channel": waste_idx,
                   "tensorflow": tf.__version__}, f, indent=2)

    print(f"Labels {labels} -> channels 0..{n_classes - 1}; "
          f"waste label {args.waste_label} -> channel {waste_idx}")

    dev = load_folder(args.dev_dir, label_map, args.img_size, "dev")
    test = (load_folder(args.test_dir, label_map, args.img_size, "test")
            if args.test_dir else None)
    if test is None:
        print("  Warning: no --test-dir given; only val metrics will be "
              "reported, and they are NOT independent (val drives early "
              "stopping and threshold choice).")
    extra = None
    if args.extra_train_dirs:
        extra = concat(*[load_folder(d, label_map, args.img_size, "extra")
                         for d in args.extra_train_dirs])

    all_metrics = pd.concat([
        run_seed(s, args, dev, test, extra, label_map, waste_idx, n_classes)
        for s in args.seeds
    ])
    all_metrics.to_csv(os.path.join(exp_dir, "metrics_all_seeds.csv"),
                       index=False)

    num_cols = [c for c in all_metrics.columns
                if c not in ("seed", "split", "rule", "threshold")
                and np.issubdtype(all_metrics[c].dtype, np.number)]
    summary = (all_metrics.groupby(["split", "rule"])[num_cols]
               .agg(["mean", "std"]))
    summary.to_csv(os.path.join(exp_dir, "metrics_summary.csv"))
    print("\n=========== Summary across seeds (mean) ===========")
    print(summary.xs("mean", axis=1, level=1)[
        ["prevalence", "precision", "recall", "f1", "iou", "fpr"]
    ].to_string())


if __name__ == "__main__":
    main()
