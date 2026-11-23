"""
U-Net waste segmentation on GPU: pretrained encoders + foundation backbone.

Why this exists
---------------
TensorFlow has no native Windows GPU support after 2.10, so 01_train_unet_v4.py
runs on CPU only (~27 s/epoch for the 16-filter scratch U-Net, and roughly
10-20x that for a pretrained encoder, which is not practical). PyTorch does
support CUDA on Windows, so every model in the comparison is trained here, on
the same data, the same spatial splits, the same decision rules and the same
prevalence-aware metrics as v4.

The scratch architecture is ported exactly from v4, so `--arch scratch` is a
cross-framework control: it should reproduce the v4 numbers. If it does, every
other row in the table is comparable to the published baseline.

Models (--arch)
---------------
  scratch          16-filter, 3-level U-Net. Identical to v4 / the paper.
  resnet34         U-Net, ImageNet-pretrained ResNet-34 encoder (smp).
  efficientnet-b0  U-Net, ImageNet-pretrained EfficientNet-B0 encoder (smp).
  dinov2           Frozen DINOv2 ViT-S/14 bottleneck + trainable conv stem and
                   U-Net decoder. The "would a better representation fix the
                   false-positive rate?" test.

Hard negatives
--------------
Two kinds, because the two folders on disk use different conventions:

  --extra-train-dirs DIR
      images/ + masks/ with the full 1-9 label scheme
      (e.g. data/tiled_data/finetuning_hard_negatives_complete, 66 tiles).
      Added to TRAINING only, supervised normally.

  --negative-train-dirs DIR
      images/ + negative_masks/ where the mask is BINARY: 1 = a confuser object
      a human rejected as not-waste, 0 = unlabelled
      (e.g. data/tiled_data/complementary_hard_negatives_v2, 26,597 tiles).
      Feeding these through the normal label map would read 1 as Waste and
      poison training. Instead the marked pixels get a "known not-waste, class
      unknown" loss, -log(1 - p_waste), and the rest of the tile is ignored.
      Added to TRAINING only.

Outputs are the same files v4 writes, so downstream analysis is unchanged:
config.json, split.csv, metrics.csv, per_tile_counts_<split>_<rule>.csv,
pr_curve_<split>.png, metrics_all_seeds.csv, metrics_summary.csv.

Example:
    python 03_unet_pytorch/01_train_unet_v5_torch.py \\
        --dev-dir  data/tiled_data/partitioned/modelDevelopment \\
        --test-dir data/tiled_data/partitioned/Validation \\
        --arch efficientnet-b0 --split-seed 42 --seeds 1 2 3
"""

import argparse
import copy
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import average_precision_score, precision_recall_curve

import torch
import torch.nn as nn
import torch.nn.functional as F

# Mask codes used internally (negative so they cannot collide with class ids)
IGNORE = -1      # unlabelled / no-data: excluded from loss and every metric
NOTWASTE = -2    # known not-waste, class unknown: waste-channel loss only
UNKNOWN = -3     # a raw label that is neither a class nor the ignore value

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# Arguments
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dev-dir", default=(
        r"data\tiled_data\partitioned\modelDevelopment"))
    p.add_argument("--test-dir", default=None,
                   help="Independent test folder. Never used for any decision.")
    p.add_argument("--extra-train-dirs", nargs="*", default=[],
                   help="Extra TRAINING folders with full 1-9 masks.")
    p.add_argument("--negative-train-dirs", nargs="*", default=[],
                   help="Extra TRAINING folders with binary negative_masks/ "
                        "(1 = confuser object, 0 = unlabelled).")
    p.add_argument("--results-dir", default="./results/03_unet_pytorch")
    p.add_argument("--exp-name", default=None)

    p.add_argument("--arch", default="scratch",
                   choices=["scratch", "resnet34", "efficientnet-b0", "dinov2",
                            "openearthmap", "dinov3-vits", "dinov3-sat",
                            "dinov3-convnext", "dofa",
                            # rs_backbones registry (strict weight checks)
                            "efficientnet-b4-imagenet", "efficientnet-b4-oem",
                            "dinov3-vitl", "dinov3-sat-vitl", "satlas-aerial"])
    p.add_argument("--oem-weights",
                   default=os.path.join("data", "pretrained_weights", "openearthmap",
                                        "RGB_Real_5_u-efficientnet-b4.pth"),
                   help="Checkpoint for --arch openearthmap (smp Unet, "
                        "efficientnet-b4 encoder, scSE decoder).")
    p.add_argument("--max-train-tiles", type=int, default=None,
                   help="Label budget: cap the number of DEVELOPMENT training "
                        "tiles (sampled with --split-seed so every model sees "
                        "the same subset). Validation is never reduced.")
    p.add_argument("--resample-budget-per-seed", action="store_true",
                   help="Draw a different label-budget subset for each "
                        "seed, so error bars include which tiles were "
                        "drawn. Without it all seeds share one subset "
                        "and small-n error bars are understated.")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Keep the foundation backbone frozen instead of "
                        "fine-tuning it after warm-up.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Same architecture, random weights (control that "
                        "separates architecture from pretraining).")
    p.add_argument("--freeze-epochs", type=int, default=10,
                   help="Pretrained encoders: warm-up epochs with encoder frozen.")
    p.add_argument("--warmup-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for --arch scratch (Adam, as in v4).")
    p.add_argument("--unfreeze-dinov2", action="store_true",
                   help="Fine-tune the DINOv2 backbone (default: frozen).")

    p.add_argument("--labels", nargs="+", type=int, default=list(range(1, 10)),
                   help="Raw labels of the REAL classes (default 1..9). "
                        "Do not list --ignore-label here.")
    p.add_argument("--waste-label", type=int, default=1)
    p.add_argument("--ignore-label", default="99",
                   help="No-data mask value (default 99); 'none' to disable.")

    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--split-seed", type=int, default=42,
                   help="Seed for the spatial train/val split. Fixed by default "
                        "so every model and seed sees the SAME split.")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--val-block-size", type=float, default=250.0)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr-patience", type=int, default=10)
    p.add_argument("--init-filters", type=int, default=16)
    p.add_argument("--neg-weight", type=float, default=1.0,
                   help="Weight on the not-waste loss for --negative-train-dirs.")
    p.add_argument("--max-negatives", type=int, default=None,
                   help="Randomly subsample the negative tiles to this many "
                        "(they can outnumber the real tiles 30:1).")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--landscape-prevalence", type=float, nargs="*",
                   default=[0.00045])
    p.add_argument("--target-precision", type=float, default=0.5)

    args = p.parse_args()
    raw = str(args.ignore_label).strip().lower()
    args.ignore_label = None if raw in ("none", "", "null") else int(raw)
    return args


def set_seed(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# =============================================================================
# Data loading  (ported from v4 so splits and prevalence match exactly)
# =============================================================================
def list_pairs(folder, mask_sub="masks"):
    img_dir = os.path.join(folder, "images")
    msk_dir = os.path.join(folder, mask_sub)

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


def load_folder(folder, label_map, img_size, tag, ignore_label=None,
                negative=False, max_tiles=None, rng=None):
    """Load tiles. Returns dict of arrays.

    negative=False: masks/ hold raw class labels, mapped through label_map;
                    ignore_label -> IGNORE.
    negative=True:  negative_masks/ are binary; 1 -> NOTWASTE, 0 -> IGNORE.
    """
    mask_sub = "negative_masks" if negative else "masks"
    img_dir, msk_dir, names = list_pairs(folder, mask_sub)

    if max_tiles is not None and len(names) > max_tiles:
        rng = rng or np.random.default_rng(0)
        names = [names[i] for i in
                 sorted(rng.choice(len(names), max_tiles, replace=False))]

    if not negative:
        top = max(list(label_map) + ([ignore_label] if ignore_label else []))
        lut = np.full(top + 1, UNKNOWN, dtype=np.int16)
        for raw, idx in label_map.items():
            lut[raw] = idx
        if ignore_label is not None:
            lut[ignore_label] = IGNORE

    images, masks, cx, cy = [], [], [], []
    n_ignored = n_notwaste = 0
    warned_dtype = False

    for name in names:
        with rasterio.open(os.path.join(img_dir, name)) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
            if img.dtype != np.uint8 and not warned_dtype:
                print(f"  Warning: imagery dtype {img.dtype}, not uint8.")
                warned_dtype = True
            t = src.transform
            cx.append(t.c + t.a * src.width / 2.0)
            cy.append(t.f + t.e * src.height / 2.0)

        with rasterio.open(os.path.join(msk_dir, name)) as src:
            m = src.read(1)

        if img.shape[:2] != (img_size, img_size) or m.shape != (img_size, img_size):
            raise ValueError(f"{name}: expected {img_size}x{img_size}, got "
                             f"image {img.shape[:2]}, mask {m.shape}")

        if negative:
            encoded = np.where(m > 0, NOTWASTE, IGNORE).astype(np.int16)
            n_notwaste += int(np.sum(encoded == NOTWASTE))
        else:
            if m.min() < 0 or m.max() >= len(lut) or np.any(lut[m] == UNKNOWN):
                bad = sorted(set(np.unique(m).tolist()) - set(label_map)
                             - {ignore_label})
                raise ValueError(
                    f"{name}: mask has labels not in --labels: {bad}. If these "
                    f"are unlabelled pixels pass --ignore-label {bad[0] if bad else ''}; "
                    f"if they are a real class add them to --labels.")
            encoded = lut[m].astype(np.int16)
        n_ignored += int(np.sum(encoded == IGNORE))

        images.append(np.clip(img, 0, 255).astype(np.uint8))
        masks.append(encoded)

    total = len(names) * img_size * img_size
    print(f"  Loaded {len(names)} tiles from {tag}: {folder}")
    if negative:
        print(f"    not-waste pixels: {n_notwaste:,} "
              f"({100 * n_notwaste / total:.2f}%); rest ignored")
    elif ignore_label is not None:
        print(f"    ignored pixels (label {ignore_label}): {n_ignored:,} "
              f"({100 * n_ignored / total:.2f}%)")
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


def spatial_split(data, block_size, val_frac, seed, waste_idx):
    """Identical to v4: same numpy RNG, so the same split is produced."""
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
        raise ValueError("Spatial split produced an empty set.")
    if not np.any(data["masks"][is_val] == waste_idx):
        raise ValueError("Validation blocks contain no waste.")

    print(f"  Spatial split: {len(unique)} blocks, {len(val_groups)} to val "
          f"({is_val.sum()} tiles), {(~is_val).sum()} tiles to train")
    return np.where(~is_val)[0], np.where(is_val)[0], groups


# =============================================================================
# Dataset
# =============================================================================
class TileDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, augment, seed=0):
        self.images, self.masks = images, masks
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        msk = self.masks[i]
        if self.augment:
            if self.rng.random() < 0.5:
                img, msk = img[::-1], msk[::-1]
            if self.rng.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            k = int(self.rng.integers(0, 4))
            img, msk = np.rot90(img, k), np.rot90(msk, k)
        x = torch.from_numpy(np.ascontiguousarray(
            img.transpose(2, 0, 1))).float().div_(255.0)
        y = torch.from_numpy(np.ascontiguousarray(msk)).long()
        return x, y


# =============================================================================
# Models
# =============================================================================
class ScratchUNet(nn.Module):
    """Port of v4's build_model: 16 filters, 3 levels, BN after each conv."""

    def __init__(self, n_classes, f=16):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                nn.BatchNorm2d(cout),
                nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
                nn.BatchNorm2d(cout))

        self.s1 = block(3, f)
        self.s2 = block(f, f * 2)
        self.s3 = block(f * 2, f * 4)
        self.b1 = block(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.u1 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.d1 = block(f * 8, f * 4)
        self.u2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.d2 = block(f * 4, f * 2)
        self.u3 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.d3 = block(f * 2, f)
        self.head = nn.Conv2d(f, n_classes, 1)

    def forward(self, x):
        s1 = self.s1(x)
        s2 = self.s2(self.pool(s1))
        s3 = self.s3(self.pool(s2))
        b1 = self.b1(self.pool(s3))
        d1 = self.d1(torch.cat([self.u1(b1), s3], 1))
        d2 = self.d2(torch.cat([self.u2(d1), s2], 1))
        d3 = self.d3(torch.cat([self.u3(d2), s1], 1))
        return self.head(d3)


class Normalize(nn.Module):
    """Dataset always yields [0,1]; each model normalises as its encoder expects."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("m", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.m) / self.s


class SmpUNet(nn.Module):
    def __init__(self, encoder, n_classes, pretrained=True,
                 decoder_attention_type=None):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.norm = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.net = smp.Unet(encoder_name=encoder,
                            encoder_weights="imagenet" if pretrained else None,
                            in_channels=3, classes=n_classes,
                            decoder_attention_type=decoder_attention_type)

    @property
    def encoder(self):
        return self.net.encoder

    def forward(self, x):
        return self.net(self.norm(x))


VIT_BACKBONES = {
    "dinov2": "vit_small_patch14_dinov2.lvd142m",
    "dinov3-vits": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3-sat": "vit_large_patch16_dinov3.sat493m",
}
CNN_BACKBONES = {
    "dinov3-convnext": "convnext_small.dinov3_lvd1689m",
}


def _patch_size(name):
    import re
    m = re.search(r"patch(\d+)", name)
    if not m:
        raise ValueError(f"cannot infer patch size from {name}")
    return int(m.group(1))


class ViTUNet(nn.Module):
    """ViT foundation backbone as the bottleneck of a U-Net.

    Works for any patch size: the tile is resized to the largest multiple of
    the patch size that fits in img_size (252 for DINOv2's stride 14, 256 for
    DINOv3's stride 16). Skip connections come from a small trainable stem run
    on the FULL-resolution tile, so fine detail is not lost to the patch grid
    -- which matters here because most commission error is boundary error.
    """

    def __init__(self, timm_name, n_classes, img_size=256, pretrained=True,
                 freeze=True):
        super().__init__()
        import timm
        self.norm = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        patch = _patch_size(timm_name)
        self.vit_size = (img_size // patch) * patch
        self.grid = self.vit_size // patch
        self.vit = timm.create_model(
            timm_name, pretrained=pretrained,
            img_size=self.vit_size, num_classes=0)
        self.n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        c = self.vit.embed_dim

        self.freeze = freeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.stem1 = block(3, 32)                # 256
        self.stem2 = block(32, 64)               # 128
        self.stem3 = block(64, 128)              # 64
        self.pool = nn.MaxPool2d(2)
        self.proj = nn.Conv2d(c, 128, 1)
        self.dec3 = block(128 + 128, 128)        # 64
        self.dec2 = block(128 + 64, 64)          # 128
        self.dec1 = block(64 + 32, 32)           # 256
        self.head = nn.Conv2d(32, n_classes, 1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.vit.eval()                      # keep frozen stats/dropout off
        return self

    def forward(self, x):
        xn = self.norm(x)
        s1 = self.stem1(xn)
        s2 = self.stem2(self.pool(s1))
        s3 = self.stem3(self.pool(s2))

        v = F.interpolate(xn, size=(self.vit_size, self.vit_size),
                          mode="bilinear", align_corners=False)
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            tok = self.vit.forward_features(v)
        tok = tok[:, self.n_prefix:, :]
        b, n, c = tok.shape
        f = tok.transpose(1, 2).reshape(b, c, self.grid, self.grid).float()

        f = self.proj(f)
        f = F.interpolate(f, size=s3.shape[-2:], mode="bilinear",
                          align_corners=False)
        d3 = self.dec3(torch.cat([f, s3], 1))
        d3 = F.interpolate(d3, size=s2.shape[-2:], mode="bilinear",
                           align_corners=False)
        d2 = self.dec2(torch.cat([d3, s2], 1))
        d2 = F.interpolate(d2, size=s1.shape[-2:], mode="bilinear",
                           align_corners=False)
        d1 = self.dec1(torch.cat([d2, s1], 1))
        return self.head(d1)


class CnnFeatureUNet(nn.Module):
    """Hierarchical CNN backbone (timm features_only) + U-Net decoder.

    Unlike a ViT this has no patch grid: features come out at strides 4, 8, 16
    and 32, so boundaries survive. Used for the ConvNeXt-DINOv3 weights, which
    test whether the boundary over-segmentation that dominates commission is a
    patch-resolution artefact rather than a representation failure.
    """

    def __init__(self, timm_name, n_classes, pretrained=True):
        super().__init__()
        import timm
        self.norm = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.enc = timm.create_model(timm_name, pretrained=pretrained,
                                     features_only=True, out_indices=(0, 1, 2, 3))
        chs = self.enc.feature_info.channels()          # stride 4, 8, 16, 32

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.d3 = block(chs[3] + chs[2], 256)
        self.d2 = block(256 + chs[1], 128)
        self.d1 = block(128 + chs[0], 64)
        self.d0 = block(64, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    @property
    def encoder(self):
        return self.enc

    def forward(self, x):
        f0, f1, f2, f3 = self.enc(self.norm(x))

        def up(t, ref):
            return F.interpolate(t, size=ref.shape[-2:], mode="bilinear",
                                 align_corners=False)

        y = self.d3(torch.cat([up(f3, f2), f2], 1))
        y = self.d2(torch.cat([up(y, f1), f1], 1))
        y = self.d1(torch.cat([up(y, f0), f0], 1))
        y = self.d0(F.interpolate(y, size=x.shape[-2:], mode="bilinear",
                                  align_corners=False))
        return self.head(y)


def _load_openearthmap(model, path):
    """Load OpenEarthMap U-Net weights, tolerating smp version drift.

    The published checkpoint was saved under smp 0.3.3; decoder key names have
    since changed. The encoder is what carries the aerial-domain pretraining,
    so it is loaded strictly and the decoder best-effort. The segmentation head
    is always dropped: OEM has its own 9 land-cover classes, not ours.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k: v for k, v in sd.items() if not k.startswith("segmentation_head")}
    tgt = model.net.state_dict()
    ok = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
    missing = model.net.load_state_dict(ok, strict=False)
    enc_tot = sum(1 for k in tgt if k.startswith("encoder"))
    enc_ok = sum(1 for k in ok if k.startswith("encoder"))
    dec_tot = sum(1 for k in tgt if k.startswith("decoder"))
    dec_ok = sum(1 for k in ok if k.startswith("decoder"))
    print(f"  OpenEarthMap weights: encoder {enc_ok}/{enc_tot}, "
          f"decoder {dec_ok}/{dec_tot} tensors loaded")
    if enc_ok < 0.9 * enc_tot:
        raise RuntimeError(
            f"only {enc_ok}/{enc_tot} encoder tensors matched -- the "
            f"checkpoint does not fit this architecture; do not report this "
            f"row as OpenEarthMap-pretrained.")
    return missing


def build(arch, n_classes, img_size, pretrained, freeze_backbone=False,
          oem_weights=None):
    """Returns (model, warmup_encoder). warmup_encoder is frozen during the
    warm-up phase then unfrozen; None means nothing to unfreeze.

    Architectures in the rs_backbones registry are delegated there, which
    enforces strict pretrained-weight verification (never a random fallback)
    and records how many tensors were transferred.
    """
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import rs_backbones as _B
    if arch in _B.REGISTRY:
        m, enc, rep = _B.build_registered(arch, n_classes, img_size,
                                          freeze_backbone, oem_weights)
        print("  " + rep.summary())
        print(f"    head: {rep.head_replaced}")
        print(f"    {rep.notes}")
        build.last_report = rep
        return m, enc
    if arch == "scratch":
        return ScratchUNet(n_classes), None
    if arch in ("resnet34", "efficientnet-b0"):
        m = SmpUNet(arch, n_classes, pretrained)
        return m, m.encoder
    if arch == "openearthmap":
        m = SmpUNet("efficientnet-b4", n_classes, pretrained=False,
                    decoder_attention_type="scse")
        if not oem_weights or not os.path.exists(oem_weights):
            raise FileNotFoundError(
                f"--oem-weights not found: {oem_weights}")
        _load_openearthmap(m, oem_weights)
        return m, m.encoder
    if arch in VIT_BACKBONES:
        m = ViTUNet(VIT_BACKBONES[arch], n_classes, img_size, pretrained,
                    freeze=freeze_backbone)
        return m, (None if freeze_backbone else m.vit)
    if arch in CNN_BACKBONES:
        m = CnnFeatureUNet(CNN_BACKBONES[arch], n_classes, pretrained)
        if freeze_backbone:
            for p in m.enc.parameters():
                p.requires_grad = False
            return m, None
        return m, m.enc
    if arch == "dofa":
        raise NotImplementedError(
            "DOFA needs torchgeo, which is not installed yet. Install it once "
            "the running queue finishes, then add the backbone here.")
    raise ValueError(arch)


# =============================================================================
# Loss
# =============================================================================
def seg_loss(logits, target, waste_idx, neg_weight):
    """Cross-entropy on labelled pixels + -log(1 - p_waste) on NOTWASTE pixels."""
    ce = F.cross_entropy(logits, target.clamp_min(0), reduction="none")
    labelled = target >= 0
    n_lab = labelled.sum()
    loss = (ce * labelled).sum() / n_lab.clamp_min(1) if n_lab > 0 else \
        logits.sum() * 0.0

    nw = target == NOTWASTE
    if neg_weight > 0 and nw.any():
        logp = F.log_softmax(logits, dim=1)
        p_waste = logp[:, waste_idx].exp().clamp(1e-6, 1 - 1e-6)
        loss = loss + neg_weight * (
            -torch.log1p(-p_waste) * nw).sum() / nw.sum()
    return loss


# =============================================================================
# Evaluation  (ported verbatim from v4)
# =============================================================================
@torch.no_grad()
def predict_waste(model, images, waste_idx, batch_size, device, amp):
    model.eval()
    is_waste, p_waste = [], []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i:i + batch_size].transpose(0, 3, 1, 2))
        x = x.to(device).float().div_(255.0)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            logits = model(x)
        probs = torch.softmax(logits.float(), dim=1)
        is_waste.append((probs.argmax(1) == waste_idx).cpu().numpy())
        p_waste.append(probs[:, waste_idx].cpu().numpy().astype(np.float16))
    return np.concatenate(is_waste), np.concatenate(p_waste)


def per_tile_counts(truth, pred, names, groups=None, valid=None):
    t = truth.reshape(len(truth), -1)
    p = pred.reshape(len(pred), -1)
    v = (np.ones_like(t) if valid is None else valid.reshape(len(valid), -1))
    df = pd.DataFrame({
        "tile": names,
        "tp": np.sum(t & p & v, axis=1),
        "fp": np.sum(~t & p & v, axis=1),
        "fn": np.sum(t & ~p & v, axis=1),
        "tn": np.sum(~t & ~p & v, axis=1),
        "n_ignored": np.sum(~v, axis=1),
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
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec + eps),
        "iou": tp / (tp + fp + fn + eps),
        "fpr": fpr, "specificity": 1 - fpr,
    }
    for p in landscape_prev:
        out[f"implied_precision_at_p={p:g}"] = (
            rec * p / (rec * p + fpr * (1 - p) + eps))
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
# Train / evaluate one seed
# =============================================================================
def run_epoch(model, loader, opt, scaler, device, waste_idx, neg_weight, amp,
              train):
    model.train(train)
    total, nb = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            loss = seg_loss(model(x), y, waste_idx, neg_weight)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        total += float(loss.detach())
        nb += 1
    return total / max(nb, 1)


def run_seed(seed, args, dev, test, extra, negatives, waste_idx, n_classes,
             device):
    out_dir = os.path.join(args.results_dir, args.exp_name, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n==================== seed {seed} ====================")
    set_seed(seed, args.deterministic)

    split_seed = args.split_seed if args.split_seed is not None else seed
    tr_idx, va_idx, groups = spatial_split(
        dev, args.val_block_size, args.val_frac, split_seed, waste_idx)
    train = subset(dev, tr_idx)
    val = subset(dev, va_idx)

    # Label budget: the data-scarcity axis. Sampled with split_seed, not the
    # run seed, so every architecture is trained on exactly the same tiles.
    if args.max_train_tiles and len(train["names"]) > args.max_train_tiles:
        # Draw depends on BOTH the split seed and the run seed, so each seed
        # sees a DIFFERENT subset (error bars then include draw variance, not
        # just initialisation) while every architecture at a given (budget,
        # seed) sees the SAME tiles, keeping models comparable.
        rng = np.random.default_rng(1000 * split_seed + seed
                                    if args.resample_budget_per_seed
                                    else split_seed)
        pick = np.sort(rng.choice(len(train["names"]),
                                  args.max_train_tiles, replace=False))
        train = subset(train, pick)
        n_waste_tiles = int((train["masks"] == waste_idx).any(axis=(1, 2)).sum())
        print(f"  Label budget: {args.max_train_tiles} of {len(tr_idx)} "
              f"training tiles ({n_waste_tiles} contain waste)")

    for pool in (extra, negatives):
        if pool is not None:
            train = concat(train, pool)

    pd.DataFrame({
        "tile": np.concatenate([dev["names"][tr_idx], dev["names"][va_idx]]),
        "block": np.concatenate([groups[tr_idx], groups[va_idx]]),
        "split": ["train"] * len(tr_idx) + ["val"] * len(va_idx),
    }).to_csv(os.path.join(out_dir, "split.csv"), index=False)

    tr_ds = TileDataset(train["images"], train["masks"], augment=True, seed=seed)
    va_ds = TileDataset(val["images"], val["masks"], augment=False, seed=seed)
    tr_ld = torch.utils.data.DataLoader(
        tr_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False)
    va_ld = torch.utils.data.DataLoader(
        va_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model, encoder = build(args.arch, n_classes, args.img_size,
                           not args.no_pretrained, args.freeze_backbone,
                           args.oem_weights)
    model.to(device)
    n_par = sum(p.numel() for p in model.parameters())
    n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {args.arch}: {n_par/1e6:.2f} M parameters "
          f"({n_trn/1e6:.2f} M trainable)")

    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    two_phase = encoder is not None and args.freeze_epochs > 0
    if two_phase:
        for p in encoder.parameters():
            p.requires_grad = False
        lr = args.warmup_lr
    else:
        lr = args.lr if args.arch == "scratch" else args.finetune_lr
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=args.lr_patience, min_lr=1e-6)

    best_loss, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for epoch in range(args.epochs):
        if two_phase and epoch == args.freeze_epochs:
            for p in encoder.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.finetune_lr)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.5, patience=args.lr_patience, min_lr=1e-6)
            print(f"  Epoch {epoch}: encoder unfrozen, lr -> {args.finetune_lr}")

        tr_loss = run_epoch(model, tr_ld, opt, scaler, device, waste_idx,
                            args.neg_weight, amp, train=True)
        with torch.no_grad():
            va_loss = run_epoch(model, va_ld, opt, scaler, device, waste_idx,
                                args.neg_weight, amp, train=False)
        sched.step(va_loss)
        hist.append({"epoch": epoch, "loss": tr_loss, "val_loss": va_loss,
                     "lr": opt.param_groups[0]["lr"]})

        improved = va_loss < best_loss - 1e-6
        if improved:
            best_loss, bad = va_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if epoch % 5 == 0 or improved:
            print(f"  epoch {epoch:3d}  train {tr_loss:.4f}  val {va_loss:.4f}"
                  f"{'  *' if improved else ''}  "
                  f"[{(time.time()-t0)/60:.1f} min]")
        if bad >= args.patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(best val_loss {best_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # ---- Threshold chosen on VAL only ----
    va_arg, va_prob = predict_waste(model, val["images"], waste_idx,
                                    args.batch_size, device, amp)
    va_truth = val["masks"] == waste_idx
    va_valid = val["masks"] >= 0
    t_star = choose_threshold(va_truth[va_valid],
                              va_prob[va_valid].astype(np.float32))
    print(f"  Threshold chosen on val: t* = {t_star:.4f}")

    sets = {"val": (val, va_arg, va_prob, groups[va_idx])}
    if test is not None:
        te_arg, te_prob = predict_waste(model, test["images"], waste_idx,
                                        args.batch_size, device, amp)
        sets["test"] = (test, te_arg, te_prob, None)

    rows = []
    for split_name, (data, pred_arg, prob, grp) in sets.items():
        truth = data["masks"] == waste_idx
        valid = data["masks"] >= 0
        prev = float(truth[valid].mean())
        ap = plot_pr(truth[valid], prob[valid].astype(np.float32), prev,
                     args.landscape_prevalence,
                     os.path.join(out_dir, f"pr_curve_{split_name}.png"),
                     f"Waste PR curve ({args.exp_name}, {split_name}, seed {seed})")
        for rule, pred in (("argmax", pred_arg),
                           ("thresh", prob.astype(np.float32) >= t_star)):
            counts = per_tile_counts(truth, pred, data["names"], grp, valid)
            counts.to_csv(os.path.join(
                out_dir, f"per_tile_counts_{split_name}_{rule}.csv"), index=False)
            m = pooled_metrics(counts, args.landscape_prevalence,
                               args.target_precision)
            rows.append({"seed": seed, "arch": args.arch, "split": split_name,
                         "rule": rule,
                         "threshold": t_star if rule == "thresh" else None,
                         "average_precision": ap,
                         "epochs_run": len(hist),
                         "minutes": (time.time() - t0) / 60, **m})

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
    if args.exp_name is None:
        tag = args.arch.replace("-", "")
        if args.no_pretrained:
            tag += "_scratchinit"
        if args.negative_train_dirs or args.extra_train_dirs:
            tag += "_hn"
        args.exp_name = f"torch_{tag}"

    labels = sorted(args.labels)
    if args.waste_label not in labels:
        raise ValueError("--waste-label must be in --labels")
    if args.ignore_label is not None and args.ignore_label in labels:
        raise ValueError(f"--ignore-label {args.ignore_label} is also in "
                         f"--labels; it is either a class or unlabelled.")
    label_map = {raw: i for i, raw in enumerate(labels)}
    waste_idx = label_map[args.waste_label]
    n_classes = len(labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)}  "
              f"(torch {torch.__version__})")
    else:
        print(f"Device: CPU  (torch {torch.__version__})  "
              f"-- WARNING: no CUDA, this will be very slow")

    exp_dir = os.path.join(args.results_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "label_map": label_map,
                   "waste_model_channel": waste_idx,
                   "torch": torch.__version__,
                   "device": str(device)}, f, indent=2)

    print(f"Labels {labels} -> channels 0..{n_classes - 1}; "
          f"waste label {args.waste_label} -> channel {waste_idx}")

    dev = load_folder(args.dev_dir, label_map, args.img_size, "dev",
                      args.ignore_label)
    test = (load_folder(args.test_dir, label_map, args.img_size, "test",
                        args.ignore_label) if args.test_dir else None)
    if test is None:
        print("  Warning: no --test-dir; val metrics are NOT independent.")

    extra = None
    if args.extra_train_dirs:
        extra = concat(*[load_folder(d, label_map, args.img_size, "extra",
                                     args.ignore_label)
                         for d in args.extra_train_dirs])
    negatives = None
    if args.negative_train_dirs:
        rng = np.random.default_rng(args.split_seed or 0)
        negatives = concat(*[load_folder(d, label_map, args.img_size, "neg",
                                         args.ignore_label, negative=True,
                                         max_tiles=args.max_negatives, rng=rng)
                             for d in args.negative_train_dirs])

    all_metrics = pd.concat([
        run_seed(s, args, dev, test, extra, negatives, waste_idx, n_classes,
                 device)
        for s in args.seeds])
    all_metrics.to_csv(os.path.join(exp_dir, "metrics_all_seeds.csv"),
                       index=False)

    num_cols = [c for c in all_metrics.columns
                if c not in ("seed", "arch", "split", "rule", "threshold")
                and np.issubdtype(all_metrics[c].dtype, np.number)]
    summary = (all_metrics.groupby(["split", "rule"])[num_cols]
               .agg(["mean", "std"]))
    summary.to_csv(os.path.join(exp_dir, "metrics_summary.csv"))
    print("\n=========== Summary across seeds (mean) ===========")
    print(summary.xs("mean", axis=1, level=1)[
        ["prevalence", "precision", "recall", "f1", "iou", "fpr"]].to_string())


if __name__ == "__main__":
    main()
