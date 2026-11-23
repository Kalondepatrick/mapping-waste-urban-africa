# =============================================================================
# SELECTIVE ERROR-CORRECTION FINE-TUNING OF THE BASELINE 10-CLASS U-NET
# =============================================================================
#
# SCIENTIFIC IDEA
# ---------------
# Start from the ORIGINAL March baseline model.
#
# Two types of supervision are used:
#
#   A. Original fully labelled training data
#      -------------------------------------
#      These data contain exact class labels, including exact Waste pixels.
#      They continue to use ordinary 10-class categorical cross-entropy.
#
#      In addition, genuine Waste pixels receive an explicit "positive
#      preservation" loss so that fine-tuning does not solve false positives
#      by simply becoming reluctant to predict Waste everywhere.
#
#   B. Reviewed waste-absence / false-positive regions
#      ------------------------------------------------
#      These regions are known NOT to contain Waste. The model is NOT told
#      which of the other nine classes is correct.
#
#      The frozen baseline model is used to identify the pixels that actually
#      caused the problem. Negative correction is applied only where:
#
#          teacher argmax == Waste
#          OR
#          P_teacher(Waste) >= 0.50
#
#      Within those confirmed non-waste pixels, the correction is weighted by
#      the baseline model's original Waste probability:
#
#          weight = P_teacher(Waste) ** ERROR_POWER
#
#      Therefore:
#          - a pixel that the baseline called Waste with 0.95 probability
#            receives strong corrective pressure;
#          - a pixel already given low Waste probability receives little or no
#            pressure;
#          - the model is NOT forced into a guessed alternative class.
#
# LOSS
# ----
#
#   L =
#       L_10class_supervised
#       + POSITIVE_PRESERVATION_WEIGHT * L_confirmed_waste
#       + SELECTIVE_NEGATIVE_WEIGHT * L_targeted_not_waste
#
# where:
#
#   L_confirmed_waste =
#       -log(P_student(Waste))
#       on manually labelled genuine Waste pixels
#
#   L_targeted_not_waste =
#       teacher-error-weighted [-log(1 - P_student(Waste))]
#       only in reviewed non-waste regions that fooled the baseline.
#
# IMPORTANT DATA SEMANTICS
# ------------------------
# The binary negative mask defines where "Waste is absent" is actually known.
# In the current hard-negative library, mask=1 corresponds to verified
# false-positive regions and mask=0 means unknown / ignore. The script does
# NOT automatically treat the whole 256x256 chip as waste-free.
#
# If future chips are exhaustively reviewed and truly contain no Waste
# anywhere, their binary negative mask may simply contain 1 over the entire
# reviewed region.
#
# MODEL SELECTION
# ---------------
# A fixed internal validation set is used. A candidate checkpoint is accepted
# only if:
#   1. validation loss improves;
#   2. validation Waste recognition on genuine labelled Waste remains within
#      a small tolerance of the frozen baseline; and
#   3. the student improves the false-positive Waste argmax rate relative to
#      the frozen baseline.
#
# The locked independent test AOI is not used during training, validation,
# checkpointing, or early stopping.
# =============================================================================

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from scipy.ndimage import rotate
from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TerminateOnNaN,
    Callback,
)

# =============================================================================
# 1. REPRODUCIBILITY
# =============================================================================

SEED = 20260919
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
# 2. PATHS
# =============================================================================

BASE_PATH = (
    Path.home()
    / "Documents"
    / "GitHub"
    / "mapping-waste-urban-africa"
)

# ORIGINAL model used for both frozen teacher and trainable student start.
BASE_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_march_finetuned_best.keras"
)

ORIGINAL_DATA_DIR = (
    BASE_PATH
    / "data"
    / "tiled_data"
    / "partitioned"
    / "modelDevelopment"
)

ORIGINAL_IMAGE_DIR = ORIGINAL_DATA_DIR / "images"
ORIGINAL_MASK_DIR = ORIGINAL_DATA_DIR / "masks"

NEGATIVE_DATA_DIR = (
    BASE_PATH
    / "data"
    / "tiled_data"
    / "complementary_hard_negatives_v2"
)

NEGATIVE_IMAGE_DIR = NEGATIVE_DATA_DIR / "images"
NEGATIVE_MASK_DIR = NEGATIVE_DATA_DIR / "negative_masks"
NEGATIVE_MANIFEST_PATH = NEGATIVE_DATA_DIR / "finetuning_manifest.csv"

RESULTS_DIR = (
    BASE_PATH
    / "results"
    / "selective_error_correction"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_selective_error_correction_best.keras"
)

FINAL_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_selective_error_correction_finetuned.keras"
)

# =============================================================================
# 3. SETTINGS
# =============================================================================

IMG_HEIGHT = 256
IMG_WIDTH = 256

# Frozen teacher + trainable student are both resident in memory.
BATCH_SIZE = 4

N_CLASSES = 10
WASTE_RAW_LABEL = 1
WASTE_CLASS_IDX = 0

LABELS = list(range(1, 11))
VALID_RAW_LABELS = set(LABELS)
LABEL_MAP = {raw: i for i, raw in enumerate(LABELS)}

MAX_EPOCHS = 300
INITIAL_LEARNING_RATE = 1e-5
MIN_LEARNING_RATE = 1e-7
EARLY_STOP_PATIENCE = 25
REDUCE_LR_PATIENCE = 8
MIN_DELTA = 1e-4

# ---------------------------------------------------------------------
# Rotating hard-negative sampling
# ---------------------------------------------------------------------
#
# All original fully labelled training chips are replayed every epoch.
# Hard-negative objects rotate through the training pool.
#
N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH = 300
N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH = 200

# Fixed internal validation sets.
ORIGINAL_VALIDATION_FRACTION = 0.15
N_STAGE1_VALIDATION_OBJECTS = 150
N_STAGE2_VALIDATION_OBJECTS = 100

# ---------------------------------------------------------------------
# Selective error targeting
# ---------------------------------------------------------------------
#
# A confirmed non-waste pixel is corrected only if the frozen baseline either
# predicts Waste as the winning class or gives Waste at least this probability.
#
TEACHER_WASTE_PROBABILITY_TRIGGER = 0.50

# Stronger baseline mistakes receive disproportionately stronger correction.
# Examples:
#   P=0.95 -> weight 0.9025
#   P=0.70 -> weight 0.4900
#   P=0.55 -> weight 0.3025
#
TEACHER_ERROR_POWER = 2.0

# ---------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------
#
# Positive preservation is deliberately strong because the previous broad
# negative fine-tuning reduced true-waste recall.
#
POSITIVE_PRESERVATION_LOSS_WEIGHT = 1.0

# Selective negative correction is deliberately conservative. It only acts on
# reviewed non-waste pixels that actually fooled the baseline.
#
SELECTIVE_NEGATIVE_LOSS_WEIGHT = 0.50

# ---------------------------------------------------------------------
# Checkpoint protection for genuine Waste
# ---------------------------------------------------------------------
#
# A fine-tuned checkpoint is rejected if its validation Waste argmax recall
# falls by more than 2 percentage points relative to the frozen baseline on
# the original labelled validation Waste pixels.
#
MAX_ALLOWED_POSITIVE_RECALL_DROP = 0.02

EPSILON = 1e-7

# =============================================================================
# 4. INPUT CHECKS
# =============================================================================

required = [
    BASE_MODEL_PATH,
    ORIGINAL_IMAGE_DIR,
    ORIGINAL_MASK_DIR,
    NEGATIVE_IMAGE_DIR,
    NEGATIVE_MASK_DIR,
    NEGATIVE_MANIFEST_PATH,
]

for p in required:
    if not p.exists():
        raise FileNotFoundError(f"\nRequired input not found:\n{p}")

print("\n" + "=" * 80)
print("SELECTIVE ERROR-CORRECTION FINE-TUNING")
print("=" * 80)

print(
    f"\nOriginal baseline used as frozen teacher and student initialisation:\n"
    f"{BASE_MODEL_PATH}"
)

print(
    f"\nReviewed hard-negative library:\n"
    f"{NEGATIVE_DATA_DIR}"
)

print(
    f"\nTeacher Waste-probability trigger: "
    f"{TEACHER_WASTE_PROBABILITY_TRIGGER}"
)

print(
    f"Teacher error-weight power: "
    f"{TEACHER_ERROR_POWER}"
)

print(
    f"Positive-preservation loss weight: "
    f"{POSITIVE_PRESERVATION_LOSS_WEIGHT}"
)

print(
    f"Selective-negative loss weight: "
    f"{SELECTIVE_NEGATIVE_LOSS_WEIGHT}"
)

print(
    f"Maximum allowed validation Waste-recall drop: "
    f"{MAX_ALLOWED_POSITIVE_RECALL_DROP:.3f}"
)

# =============================================================================
# 5. IO / AUGMENTATION
# =============================================================================

def encode_raw_mask(mask):
    out = np.full(mask.shape, -1, dtype=np.int32)
    for raw, idx in LABEL_MAP.items():
        out[mask == raw] = idx
    if np.any(out < 0):
        bad = np.unique(mask[out < 0])
        raise ValueError(f"Original mask has invalid labels: {bad.tolist()}")
    return out


def validate_negative_mask(mask, path):
    values = {int(v) for v in np.unique(mask)}
    invalid = sorted(values - {0, 1, 255})
    if invalid:
        raise ValueError(
            f"\nInvalid complementary mask:\n{path}\n"
            f"Observed values: {sorted(values)}\n"
            f"Invalid values: {invalid}"
        )
    binary = (mask > 0).astype(np.float32)
    if binary.sum() == 0:
        raise ValueError(f"\nNo confirmed-negative pixels in:\n{path}")
    return binary


def read_rgb(path):
    with rasterio.open(path) as src:
        x = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
    x = resize(
        x,
        (IMG_HEIGHT, IMG_WIDTH),
        preserve_range=True,
    )
    return (x / 255.0).astype(np.float32)


def read_original_mask(path):
    with rasterio.open(path) as src:
        y = src.read(1)
    y = resize(
        y,
        (IMG_HEIGHT, IMG_WIDTH),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return np.rint(y).astype(np.int32)


def read_negative_mask(path):
    with rasterio.open(path) as src:
        y = src.read(1)
    y = resize(
        y,
        (IMG_HEIGHT, IMG_WIDTH),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return validate_negative_mask(np.rint(y).astype(np.int32), path)


def augment_pair(image, mask):
    if random.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)

    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        image = rotate(
            image,
            angle,
            reshape=False,
            order=1,
            mode="reflect",
        )
        mask = rotate(
            mask,
            angle,
            reshape=False,
            order=0,
            mode="nearest",
        )
    return image, mask

# =============================================================================
# 6. ORIGINAL FULLY LABELLED DATA
# =============================================================================

print("\n" + "=" * 80)
print("AUDITING ORIGINAL FULLY LABELLED DATA")
print("=" * 80)

original_images = {
    p.name: p
    for p in ORIGINAL_IMAGE_DIR.glob("*.tif")
    if not p.name.endswith(".aux.xml")
}

original_masks = {
    p.name: p
    for p in ORIGINAL_MASK_DIR.glob("*.tif")
    if not p.name.endswith(".aux.xml")
}

common = sorted(set(original_images).intersection(original_masks))
if not common:
    raise FileNotFoundError("No matching original image-mask pairs found.")

original_records = []
excluded = []

for filename in common:
    image_path = original_images[filename]
    mask_path = original_masks[filename]

    with rasterio.open(mask_path) as src:
        raw = src.read(1)

    observed = {int(v) for v in np.unique(raw)}
    bad = sorted(observed - VALID_RAW_LABELS)

    if bad:
        excluded.append({"filename": filename, "invalid_labels": str(bad)})
        continue

    original_records.append(
        {
            "image_path": image_path,
            "mask_path": mask_path,
            "negative_mask_path": None,
            "source": "original",
            "negative_source": None,
            "sample_id": f"original::{filename}",
            "tile_id": filename,
            "contains_waste": bool(np.any(raw == WASTE_RAW_LABEL)),
        }
    )

print(f"Clean original pairs: {len(original_records):,}")
print(
    "Original chips containing Waste: "
    f"{sum(r['contains_waste'] for r in original_records):,}"
)

pd.DataFrame(excluded).to_csv(
    RESULTS_DIR / "excluded_original_masks.csv",
    index=False,
)

# Fixed internal validation split, stratified by presence of Waste.
original_df = pd.DataFrame(original_records)
train_parts = []
val_parts = []

for waste_flag, group in original_df.groupby("contains_waste"):
    group = group.sample(
        frac=1.0,
        random_state=SEED + (1 if waste_flag else 2),
    ).reset_index(drop=True)

    n_val = max(
        1,
        int(round(len(group) * ORIGINAL_VALIDATION_FRACTION)),
    )

    val_parts.append(group.iloc[:n_val].copy())
    train_parts.append(group.iloc[n_val:].copy())

original_train_df = pd.concat(train_parts, ignore_index=True)
original_val_df = pd.concat(val_parts, ignore_index=True)

original_train_records = original_train_df.to_dict("records")
original_val_records = original_val_df.to_dict("records")

print(f"Original training records: {len(original_train_records):,}")
print(f"Original validation records: {len(original_val_records):,}")
print(
    "Validation chips containing Waste: "
    f"{sum(r['contains_waste'] for r in original_val_records):,}"
)

pd.concat(
    [
        original_train_df.assign(internal_split="train"),
        original_val_df.assign(internal_split="validation"),
    ],
    ignore_index=True,
).to_csv(
    RESULTS_DIR / "original_internal_split.csv",
    index=False,
)

# =============================================================================
# 7. HARD-NEGATIVE LIBRARY
# =============================================================================

print("\n" + "=" * 80)
print("LOADING HARD-NEGATIVE LIBRARY")
print("=" * 80)

manifest = pd.read_csv(NEGATIVE_MANIFEST_PATH, dtype=str)

for col in ["tile_id", "image_file", "negative_mask_file"]:
    if col not in manifest.columns:
        raise ValueError(f"Manifest missing required column: {col}")

if "focal_candidate_id" in manifest.columns:
    manifest["object_id"] = manifest["focal_candidate_id"].astype(str)
elif "sample_id" in manifest.columns:
    manifest["object_id"] = manifest["sample_id"].astype(str)
else:
    raise ValueError("Manifest needs focal_candidate_id or sample_id.")

if "focal_source" in manifest.columns:
    manifest["negative_source"] = manifest["focal_source"].astype(str)
elif "negative_source" in manifest.columns:
    manifest["negative_source"] = manifest["negative_source"].astype(str)
elif "source_type" in manifest.columns:
    source_map = {
        "stage1_rejected_full_negative": "stage1_rejected",
        "stage2_not_waste_partial_negative": "stage2_not_waste",
    }
    manifest["negative_source"] = (
        manifest["source_type"].map(source_map).fillna(manifest["source_type"])
    )
else:
    raise ValueError("Manifest has no recognised negative-source field.")

manifest = manifest[
    manifest["negative_source"].isin(
        ["stage1_rejected", "stage2_not_waste"]
    )
].copy()

negative_records = []

for _, row in manifest.iterrows():
    image_path = NEGATIVE_IMAGE_DIR / str(row["image_file"])
    mask_path = NEGATIVE_MASK_DIR / str(row["negative_mask_file"])

    if not image_path.exists():
        raise FileNotFoundError(f"\nHard-negative image missing:\n{image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"\nHard-negative mask missing:\n{mask_path}")

    negative_records.append(
        {
            "image_path": image_path,
            "mask_path": None,
            "negative_mask_path": mask_path,
            "source": "hard_negative",
            "negative_source": str(row["negative_source"]),
            "sample_id": str(row["object_id"]),
            "tile_id": str(row["tile_id"]),
            "contains_waste": False,
        }
    )

negative_df = pd.DataFrame(negative_records)

for source_name in ["stage1_rejected", "stage2_not_waste"]:
    d = negative_df[negative_df["negative_source"] == source_name]
    print(
        f"{source_name}: {len(d):,} chips from "
        f"{d['sample_id'].nunique():,} unique objects"
    )

# =============================================================================
# 8. FIXED NEGATIVE VALIDATION SPLIT BY OBJECT
# =============================================================================

rng = np.random.default_rng(SEED)

n_val_by_source = {
    "stage1_rejected": N_STAGE1_VALIDATION_OBJECTS,
    "stage2_not_waste": N_STAGE2_VALIDATION_OBJECTS,
}

negative_train_parts = []
negative_val_parts = []

for source_name in ["stage1_rejected", "stage2_not_waste"]:
    d = negative_df[negative_df["negative_source"] == source_name].copy()

    object_ids = d["sample_id"].drop_duplicates().astype(str).to_numpy()
    rng.shuffle(object_ids)

    n_val = n_val_by_source[source_name]

    if n_val >= len(object_ids):
        raise ValueError(
            f"Too many validation objects requested for {source_name}."
        )

    val_ids = set(object_ids[:n_val])

    train_d = d[~d["sample_id"].isin(val_ids)].copy()
    val_all = d[d["sample_id"].isin(val_ids)].copy()

    # One deterministic chip per held-out validation object.
    val_d = (
        val_all
        .sort_values(["sample_id", "tile_id"])
        .groupby("sample_id", as_index=False)
        .first()
    )

    negative_train_parts.append(train_d)
    negative_val_parts.append(val_d)

negative_train_df = pd.concat(negative_train_parts, ignore_index=True)
negative_val_df = pd.concat(negative_val_parts, ignore_index=True)

for source_name in ["stage1_rejected", "stage2_not_waste"]:
    tr = negative_train_df[
        negative_train_df["negative_source"] == source_name
    ]
    va = negative_val_df[
        negative_val_df["negative_source"] == source_name
    ]
    print(
        f"{source_name}: "
        f"{tr['sample_id'].nunique():,} training objects, "
        f"{va['sample_id'].nunique():,} validation objects"
    )

pd.concat(
    [
        negative_train_df.assign(internal_split="train"),
        negative_val_df.assign(internal_split="validation"),
    ],
    ignore_index=True,
).to_csv(
    RESULTS_DIR / "hard_negative_internal_split.csv",
    index=False,
)

# Quick binary-mask check.
audit_df = negative_df.sample(
    n=min(200, len(negative_df)),
    random_state=SEED,
)

for _, row in audit_df.iterrows():
    with rasterio.open(row["negative_mask_path"]) as src:
        raw = src.read(1)
    validate_negative_mask(raw, row["negative_mask_path"])

print("Random complementary-mask audit passed.")

# =============================================================================
# 9. CYCLING OBJECT SAMPLER
# =============================================================================

class CyclingObjectSampler:

    def __init__(self, df, source_name, seed):
        self.df = df[df["negative_source"] == source_name].copy()
        if self.df.empty:
            raise ValueError(f"No training negatives for {source_name}")

        self.rng = np.random.default_rng(seed)

        self.object_to_indices = {
            str(object_id): group.index.to_numpy()
            for object_id, group in self.df.groupby("sample_id")
        }

        self.object_ids = np.array(
            list(self.object_to_indices.keys()),
            dtype=object,
        )

        self.order = np.array([], dtype=object)
        self.position = 0
        self._reshuffle()

    def _reshuffle(self):
        self.order = self.object_ids.copy()
        self.rng.shuffle(self.order)
        self.position = 0

    def sample(self, n_objects):
        chosen = []

        while len(chosen) < n_objects:
            remaining = len(self.order) - self.position
            needed = n_objects - len(chosen)
            take = min(remaining, needed)

            if take > 0:
                chosen.extend(
                    self.order[
                        self.position:
                        self.position + take
                    ].tolist()
                )
                self.position += take

            if len(chosen) < n_objects:
                self._reshuffle()

        records = []

        for object_id in chosen:
            indices = self.object_to_indices[str(object_id)]
            selected_index = int(self.rng.choice(indices))
            records.append(self.df.loc[selected_index].to_dict())

        return records

# =============================================================================
# 10. DATA SEQUENCES
# =============================================================================
#
# y_true has 11 channels:
#   0-9 : ordinary one-hot class target for fully labelled original pixels
#   10  : manually confirmed NOT-WASTE mask
#
# The teacher distribution is generated on-the-fly inside train_step/test_step.
# =============================================================================

class MixedSequence(Sequence):

    def __init__(
        self,
        records,
        batch_size,
        shuffle,
        augment,
        seed,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.records = list(records)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.seed = seed
        self.epoch_number = 0
        self.indices = np.arange(len(self.records))
        self._reshuffle_indices()

    def _reshuffle_indices(self):
        self.indices = np.arange(len(self.records))
        if self.shuffle:
            rng = np.random.default_rng(
                self.seed + self.epoch_number
            )
            rng.shuffle(self.indices)

    def __len__(self):
        return int(math.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        self.epoch_number += 1
        self._reshuffle_indices()

    def __getitem__(self, idx):
        inds = self.indices[
            idx * self.batch_size:
            (idx + 1) * self.batch_size
        ]

        xs = []
        ys = []

        for i in inds:
            record = self.records[i]
            image = read_rgb(record["image_path"])

            if record["source"] == "original":
                mask = read_original_mask(record["mask_path"])

                if self.augment:
                    image, mask = augment_pair(image, mask)

                mapped = encode_raw_mask(np.rint(mask).astype(np.int32))

                supervised = to_categorical(
                    mapped,
                    num_classes=N_CLASSES,
                ).astype(np.float32)

                neg = np.zeros(
                    (IMG_HEIGHT, IMG_WIDTH, 1),
                    dtype=np.float32,
                )

            elif record["source"] == "hard_negative":
                neg_mask = read_negative_mask(
                    record["negative_mask_path"]
                )

                if self.augment:
                    image, neg_mask = augment_pair(
                        image,
                        neg_mask,
                    )

                neg_mask = (neg_mask > 0.5).astype(np.float32)

                supervised = np.zeros(
                    (IMG_HEIGHT, IMG_WIDTH, N_CLASSES),
                    dtype=np.float32,
                )

                neg = neg_mask[..., np.newaxis]

            else:
                raise ValueError(
                    f"Unknown record source: {record['source']}"
                )

            target = np.concatenate([supervised, neg], axis=-1)

            xs.append(image.astype(np.float32))
            ys.append(target.astype(np.float32))

        return (
            np.asarray(xs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32),
        )


class RotatingTrainingSequence(MixedSequence):

    def __init__(
        self,
        original_records,
        negative_df,
        batch_size,
        n_stage1,
        n_stage2,
        seed,
        **kwargs,
    ):
        self.original_records = list(original_records)
        self.n_stage1 = n_stage1
        self.n_stage2 = n_stage2
        self.pool_epoch = 0
        self.selection_history = []

        self.stage1_sampler = CyclingObjectSampler(
            negative_df,
            "stage1_rejected",
            seed + 1001,
        )

        self.stage2_sampler = CyclingObjectSampler(
            negative_df,
            "stage2_not_waste",
            seed + 2001,
        )

        first_pool = self._build_pool()

        super().__init__(
            records=first_pool,
            batch_size=batch_size,
            shuffle=True,
            augment=True,
            seed=seed,
            **kwargs,
        )

    def _build_pool(self):
        self.pool_epoch += 1

        stage1 = self.stage1_sampler.sample(self.n_stage1)
        stage2 = self.stage2_sampler.sample(self.n_stage2)

        for record in stage1 + stage2:
            self.selection_history.append(
                {
                    "epoch": self.pool_epoch,
                    "negative_source": record["negative_source"],
                    "sample_id": record["sample_id"],
                    "tile_id": record["tile_id"],
                }
            )

        records = list(self.original_records) + stage1 + stage2

        if self.pool_epoch == 1 or self.pool_epoch % 10 == 0:
            print(
                f"\n[Epoch pool {self.pool_epoch}] "
                f"original={len(self.original_records):,}, "
                f"stage1_neg={len(stage1):,}, "
                f"stage2_neg={len(stage2):,}, "
                f"total={len(records):,}"
            )

        return records

    def on_epoch_end(self):
        # Keras calls this after each completed epoch.
        self.records = self._build_pool()
        self.epoch_number += 1
        self._reshuffle_indices()


train_gen = RotatingTrainingSequence(
    original_records=original_train_records,
    negative_df=negative_train_df,
    batch_size=BATCH_SIZE,
    n_stage1=N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH,
    n_stage2=N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH,
    seed=SEED,
)

validation_records = (
    list(original_val_records)
    + negative_val_df.to_dict("records")
)

val_gen = MixedSequence(
    records=validation_records,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False,
    seed=SEED + 50000,
)

print("\n" + "=" * 80)
print("DATA COMPOSITION")
print("=" * 80)
print(
    f"Original training records / epoch: "
    f"{len(original_train_records):,}"
)
print(
    f"Stage-1 negatives / epoch: "
    f"{N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH:,}"
)
print(
    f"Stage-2 negatives / epoch: "
    f"{N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH:,}"
)
print(f"Training records / epoch: {len(train_gen.records):,}")
print(f"Training batches / epoch: {len(train_gen):,}")
print(f"Fixed validation records: {len(validation_records):,}")
print(f"Validation batches: {len(val_gen):,}")

# =============================================================================
# 11. LOAD FROZEN TEACHER AND TRAINABLE STUDENT
# =============================================================================

print("\n" + "=" * 80)
print("LOADING TEACHER AND STUDENT")
print("=" * 80)

tf.keras.backend.clear_session()

teacher = load_model(
    BASE_MODEL_PATH,
    compile=False,
)

student = load_model(
    BASE_MODEL_PATH,
    compile=False,
)

for model_name, model in [
    ("teacher", teacher),
    ("student", student),
]:
    if tuple(model.input_shape[1:]) != (
        IMG_HEIGHT,
        IMG_WIDTH,
        3,
    ):
        raise ValueError(
            f"Unexpected {model_name} input shape: "
            f"{model.input_shape}"
        )

    if int(model.output_shape[-1]) != N_CLASSES:
        raise ValueError(
            f"{model_name} does not have "
            f"{N_CLASSES} output classes."
        )

teacher.trainable = False
for layer in teacher.layers:
    layer.trainable = False

for layer in student.layers:
    layer.trainable = True

print("Teacher frozen.")
print("Student trainable.")
print(f"Output classes: {student.output_shape[-1]}")

# =============================================================================
# 12. CUSTOM TRAINER
# =============================================================================

class SelectiveErrorCorrectionFineTuner(tf.keras.Model):

    def __init__(self, student, teacher, **kwargs):
        super().__init__(**kwargs)

        self.student = student
        self.teacher = teacher

        # Loss trackers.
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.supervised_loss_tracker = tf.keras.metrics.Mean(
            name="supervised_loss"
        )
        self.positive_preservation_loss_tracker = tf.keras.metrics.Mean(
            name="positive_preservation_loss"
        )
        self.selective_negative_loss_tracker = tf.keras.metrics.Mean(
            name="selective_negative_loss"
        )

        # Original fully supervised task.
        self.supervised_accuracy_tracker = tf.keras.metrics.Mean(
            name="supervised_accuracy"
        )

        # Genuine labelled Waste pixels.
        self.student_positive_mean_p_tracker = tf.keras.metrics.Mean(
            name="student_positive_mean_waste_probability"
        )
        self.teacher_positive_mean_p_tracker = tf.keras.metrics.Mean(
            name="teacher_positive_mean_waste_probability"
        )
        self.student_positive_recall_tracker = tf.keras.metrics.Mean(
            name="student_positive_waste_argmax_recall"
        )
        self.teacher_positive_recall_tracker = tf.keras.metrics.Mean(
            name="teacher_positive_waste_argmax_recall"
        )

        # Confirmed non-waste pixels.
        self.student_negative_mean_p_tracker = tf.keras.metrics.Mean(
            name="student_negative_mean_waste_probability"
        )
        self.teacher_negative_mean_p_tracker = tf.keras.metrics.Mean(
            name="teacher_negative_mean_waste_probability"
        )
        self.student_negative_argmax_tracker = tf.keras.metrics.Mean(
            name="student_negative_waste_argmax_rate"
        )
        self.teacher_negative_argmax_tracker = tf.keras.metrics.Mean(
            name="teacher_negative_waste_argmax_rate"
        )

        # Pixels that actually fooled the frozen baseline.
        self.targeted_negative_fraction_tracker = tf.keras.metrics.Mean(
            name="targeted_negative_fraction"
        )
        self.student_targeted_mean_p_tracker = tf.keras.metrics.Mean(
            name="student_targeted_mean_waste_probability"
        )
        self.teacher_targeted_mean_p_tracker = tf.keras.metrics.Mean(
            name="teacher_targeted_mean_waste_probability"
        )

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.supervised_loss_tracker,
            self.positive_preservation_loss_tracker,
            self.selective_negative_loss_tracker,
            self.supervised_accuracy_tracker,
            self.student_positive_mean_p_tracker,
            self.teacher_positive_mean_p_tracker,
            self.student_positive_recall_tracker,
            self.teacher_positive_recall_tracker,
            self.student_negative_mean_p_tracker,
            self.teacher_negative_mean_p_tracker,
            self.student_negative_argmax_tracker,
            self.teacher_negative_argmax_tracker,
            self.targeted_negative_fraction_tracker,
            self.student_targeted_mean_p_tracker,
            self.teacher_targeted_mean_p_tracker,
        ]

    def call(self, inputs, training=False):
        return self.student(
            inputs,
            training=training,
        )

    def compute_components(self, x, y_true, training):

        # -----------------------------------------------------------------
        # Targets
        # -----------------------------------------------------------------

        supervised_target = y_true[..., :N_CLASSES]
        negative_mask = y_true[..., N_CLASSES]

        supervised_mask = tf.cast(
            tf.reduce_sum(
                supervised_target,
                axis=-1,
            ) > 0.5,
            tf.float32,
        )

        # Ground-truth Waste pixels from the original fully labelled data.
        positive_waste_mask = supervised_target[
            ...,
            WASTE_CLASS_IDX
        ]

        # -----------------------------------------------------------------
        # Frozen baseline teacher and trainable student
        # -----------------------------------------------------------------

        teacher_pred = tf.stop_gradient(
            self.teacher(
                x,
                training=False,
            )
        )

        student_pred = self.student(
            x,
            training=training,
        )

        student_p_waste = tf.clip_by_value(
            student_pred[
                ...,
                WASTE_CLASS_IDX
            ],
            EPSILON,
            1.0 - EPSILON,
        )

        teacher_p_waste = tf.clip_by_value(
            teacher_pred[
                ...,
                WASTE_CLASS_IDX
            ],
            EPSILON,
            1.0 - EPSILON,
        )

        student_argmax_waste = tf.cast(
            tf.argmax(
                student_pred,
                axis=-1,
                output_type=tf.int32,
            ) == WASTE_CLASS_IDX,
            tf.float32,
        )

        teacher_argmax_waste = tf.cast(
            tf.argmax(
                teacher_pred,
                axis=-1,
                output_type=tf.int32,
            ) == WASTE_CLASS_IDX,
            tf.float32,
        )

        # -----------------------------------------------------------------
        # A. Ordinary 10-class supervision on original labelled data
        # -----------------------------------------------------------------

        per_pixel_ce = (
            tf.keras.losses.categorical_crossentropy(
                supervised_target,
                student_pred,
            )
        )

        supervised_count = tf.reduce_sum(
            supervised_mask
        )

        supervised_loss = (
            tf.reduce_sum(
                per_pixel_ce
                * supervised_mask
            )
            / tf.maximum(
                supervised_count,
                1.0,
            )
        )

        true_class = tf.argmax(
            supervised_target,
            axis=-1,
            output_type=tf.int32,
        )

        student_class = tf.argmax(
            student_pred,
            axis=-1,
            output_type=tf.int32,
        )

        supervised_accuracy = (
            tf.reduce_sum(
                tf.cast(
                    tf.equal(
                        true_class,
                        student_class,
                    ),
                    tf.float32,
                )
                * supervised_mask
            )
            / tf.maximum(
                supervised_count,
                1.0,
            )
        )

        # -----------------------------------------------------------------
        # B. Explicit preservation of confirmed genuine Waste
        # -----------------------------------------------------------------
        #
        # This counterbalances the hard-negative signal. The model is reminded
        # that the exact pixels labelled Waste in the original training data
        # must remain strongly recognisable as Waste.
        # -----------------------------------------------------------------

        positive_count = tf.reduce_sum(
            positive_waste_mask
        )

        positive_preservation_loss = (
            tf.reduce_sum(
                (
                    -tf.math.log(
                        student_p_waste
                    )
                )
                * positive_waste_mask
            )
            / tf.maximum(
                positive_count,
                1.0,
            )
        )

        student_positive_mean_p = (
            tf.reduce_sum(
                student_p_waste
                * positive_waste_mask
            )
            / tf.maximum(
                positive_count,
                1.0,
            )
        )

        teacher_positive_mean_p = (
            tf.reduce_sum(
                teacher_p_waste
                * positive_waste_mask
            )
            / tf.maximum(
                positive_count,
                1.0,
            )
        )

        student_positive_recall = (
            tf.reduce_sum(
                student_argmax_waste
                * positive_waste_mask
            )
            / tf.maximum(
                positive_count,
                1.0,
            )
        )

        teacher_positive_recall = (
            tf.reduce_sum(
                teacher_argmax_waste
                * positive_waste_mask
            )
            / tf.maximum(
                positive_count,
                1.0,
            )
        )

        # -----------------------------------------------------------------
        # C. Selective correction inside reviewed non-waste regions
        # -----------------------------------------------------------------
        #
        # The reviewed negative mask says only:
        #
        #       "Waste is absent here."
        #
        # We do not assign another class.
        #
        # Correction is restricted to pixels that actually fooled the frozen
        # baseline, and high-confidence baseline mistakes receive stronger
        # weights.
        # -----------------------------------------------------------------

        negative_count = tf.reduce_sum(
            negative_mask
        )

        teacher_high_waste = tf.cast(
            teacher_p_waste
            >= TEACHER_WASTE_PROBABILITY_TRIGGER,
            tf.float32,
        )

        teacher_error_trigger = tf.maximum(
            teacher_argmax_waste,
            teacher_high_waste,
        )

        targeted_negative_mask = (
            negative_mask
            * teacher_error_trigger
        )

        targeted_count = tf.reduce_sum(
            targeted_negative_mask
        )

        teacher_error_weight = (
            targeted_negative_mask
            * tf.stop_gradient(
                tf.pow(
                    teacher_p_waste,
                    TEACHER_ERROR_POWER,
                )
            )
        )

        error_weight_sum = tf.reduce_sum(
            teacher_error_weight
        )

        per_pixel_not_waste_loss = (
            -tf.math.log(
                1.0
                - student_p_waste
            )
        )

        selective_negative_loss = (
            tf.reduce_sum(
                per_pixel_not_waste_loss
                * teacher_error_weight
            )
            / tf.maximum(
                error_weight_sum,
                1.0,
            )
        )

        # -----------------------------------------------------------------
        # D. Total loss
        # -----------------------------------------------------------------

        total_loss = (
            supervised_loss
            + POSITIVE_PRESERVATION_LOSS_WEIGHT
            * positive_preservation_loss
            + SELECTIVE_NEGATIVE_LOSS_WEIGHT
            * selective_negative_loss
        )

        # -----------------------------------------------------------------
        # E. Diagnostics over all confirmed non-waste pixels
        # -----------------------------------------------------------------

        student_negative_mean_p = (
            tf.reduce_sum(
                student_p_waste
                * negative_mask
            )
            / tf.maximum(
                negative_count,
                1.0,
            )
        )

        teacher_negative_mean_p = (
            tf.reduce_sum(
                teacher_p_waste
                * negative_mask
            )
            / tf.maximum(
                negative_count,
                1.0,
            )
        )

        student_negative_argmax = (
            tf.reduce_sum(
                student_argmax_waste
                * negative_mask
            )
            / tf.maximum(
                negative_count,
                1.0,
            )
        )

        teacher_negative_argmax = (
            tf.reduce_sum(
                teacher_argmax_waste
                * negative_mask
            )
            / tf.maximum(
                negative_count,
                1.0,
            )
        )

        targeted_negative_fraction = (
            targeted_count
            / tf.maximum(
                negative_count,
                1.0,
            )
        )

        student_targeted_mean_p = (
            tf.reduce_sum(
                student_p_waste
                * targeted_negative_mask
            )
            / tf.maximum(
                targeted_count,
                1.0,
            )
        )

        teacher_targeted_mean_p = (
            tf.reduce_sum(
                teacher_p_waste
                * targeted_negative_mask
            )
            / tf.maximum(
                targeted_count,
                1.0,
            )
        )

        return {
            "total_loss":
                total_loss,

            "supervised_loss":
                supervised_loss,

            "positive_preservation_loss":
                positive_preservation_loss,

            "selective_negative_loss":
                selective_negative_loss,

            "supervised_accuracy":
                supervised_accuracy,

            "student_positive_mean_p":
                student_positive_mean_p,

            "teacher_positive_mean_p":
                teacher_positive_mean_p,

            "student_positive_recall":
                student_positive_recall,

            "teacher_positive_recall":
                teacher_positive_recall,

            "student_negative_mean_p":
                student_negative_mean_p,

            "teacher_negative_mean_p":
                teacher_negative_mean_p,

            "student_negative_argmax":
                student_negative_argmax,

            "teacher_negative_argmax":
                teacher_negative_argmax,

            "targeted_negative_fraction":
                targeted_negative_fraction,

            "student_targeted_mean_p":
                student_targeted_mean_p,

            "teacher_targeted_mean_p":
                teacher_targeted_mean_p,

            "supervised_count":
                supervised_count,

            "positive_count":
                positive_count,

            "negative_count":
                negative_count,

            "targeted_count":
                targeted_count,
        }

    def update_trackers(self, c):

        self.loss_tracker.update_state(
            c[
                "total_loss"
            ]
        )

        self.supervised_loss_tracker.update_state(
            c[
                "supervised_loss"
            ]
        )

        self.positive_preservation_loss_tracker.update_state(
            c[
                "positive_preservation_loss"
            ]
        )

        self.selective_negative_loss_tracker.update_state(
            c[
                "selective_negative_loss"
            ]
        )

        self.supervised_accuracy_tracker.update_state(
            c[
                "supervised_accuracy"
            ],
            sample_weight=c[
                "supervised_count"
            ],
        )

        self.student_positive_mean_p_tracker.update_state(
            c[
                "student_positive_mean_p"
            ],
            sample_weight=c[
                "positive_count"
            ],
        )

        self.teacher_positive_mean_p_tracker.update_state(
            c[
                "teacher_positive_mean_p"
            ],
            sample_weight=c[
                "positive_count"
            ],
        )

        self.student_positive_recall_tracker.update_state(
            c[
                "student_positive_recall"
            ],
            sample_weight=c[
                "positive_count"
            ],
        )

        self.teacher_positive_recall_tracker.update_state(
            c[
                "teacher_positive_recall"
            ],
            sample_weight=c[
                "positive_count"
            ],
        )

        self.student_negative_mean_p_tracker.update_state(
            c[
                "student_negative_mean_p"
            ],
            sample_weight=c[
                "negative_count"
            ],
        )

        self.teacher_negative_mean_p_tracker.update_state(
            c[
                "teacher_negative_mean_p"
            ],
            sample_weight=c[
                "negative_count"
            ],
        )

        self.student_negative_argmax_tracker.update_state(
            c[
                "student_negative_argmax"
            ],
            sample_weight=c[
                "negative_count"
            ],
        )

        self.teacher_negative_argmax_tracker.update_state(
            c[
                "teacher_negative_argmax"
            ],
            sample_weight=c[
                "negative_count"
            ],
        )

        self.targeted_negative_fraction_tracker.update_state(
            c[
                "targeted_negative_fraction"
            ],
            sample_weight=c[
                "negative_count"
            ],
        )

        self.student_targeted_mean_p_tracker.update_state(
            c[
                "student_targeted_mean_p"
            ],
            sample_weight=c[
                "targeted_count"
            ],
        )

        self.teacher_targeted_mean_p_tracker.update_state(
            c[
                "teacher_targeted_mean_p"
            ],
            sample_weight=c[
                "targeted_count"
            ],
        )

    def train_step(self, data):

        x, y_true = data

        with tf.GradientTape() as tape:

            c = self.compute_components(
                x,
                y_true,
                training=True,
            )

        grads = tape.gradient(
            c[
                "total_loss"
            ],
            self.student.trainable_variables,
        )

        grads_and_vars = [
            (grad, var)

            for grad, var in zip(
                grads,
                self.student.trainable_variables,
            )

            if grad is not None
        ]

        self.optimizer.apply_gradients(
            grads_and_vars
        )

        self.update_trackers(
            c
        )

        return {
            metric.name:
                metric.result()

            for metric in self.metrics
        }

    def test_step(self, data):

        x, y_true = data

        c = self.compute_components(
            x,
            y_true,
            training=False,
        )

        self.update_trackers(
            c
        )

        return {
            metric.name:
                metric.result()

            for metric in self.metrics
        }


trainer = SelectiveErrorCorrectionFineTuner(
    student=student,
    teacher=teacher,
    name="selective_error_correction_finetuner",
)

trainer.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE
    )
)

# =============================================================================
# 13. CALLBACKS
# =============================================================================

class LearningRateHistory(Callback):

    def __init__(self):
        super().__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):

        lr = float(
            tf.keras.backend.get_value(
                self.model.optimizer.learning_rate
            )
        )

        self.learning_rates.append(
            lr
        )

        print(
            f" - learning_rate: "
            f"{lr:.8g}"
        )


class PositivePreservingCheckpoint(Callback):
    """
    Save only validation models that improve the objective while preserving
    genuine Waste recognition relative to the frozen baseline.

    The baseline model itself is written to filepath before training, so if no
    fine-tuned epoch satisfies the constraints, the selected model remains the
    original baseline rather than a degraded fine-tuned model.
    """

    def __init__(
        self,
        filepath,
        max_positive_recall_drop,
    ):

        super().__init__()

        self.filepath = Path(
            filepath
        )

        self.max_positive_recall_drop = float(
            max_positive_recall_drop
        )

        self.best = np.inf
        self.best_epoch = 0
        self.best_logs = None

    def on_epoch_end(
        self,
        epoch,
        logs=None,
    ):

        logs = logs or {}

        required = [
            "val_loss",
            "val_student_positive_waste_argmax_recall",
            "val_teacher_positive_waste_argmax_recall",
            "val_student_negative_waste_argmax_rate",
            "val_teacher_negative_waste_argmax_rate",
        ]

        if any(
            key not in logs
            for key in required
        ):
            return

        val_loss = float(
            logs[
                "val_loss"
            ]
        )

        student_positive_recall = float(
            logs[
                "val_student_positive_waste_argmax_recall"
            ]
        )

        teacher_positive_recall = float(
            logs[
                "val_teacher_positive_waste_argmax_recall"
            ]
        )

        student_negative_rate = float(
            logs[
                "val_student_negative_waste_argmax_rate"
            ]
        )

        teacher_negative_rate = float(
            logs[
                "val_teacher_negative_waste_argmax_rate"
            ]
        )

        positive_ok = (
            student_positive_recall
            >= (
                teacher_positive_recall
                - self.max_positive_recall_drop
            )
        )

        negative_improved = (
            student_negative_rate
            < teacher_negative_rate
        )

        improved_loss = (
            val_loss
            < self.best
        )

        print(
            "\nCheckpoint guard:"
            f" positive_ok={positive_ok}"
            f" | negative_improved={negative_improved}"
            f" | val_loss_improved={improved_loss}"
        )

        print(
            "  Validation genuine-Waste recall:"
            f" teacher={teacher_positive_recall:.4f},"
            f" student={student_positive_recall:.4f}"
        )

        print(
            "  Validation false-positive Waste argmax rate:"
            f" teacher={teacher_negative_rate:.4f},"
            f" student={student_negative_rate:.4f}"
        )

        if (
            positive_ok
            and negative_improved
            and improved_loss
        ):

            self.best = val_loss
            self.best_epoch = (
                epoch
                + 1
            )

            self.best_logs = dict(
                logs
            )

            self.model.student.save(
                self.filepath
            )

            print(
                f"\nSaved SELECTED student at epoch "
                f"{self.best_epoch}:"
            )

            print(
                f"  val_loss={val_loss:.6f}"
            )

            print(
                f"  genuine-Waste recall="
                f"{student_positive_recall:.4f}"
            )

            print(
                f"  false-positive Waste rate="
                f"{student_negative_rate:.4f}"
            )

            print(
                f"  model={self.filepath}"
            )


class NegativeSelectionWriter(Callback):

    def __init__(
        self,
        generator,
        output_path,
    ):

        super().__init__()

        self.generator = (
            generator
        )

        self.output_path = Path(
            output_path
        )

    def on_epoch_end(
        self,
        epoch,
        logs=None,
    ):

        completed_epoch = (
            epoch
            + 1
        )

        df = pd.DataFrame(
            self.generator.selection_history
        )

        df = df[
            df[
                "epoch"
            ]
            <= completed_epoch
        ].copy()

        df.to_csv(
            self.output_path,
            index=False,
        )


lr_history = LearningRateHistory()

selection_history_path = (
    RESULTS_DIR
    / "negative_selection_history.csv"
)

selection_writer = NegativeSelectionWriter(
    train_gen,
    selection_history_path,
)

# Baseline is the safe fallback if no fine-tuned epoch preserves genuine Waste.
student.save(
    BEST_MODEL_PATH
)

print(
    f"\nBaseline fallback checkpoint written to:\n"
    f"{BEST_MODEL_PATH}"
)

best_checkpoint = PositivePreservingCheckpoint(
    BEST_MODEL_PATH,
    max_positive_recall_drop=MAX_ALLOWED_POSITIVE_RECALL_DROP,
)

early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    min_delta=MIN_DELTA,
    patience=EARLY_STOP_PATIENCE,
    restore_best_weights=False,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.5,
    patience=REDUCE_LR_PATIENCE,
    min_delta=MIN_DELTA,
    min_lr=MIN_LEARNING_RATE,
    verbose=1,
)

csv_logger = CSVLogger(
    RESULTS_DIR
    / "training_log.csv"
)

terminate_nan = TerminateOnNaN()

# =============================================================================
# 14. TRAIN
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING DESIGN")
print("=" * 80)

print(
    "\nOriginal fully labelled pixels:"
    "\n  ordinary 10-class categorical cross-entropy"
)

print(
    "\nConfirmed genuine Waste pixels:"
    "\n  explicit positive-preservation loss"
)

print(
    "\nReviewed non-waste regions:"
    "\n  only baseline Waste errors receive negative correction"
)

print(
    "\nStrength of each negative correction:"
    "\n  baseline P(Waste) ** "
    f"{TEACHER_ERROR_POWER}"
)

print(
    "\nAlternative non-waste class:"
    "\n  NOT specified"
)

print(
    "\nCheckpoint protection:"
    "\n  preserve validation genuine-Waste argmax recall within "
    f"{MAX_ALLOWED_POSITIVE_RECALL_DROP:.3f} of baseline"
)

print(
    "\nIndependent test AOI:"
    "\n  NOT USED"
)

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

history = trainer.fit(
    train_gen,
    validation_data=val_gen,
    epochs=MAX_EPOCHS,
    callbacks=[
        reduce_lr,
        early_stop,
        best_checkpoint,
        csv_logger,
        terminate_nan,
        lr_history,
        selection_writer,
    ],
    verbose=1,
)

# =============================================================================
# 15. SAVE RESULTS
# =============================================================================

epochs_completed = len(
    history.history[
        "loss"
    ]
)

history_df = pd.DataFrame(
    history.history
)

history_df.insert(
    0,
    "epoch",
    np.arange(
        1,
        len(
            history_df
        )
        + 1,
    ),
)

history_df[
    "learning_rate"
] = lr_history.learning_rates

history_path = (
    RESULTS_DIR
    / "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False,
)

# ---------------------------------------------------------------------
# Selected checkpoint
# ---------------------------------------------------------------------
#
# The selected checkpoint is the best validation-loss model that ALSO:
#   - preserves genuine labelled Waste recognition; and
#   - improves the false-positive Waste rate.
#
# BEST_MODEL_PATH already contains the baseline as a fallback.
# ---------------------------------------------------------------------

selected_epoch = int(
    best_checkpoint.best_epoch
)

selected_val_loss = (
    float(
        best_checkpoint.best
    )
    if np.isfinite(
        best_checkpoint.best
    )
    else np.nan
)

if selected_epoch == 0:

    print(
        "\nWARNING:"
        "\nNo fine-tuned epoch satisfied the positive-preservation and"
        "\nfalse-positive-improvement checkpoint constraints."
        "\nThe selected model therefore remains the original baseline."
    )

else:

    print(
        f"\nSelected fine-tuned epoch: "
        f"{selected_epoch}"
    )

    print(
        f"Selected validation loss: "
        f"{selected_val_loss:.8f}"
    )

# Reload the selected checkpoint, not the final training epoch.
selected_student = load_model(
    BEST_MODEL_PATH,
    compile=False,
)

selected_student.save(
    FINAL_MODEL_PATH
)

print(
    f"\nSelected model copied to final path:\n"
    f"{FINAL_MODEL_PATH}"
)

# ---------------------------------------------------------------------
# Selected-epoch metrics
# ---------------------------------------------------------------------

best_epoch_metrics_path = (
    RESULTS_DIR
    / "selected_epoch_metrics.csv"
)

if selected_epoch > 0:

    selected_metrics_df = history_df[
        history_df[
            "epoch"
        ]
        == selected_epoch
    ].copy()

else:

    selected_metrics_df = pd.DataFrame(
        [
            {
                "epoch":
                    0,

                "selection_status":
                    "baseline_fallback",
            }
        ]
    )

selected_metrics_df.to_csv(
    best_epoch_metrics_path,
    index=False,
)

# ---------------------------------------------------------------------
# Negative-object coverage through selected epoch
# ---------------------------------------------------------------------

selection_df = pd.DataFrame(
    train_gen.selection_history
)

selection_df = selection_df[
    selection_df[
        "epoch"
    ]
    <= epochs_completed
].copy()

selection_df.to_csv(
    selection_history_path,
    index=False,
)

selection_best = selection_df[
    selection_df[
        "epoch"
    ]
    <= selected_epoch
].copy()

selection_best.to_csv(
    RESULTS_DIR
    / "negative_selection_through_selected_epoch.csv",
    index=False,
)

coverage_records = []

for source_name in [
    "stage1_rejected",
    "stage2_not_waste",
]:

    available = set(
        negative_train_df.loc[
            negative_train_df[
                "negative_source"
            ]
            == source_name,
            "sample_id",
        ].astype(
            str
        )
    )

    seen = set(
        selection_best.loc[
            selection_best[
                "negative_source"
            ]
            == source_name,
            "sample_id",
        ].astype(
            str
        )
    )

    coverage_records.append(
        {
            "negative_source":
                source_name,

            "available_training_objects":
                len(
                    available
                ),

            "seen_through_selected_epoch":
                len(
                    seen
                ),

            "coverage_percent_through_selected_epoch":
                (
                    100.0
                    * len(
                        seen
                    )
                    / len(
                        available
                    )
                    if available
                    else 0.0
                ),
        }
    )

coverage_df = pd.DataFrame(
    coverage_records
)

coverage_path = (
    RESULTS_DIR
    / "negative_object_coverage_through_selected_epoch.csv"
)

coverage_df.to_csv(
    coverage_path,
    index=False,
)

# =============================================================================
# 16. PLOTS
# =============================================================================

plt.figure(
    figsize=(
        8,
        5,
    )
)

plt.plot(
    history_df[
        "epoch"
    ],
    history_df[
        "loss"
    ],
    label="Training loss",
)

plt.plot(
    history_df[
        "epoch"
    ],
    history_df[
        "val_loss"
    ],
    label="Validation loss",
)

if selected_epoch > 0:

    plt.axvline(
        selected_epoch,
        linestyle="--",
        label=(
            f"Selected epoch = "
            f"{selected_epoch}"
        ),
    )

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Selective error-correction fine-tuning"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "training_validation_loss.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ---------------------------------------------------------------------
# Validation behaviour: genuine Waste and confirmed non-waste
# ---------------------------------------------------------------------

diagnostic_columns = [
    "val_student_positive_waste_argmax_recall",
    "val_teacher_positive_waste_argmax_recall",
    "val_student_negative_waste_argmax_rate",
    "val_teacher_negative_waste_argmax_rate",
]

available_diagnostics = [
    column

    for column in diagnostic_columns

    if column in history_df.columns
]

if available_diagnostics:

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    for column in available_diagnostics:

        plt.plot(
            history_df[
                "epoch"
            ],
            history_df[
                column
            ],
            label=column,
        )

    if selected_epoch > 0:

        plt.axvline(
            selected_epoch,
            linestyle="--",
            label=(
                f"Selected epoch = "
                f"{selected_epoch}"
            ),
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Rate"
    )

    plt.title(
        "Fixed-validation Waste preservation and false-positive correction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "validation_positive_negative_diagnostics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

# =============================================================================
# 17. SUMMARY
# =============================================================================

summary_path = (
    RESULTS_DIR
    / "fine_tuning_summary.txt"
)

with open(
    summary_path,
    "w",
) as f:

    f.write(
        "SELECTIVE ERROR-CORRECTION FINE-TUNING\n"
    )

    f.write(
        "=" * 72
        + "\n\n"
    )

    f.write(
        f"Original baseline / teacher:\n"
        f"{BASE_MODEL_PATH}\n\n"
    )

    f.write(
        f"Selected final model:\n"
        f"{FINAL_MODEL_PATH}\n\n"
    )

    f.write(
        f"Selected checkpoint:\n"
        f"{BEST_MODEL_PATH}\n\n"
    )

    f.write(
        "Training logic:\n"
        "  1. Original labelled pixels -> 10-class categorical cross-entropy\n"
        "  2. Original labelled Waste pixels -> explicit Waste-preservation loss\n"
        "  3. Reviewed non-waste pixels -> correction only where the frozen "
        "baseline predicted Waste/high Waste probability\n"
        "  4. Negative correction strength -> P_teacher(Waste)^ERROR_POWER\n"
        "  5. No alternative non-waste class is imposed\n\n"
    )

    f.write(
        f"Teacher Waste-probability trigger: "
        f"{TEACHER_WASTE_PROBABILITY_TRIGGER}\n"
    )

    f.write(
        f"Teacher error-power: "
        f"{TEACHER_ERROR_POWER}\n"
    )

    f.write(
        f"Positive-preservation loss weight: "
        f"{POSITIVE_PRESERVATION_LOSS_WEIGHT}\n"
    )

    f.write(
        f"Selective-negative loss weight: "
        f"{SELECTIVE_NEGATIVE_LOSS_WEIGHT}\n"
    )

    f.write(
        f"Maximum allowed validation genuine-Waste recall drop: "
        f"{MAX_ALLOWED_POSITIVE_RECALL_DROP}\n\n"
    )

    f.write(
        f"Epochs completed: "
        f"{epochs_completed}\n"
    )

    f.write(
        f"Selected fine-tuned epoch: "
        f"{selected_epoch}\n"
    )

    f.write(
        f"Selected validation loss: "
        f"{selected_val_loss}\n\n"
    )

    if selected_epoch == 0:

        f.write(
            "Selection status: original baseline retained because no "
            "fine-tuned epoch satisfied all checkpoint constraints.\n\n"
        )

    else:

        f.write(
            "Selection status: fine-tuned model accepted under the "
            "positive-preservation checkpoint rule.\n\n"
        )

    f.write(
        "Hard-negative coverage through selected epoch:\n"
    )

    for _, row in coverage_df.iterrows():

        f.write(
            f"  {row['negative_source']}: "
            f"{int(row['seen_through_selected_epoch'])} / "
            f"{int(row['available_training_objects'])} "
            f"({row['coverage_percent_through_selected_epoch']:.2f}%)\n"
        )

    f.write(
        "\nThe locked independent test AOI was not used for training, "
        "validation, checkpointing, or early stopping.\n"
    )

# =============================================================================
# 18. FINAL REPORT
# =============================================================================

print("\n" + "=" * 80)
print("SELECTIVE ERROR-CORRECTION FINE-TUNING COMPLETE")
print("=" * 80)

print(
    f"\nEpochs completed: "
    f"{epochs_completed}"
)

print(
    f"Selected epoch: "
    f"{selected_epoch}"
)

print(
    f"Selected validation loss: "
    f"{selected_val_loss}"
)

print(
    "\nNegative-object coverage through selected epoch:"
)

print(
    coverage_df.to_string(
        index=False
    )
)

print(
    f"\nSelected fine-tuned model:\n"
    f"{FINAL_MODEL_PATH}"
)

print(
    f"\nSelected checkpoint:\n"
    f"{BEST_MODEL_PATH}"
)

print(
    f"\nTraining history:\n"
    f"{history_path}"
)

print(
    f"\nSelected-epoch metrics:\n"
    f"{best_epoch_metrics_path}"
)

print(
    f"\nSummary:\n"
    f"{summary_path}"
)

print("\n" + "=" * 80)
print("NEXT STEP")
print("=" * 80)

print(
    "\nEvaluate at the fixed 0.50 threshold on the SAME locked independent "
    "test area:"
)

print(
    "\n  1. Original March baseline"
)

print(
    "  2. Selective error-correction model"
)

print(
    "\nPrimary question:"
    "\n  Did false-positive objects fall substantially while retaining the "
    "baseline's genuine Waste detections?"
)

print("\nDone.")
