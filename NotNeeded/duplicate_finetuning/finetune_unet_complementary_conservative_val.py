# =============================================================================
# CONSERVATIVE COMPLEMENTARY HARD-NEGATIVE FINE-TUNING
# WITH FIXED INTERNAL VALIDATION
# =============================================================================
#
# PURPOSE
# -------
# Refine the existing March 10-class U-Net more conservatively after the first
# complementary-negative experiment reduced false positives too aggressively.
#
# Main changes:
#   - complementary negative-loss weight reduced from 1.0 to 0.20
#   - 100 Stage-1 rejected objects per epoch
#   - 100 Stage-2 confirmed not-waste objects per epoch
#   - fixed internal validation set, excluded from training
#   - checkpointing / LR reduction / early stopping based on val_loss
#
# The independent test AOI is NOT used here. The complementary-negative
# library was already created with that AOI excluded.
# =============================================================================

# =============================================================================
# 0. ENVIRONMENT
# =============================================================================

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path
import random
import math

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
    ModelCheckpoint,
    CSVLogger,
    TerminateOnNaN,
    Callback,
)

# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

SEED = 20260919
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
# 3. PATHS
# =============================================================================

BASE_PATH = (
    Path.home()
    / "Documents"
    / "GitHub"
    / "mapping-waste-urban-africa"
)

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
    / "complementary_hard_negative_finetuning_conservative"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_complementary_conservative_checkpoint.keras"
)

FINAL_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_complementary_conservative_finetuned.keras"
)

# =============================================================================
# 4. MODEL / IMAGE SETTINGS
# =============================================================================

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 8
N_CLASSES = 10
WASTE_RAW_LABEL = 1
WASTE_CLASS_IDX = 0

LABEL_LIST = list(range(1, 11))
VALID_RAW_LABEL_SET = set(LABEL_LIST)
LABEL_MAP = {
    raw_label: model_index
    for model_index, raw_label in enumerate(LABEL_LIST)
}

# =============================================================================
# 5. CONSERVATIVE FINE-TUNING SETTINGS
# =============================================================================

MAX_EPOCHS = 300
INITIAL_LEARNING_RATE = 1e-5
MIN_LEARNING_RATE = 1e-7
EARLY_STOP_PATIENCE = 25
REDUCE_LR_PATIENCE = 8
MIN_DELTA = 1e-4

# Main correction from the first complementary run.
NEGATIVE_LOSS_WEIGHT = 0.20

# Fewer negative objects are shown per training epoch.
N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH = 100
N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH = 100

# Fixed internal validation. These examples are never used for training.
ORIGINAL_VALIDATION_FRACTION = 0.10
N_STAGE1_VALIDATION_OBJECTS = 100
N_STAGE2_VALIDATION_OBJECTS = 100

EPSILON = 1e-7

# =============================================================================
# 6. INPUT CHECKS
# =============================================================================

required_paths = [
    BASE_MODEL_PATH,
    ORIGINAL_IMAGE_DIR,
    ORIGINAL_MASK_DIR,
    NEGATIVE_IMAGE_DIR,
    NEGATIVE_MASK_DIR,
    NEGATIVE_MANIFEST_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(f"\nRequired input not found:\n{path}")

print("\n" + "=" * 80)
print("CONSERVATIVE COMPLEMENTARY HARD-NEGATIVE FINE-TUNING")
print("=" * 80)
print(f"\nStarting model:\n{BASE_MODEL_PATH}")
print(f"\nNegative-loss weight: {NEGATIVE_LOSS_WEIGHT}")
print(f"Stage-1 negatives / epoch: {N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH}")
print(f"Stage-2 negatives / epoch: {N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH}")

# =============================================================================
# 7. UTILITIES
# =============================================================================

def encode_raw_mask(mask):
    mapped = np.full(mask.shape, -1, dtype=np.int32)
    for raw_label, model_idx in LABEL_MAP.items():
        mapped[mask == raw_label] = model_idx
    if np.any(mapped < 0):
        bad = np.unique(mask[mapped < 0])
        raise ValueError(
            "Original mask contains values outside 1-10: "
            f"{bad.tolist()}"
        )
    return mapped


def validate_negative_mask(mask, mask_path):
    values = set(int(value) for value in np.unique(mask))
    allowed = {0, 1, 255}
    invalid = sorted(values - allowed)
    if invalid:
        raise ValueError(
            "\nInvalid complementary mask.\n"
            f"Mask:\n{mask_path}\n"
            f"Observed values: {sorted(values)}\n"
            f"Invalid values: {invalid}"
        )
    binary = (mask > 0).astype(np.float32)
    if np.sum(binary) == 0:
        raise ValueError(
            "\nComplementary mask contains no confirmed negative pixels:\n"
            f"{mask_path}"
        )
    return binary


def read_rgb(path):
    with rasterio.open(path) as src:
        image = (
            src.read([1, 2, 3])
            .transpose(1, 2, 0)
            .astype(np.float32)
        )
    image = resize(
        image,
        (IMG_HEIGHT, IMG_WIDTH),
        preserve_range=True,
    )
    return (image / 255.0).astype(np.float32)


def read_original_mask(path):
    with rasterio.open(path) as src:
        mask = src.read(1)
    mask = resize(
        mask,
        (IMG_HEIGHT, IMG_WIDTH),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return np.rint(mask).astype(np.int32)


def read_negative_mask(path):
    with rasterio.open(path) as src:
        mask = src.read(1)
    mask = resize(
        mask,
        (IMG_HEIGHT, IMG_WIDTH),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    mask = np.rint(mask).astype(np.int32)
    return validate_negative_mask(mask, path)


def augment_pair(image, target_mask):
    if random.random() < 0.5:
        image = np.flipud(image)
        target_mask = np.flipud(target_mask)

    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        image = rotate(
            image,
            angle,
            reshape=False,
            order=1,
            mode="reflect",
        )
        target_mask = rotate(
            target_mask,
            angle,
            reshape=False,
            order=0,
            mode="nearest",
        )

    return image, target_mask

# =============================================================================
# 8. AUDIT ORIGINAL FULLY LABELLED DATA
# =============================================================================

print("\n" + "=" * 80)
print("AUDITING ORIGINAL FULLY LABELLED DATA")
print("=" * 80)

original_images = {
    path.name: path
    for path in ORIGINAL_IMAGE_DIR.glob("*.tif")
    if not path.name.endswith(".aux.xml")
}

original_masks = {
    path.name: path
    for path in ORIGINAL_MASK_DIR.glob("*.tif")
    if not path.name.endswith(".aux.xml")
}

common_original = sorted(
    set(original_images.keys()).intersection(original_masks.keys())
)

if not common_original:
    raise FileNotFoundError("No matching original image-mask pairs found.")

all_original_records = []
excluded_original_records = []

for filename in common_original:
    image_path = original_images[filename]
    mask_path = original_masks[filename]

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    observed = set(int(value) for value in np.unique(mask))
    invalid = sorted(observed - VALID_RAW_LABEL_SET)

    if invalid:
        excluded_original_records.append(
            {
                "filename": filename,
                "invalid_labels": ",".join(str(value) for value in invalid),
            }
        )
        continue

    all_original_records.append(
        {
            "image_path": image_path,
            "mask_path": mask_path,
            "negative_mask_path": None,
            "source": "original",
            "negative_source": None,
            "sample_id": f"original::{filename}",
            "tile_id": filename,
        }
    )

print(f"Clean original pairs: {len(all_original_records):,}")
print(f"Excluded original pairs: {len(excluded_original_records):,}")

pd.DataFrame(excluded_original_records).to_csv(
    RESULTS_DIR / "excluded_original_masks.csv",
    index=False,
)

# =============================================================================
# 9. FIXED ORIGINAL TRAIN / VALIDATION SPLIT
# =============================================================================

rng_original = np.random.default_rng(SEED)
original_indices = np.arange(len(all_original_records))
rng_original.shuffle(original_indices)

n_original_validation = max(
    1,
    int(round(len(all_original_records) * ORIGINAL_VALIDATION_FRACTION)),
)

validation_original_indices = set(
    original_indices[:n_original_validation].tolist()
)

original_train_records = [
    record
    for index, record in enumerate(all_original_records)
    if index not in validation_original_indices
]

original_validation_records = [
    record
    for index, record in enumerate(all_original_records)
    if index in validation_original_indices
]

print(f"Original training replay pairs: {len(original_train_records):,}")
print(f"Original fixed validation pairs: {len(original_validation_records):,}")

# =============================================================================
# 10. LOAD COMPLEMENTARY NEGATIVE LIBRARY
# =============================================================================

manifest = pd.read_csv(NEGATIVE_MANIFEST_PATH, dtype=str)

required_manifest_columns = [
    "tile_id",
    "image_file",
    "negative_mask_file",
]

missing_manifest_columns = [
    column
    for column in required_manifest_columns
    if column not in manifest.columns
]

if missing_manifest_columns:
    raise ValueError(
        "\nManifest is missing required columns:\n"
        f"{missing_manifest_columns}"
    )

if "focal_candidate_id" in manifest.columns:
    manifest["object_id"] = manifest["focal_candidate_id"].astype(str)
elif "sample_id" in manifest.columns:
    manifest["object_id"] = manifest["sample_id"].astype(str)
else:
    raise ValueError("Manifest requires focal_candidate_id or sample_id.")

if "focal_source" in manifest.columns:
    manifest["negative_source"] = manifest["focal_source"].astype(str)
elif "negative_source" in manifest.columns:
    manifest["negative_source"] = manifest["negative_source"].astype(str)
else:
    raise ValueError("Manifest requires focal_source or negative_source.")

allowed_sources = {"stage1_rejected", "stage2_not_waste"}
manifest = manifest[
    manifest["negative_source"].isin(allowed_sources)
].copy()

negative_records = []

for _, row in manifest.iterrows():
    image_path = NEGATIVE_IMAGE_DIR / str(row["image_file"])
    negative_mask_path = NEGATIVE_MASK_DIR / str(row["negative_mask_file"])

    if not image_path.exists():
        raise FileNotFoundError(f"\nMissing hard-negative image:\n{image_path}")
    if not negative_mask_path.exists():
        raise FileNotFoundError(f"\nMissing hard-negative mask:\n{negative_mask_path}")

    negative_records.append(
        {
            "image_path": image_path,
            "mask_path": None,
            "negative_mask_path": negative_mask_path,
            "source": "hard_negative",
            "negative_source": str(row["negative_source"]),
            "sample_id": str(row["object_id"]),
            "tile_id": str(row["tile_id"]),
        }
    )

negative_library_df = pd.DataFrame(negative_records)

print("\n" + "=" * 80)
print("NEGATIVE LIBRARY")
print("=" * 80)

for source_name in ["stage1_rejected", "stage2_not_waste"]:
    subset = negative_library_df[
        negative_library_df["negative_source"] == source_name
    ]
    print(
        f"{source_name}: {len(subset):,} chips | "
        f"{subset['sample_id'].nunique():,} unique objects"
    )

# =============================================================================
# 11. FIXED OBJECT-LEVEL NEGATIVE VALIDATION SPLIT
# =============================================================================

def split_negative_source(library_df, source_name, n_validation_objects, seed):
    subset = library_df[
        library_df["negative_source"] == source_name
    ].copy()

    unique_objects = np.array(
        sorted(subset["sample_id"].astype(str).unique()),
        dtype=object,
    )

    if len(unique_objects) <= n_validation_objects:
        raise ValueError(
            f"Not enough {source_name} objects for requested validation size."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_objects)
    validation_ids = set(unique_objects[:n_validation_objects].tolist())

    training_df = subset[
        ~subset["sample_id"].isin(validation_ids)
    ].copy()

    validation_df = subset[
        subset["sample_id"].isin(validation_ids)
    ].copy()

    return training_df, validation_df, validation_ids


stage1_train_df, stage1_validation_df, stage1_validation_ids = split_negative_source(
    negative_library_df,
    "stage1_rejected",
    N_STAGE1_VALIDATION_OBJECTS,
    SEED + 101,
)

stage2_train_df, stage2_validation_df, stage2_validation_ids = split_negative_source(
    negative_library_df,
    "stage2_not_waste",
    N_STAGE2_VALIDATION_OBJECTS,
    SEED + 202,
)

negative_train_df = pd.concat(
    [stage1_train_df, stage2_train_df],
    ignore_index=True,
)


def choose_one_validation_chip_per_object(validation_df, seed):
    rng = np.random.default_rng(seed)
    selected_records = []

    for object_id, group in validation_df.groupby("sample_id"):
        row = group.iloc[int(rng.integers(0, len(group)))]
        selected_records.append(
            {
                "image_path": row["image_path"],
                "mask_path": None,
                "negative_mask_path": row["negative_mask_path"],
                "source": "hard_negative",
                "negative_source": row["negative_source"],
                "sample_id": str(row["sample_id"]),
                "tile_id": str(row["tile_id"]),
            }
        )

    return selected_records


negative_validation_records = (
    choose_one_validation_chip_per_object(stage1_validation_df, SEED + 303)
    + choose_one_validation_chip_per_object(stage2_validation_df, SEED + 404)
)

validation_records = (
    list(original_validation_records)
    + negative_validation_records
)

# Fixed deterministic shuffle so validation batches mix supervised and
# complementary examples but remain identical every epoch.
rng_val_order = np.random.default_rng(SEED + 505)
validation_order = np.arange(len(validation_records))
rng_val_order.shuffle(validation_order)
validation_records = [validation_records[i] for i in validation_order]

print("\n" + "=" * 80)
print("FIXED INTERNAL VALIDATION")
print("=" * 80)
print(f"Original validation pairs: {len(original_validation_records):,}")
print(f"Stage-1 negative validation objects: {N_STAGE1_VALIDATION_OBJECTS:,}")
print(f"Stage-2 negative validation objects: {N_STAGE2_VALIDATION_OBJECTS:,}")
print(f"Total fixed validation records: {len(validation_records):,}")

validation_split_records = []
for record in validation_records:
    validation_split_records.append(
        {
            "source": (
                record["negative_source"]
                if record["source"] == "hard_negative"
                else "original"
            ),
            "sample_id": record["sample_id"],
            "tile_id": record["tile_id"],
        }
    )

pd.DataFrame(validation_split_records).to_csv(
    RESULTS_DIR / "fixed_internal_validation_records.csv",
    index=False,
)

# =============================================================================
# 12. CYCLING OBJECT SAMPLER
# =============================================================================

class CyclingObjectSampler:
    def __init__(self, records_df, source_name, seed):
        self.df = records_df[
            records_df["negative_source"] == source_name
        ].copy()
        self.source_name = source_name

        if len(self.df) == 0:
            raise ValueError(f"No training records found for {source_name}.")

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
        selected_ids = []

        while len(selected_ids) < n_objects:
            remaining = len(self.order) - self.position
            needed = n_objects - len(selected_ids)
            take = min(remaining, needed)

            if take > 0:
                selected_ids.extend(
                    self.order[self.position:self.position + take].tolist()
                )
                self.position += take

            if len(selected_ids) < n_objects:
                self._reshuffle()

        records = []

        for object_id in selected_ids:
            candidate_indices = self.object_to_indices[str(object_id)]
            selected_index = int(self.rng.choice(candidate_indices))
            row = self.df.loc[selected_index]

            records.append(
                {
                    "image_path": row["image_path"],
                    "mask_path": None,
                    "negative_mask_path": row["negative_mask_path"],
                    "source": "hard_negative",
                    "negative_source": row["negative_source"],
                    "sample_id": str(row["sample_id"]),
                    "tile_id": str(row["tile_id"]),
                }
            )

        return records

# =============================================================================
# 13. TARGET CONSTRUCTION
# =============================================================================

def make_training_target(record, augment):
    image = read_rgb(record["image_path"])

    if record["source"] == "original":
        mask = read_original_mask(record["mask_path"])

        if augment:
            image, mask = augment_pair(image, mask)

        mask = np.rint(mask).astype(np.int32)
        mapped = encode_raw_mask(mask)

        supervised_target = to_categorical(
            mapped,
            num_classes=N_CLASSES,
        ).astype(np.float32)

        complementary_negative = np.zeros(
            (IMG_HEIGHT, IMG_WIDTH, 1),
            dtype=np.float32,
        )

    elif record["source"] == "hard_negative":
        negative_mask = read_negative_mask(record["negative_mask_path"])

        if augment:
            image, negative_mask = augment_pair(image, negative_mask)

        negative_mask = (negative_mask > 0.5).astype(np.float32)

        supervised_target = np.zeros(
            (IMG_HEIGHT, IMG_WIDTH, N_CLASSES),
            dtype=np.float32,
        )

        complementary_negative = negative_mask[..., np.newaxis]

    else:
        raise ValueError(f"Unknown record source: {record['source']}")

    target = np.concatenate(
        [supervised_target, complementary_negative],
        axis=-1,
    )

    return image.astype(np.float32), target.astype(np.float32)

# =============================================================================
# 14. ROTATING TRAINING GENERATOR
# =============================================================================

class RotatingTrainingGenerator(Sequence):
    def __init__(
        self,
        original_records,
        negative_train_df,
        batch_size=BATCH_SIZE,
        n_stage1=N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH,
        n_stage2=N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH,
        seed=SEED,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.original_records = list(original_records)
        self.negative_train_df = negative_train_df.copy()
        self.batch_size = batch_size
        self.n_stage1 = n_stage1
        self.n_stage2 = n_stage2
        self.seed = seed
        self.epoch_number = 0

        self.stage1_sampler = CyclingObjectSampler(
            self.negative_train_df,
            "stage1_rejected",
            seed + 1001,
        )

        self.stage2_sampler = CyclingObjectSampler(
            self.negative_train_df,
            "stage2_not_waste",
            seed + 2001,
        )

        self.current_records = []
        self.indices = np.array([], dtype=int)
        self.selection_history = []
        self._prepare_epoch()

    def _prepare_epoch(self):
        self.epoch_number += 1

        stage1_records = self.stage1_sampler.sample(self.n_stage1)
        stage2_records = self.stage2_sampler.sample(self.n_stage2)
        negative_records = stage1_records + stage2_records

        for record in negative_records:
            self.selection_history.append(
                {
                    "epoch": self.epoch_number,
                    "negative_source": record["negative_source"],
                    "sample_id": record["sample_id"],
                    "tile_id": record["tile_id"],
                }
            )

        self.current_records = list(self.original_records) + negative_records
        self.indices = np.arange(len(self.current_records))

        rng = np.random.default_rng(self.seed + self.epoch_number)
        rng.shuffle(self.indices)

        if self.epoch_number == 1 or self.epoch_number % 10 == 0:
            print(
                "\n"
                f"[Epoch pool {self.epoch_number}] "
                f"original={len(self.original_records):,}, "
                f"stage1_neg={len(stage1_records):,}, "
                f"stage2_neg={len(stage2_records):,}, "
                f"total={len(self.current_records):,}"
            )

    def __len__(self):
        return int(math.ceil(len(self.current_records) / self.batch_size))

    def on_epoch_end(self):
        self._prepare_epoch()

    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size:
            (idx + 1) * self.batch_size
        ]

        images = []
        targets = []

        for record_index in batch_indices:
            record = self.current_records[record_index]
            image, target = make_training_target(record, augment=True)
            images.append(image)
            targets.append(target)

        return (
            np.asarray(images, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
        )

# =============================================================================
# 15. FIXED VALIDATION GENERATOR
# =============================================================================

class FixedValidationGenerator(Sequence):
    def __init__(self, records, batch_size=BATCH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.records = list(records)
        self.batch_size = batch_size

    def __len__(self):
        return int(math.ceil(len(self.records) / self.batch_size))

    def __getitem__(self, idx):
        batch_records = self.records[
            idx * self.batch_size:
            (idx + 1) * self.batch_size
        ]

        images = []
        targets = []

        for record in batch_records:
            image, target = make_training_target(record, augment=False)
            images.append(image)
            targets.append(target)

        return (
            np.asarray(images, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
        )


train_gen = RotatingTrainingGenerator(
    original_records=original_train_records,
    negative_train_df=negative_train_df,
    batch_size=BATCH_SIZE,
    n_stage1=N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH,
    n_stage2=N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH,
    seed=SEED,
)

val_gen = FixedValidationGenerator(
    validation_records,
    batch_size=BATCH_SIZE,
)

print("\n" + "=" * 80)
print("TRAIN / VALIDATION COMPOSITION")
print("=" * 80)
print(f"Original training replay records: {len(original_train_records):,}")
print(f"Rotating Stage-1 negatives / epoch: {N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH:,}")
print(f"Rotating Stage-2 negatives / epoch: {N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH:,}")
print(f"Training records / epoch: {len(train_gen.current_records):,}")
print(f"Fixed validation records: {len(validation_records):,}")
print(f"Training batches / epoch: {len(train_gen):,}")
print(f"Validation batches / epoch: {len(val_gen):,}")

# =============================================================================
# 16. MIXED LOSS
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="WasteFineTuning")
def complementary_mixed_loss(y_true, y_pred):
    supervised_target = y_true[..., :N_CLASSES]
    negative_mask = y_true[..., N_CLASSES]

    supervised_mask = tf.cast(
        tf.reduce_sum(supervised_target, axis=-1) > 0.5,
        tf.float32,
    )

    per_pixel_ce = tf.keras.losses.categorical_crossentropy(
        supervised_target,
        y_pred,
    )

    supervised_count = tf.reduce_sum(supervised_mask)

    supervised_loss = (
        tf.reduce_sum(per_pixel_ce * supervised_mask)
        / tf.maximum(supervised_count, 1.0)
    )

    p_waste = tf.clip_by_value(
        y_pred[..., WASTE_CLASS_IDX],
        EPSILON,
        1.0 - EPSILON,
    )

    per_pixel_negative_loss = -tf.math.log(1.0 - p_waste)
    negative_count = tf.reduce_sum(negative_mask)

    negative_loss = (
        tf.reduce_sum(per_pixel_negative_loss * negative_mask)
        / tf.maximum(negative_count, 1.0)
    )

    return supervised_loss + NEGATIVE_LOSS_WEIGHT * negative_loss

# =============================================================================
# 17. METRICS
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="WasteFineTuning")
class SupervisedPixelAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="supervised_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        target = y_true[..., :N_CLASSES]
        mask = tf.cast(
            tf.reduce_sum(target, axis=-1) > 0.5,
            tf.float32,
        )
        true_class = tf.argmax(target, axis=-1, output_type=tf.int32)
        pred_class = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        correct = tf.cast(tf.equal(true_class, pred_class), tf.float32)
        self.correct.assign_add(tf.reduce_sum(correct * mask))
        self.count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.correct / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.correct.assign(0.0)
        self.count.assign(0.0)


@tf.keras.utils.register_keras_serializable(package="WasteFineTuning")
class NegativeMeanWasteProbability(tf.keras.metrics.Metric):
    def __init__(self, name="negative_mean_waste_probability", **kwargs):
        super().__init__(name=name, **kwargs)
        self.probability_sum = self.add_weight(
            name="probability_sum",
            initializer="zeros",
        )
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        negative_mask = y_true[..., N_CLASSES]
        p_waste = y_pred[..., WASTE_CLASS_IDX]
        self.probability_sum.assign_add(
            tf.reduce_sum(p_waste * negative_mask)
        )
        self.count.assign_add(tf.reduce_sum(negative_mask))

    def result(self):
        return self.probability_sum / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.probability_sum.assign(0.0)
        self.count.assign(0.0)


@tf.keras.utils.register_keras_serializable(package="WasteFineTuning")
class NegativeWasteArgmaxRate(tf.keras.metrics.Metric):
    def __init__(self, name="negative_waste_argmax_rate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.waste_predictions = self.add_weight(
            name="waste_predictions",
            initializer="zeros",
        )
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        negative_mask = y_true[..., N_CLASSES]
        predicted_waste = tf.cast(
            tf.argmax(y_pred, axis=-1) == WASTE_CLASS_IDX,
            tf.float32,
        )
        self.waste_predictions.assign_add(
            tf.reduce_sum(predicted_waste * negative_mask)
        )
        self.count.assign_add(tf.reduce_sum(negative_mask))

    def result(self):
        return self.waste_predictions / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.waste_predictions.assign(0.0)
        self.count.assign(0.0)

# =============================================================================
# 18. LOAD EXISTING MODEL
# =============================================================================

print("\n" + "=" * 80)
print("LOADING EXISTING U-NET")
print("=" * 80)

tf.keras.backend.clear_session()
model = load_model(BASE_MODEL_PATH, compile=False)

if tuple(model.input_shape[1:]) != (IMG_HEIGHT, IMG_WIDTH, 3):
    raise ValueError(f"Unexpected model input shape: {model.input_shape}")

if int(model.output_shape[-1]) != N_CLASSES:
    raise ValueError(f"Unexpected model output shape: {model.output_shape}")

for layer in model.layers:
    layer.trainable = True

print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# =============================================================================
# 19. COMPILE
# =============================================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=INITIAL_LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss=complementary_mixed_loss,
    metrics=[
        SupervisedPixelAccuracy(),
        NegativeMeanWasteProbability(),
        NegativeWasteArgmaxRate(),
    ],
)

# =============================================================================
# 20. CALLBACKS
# =============================================================================

class LearningRateHistory(Callback):
    def __init__(self):
        super().__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        learning_rate = float(
            tf.keras.backend.get_value(
                self.model.optimizer.learning_rate
            )
        )
        self.learning_rates.append(learning_rate)
        print(f" - learning_rate: {learning_rate:.8g}")


lr_history = LearningRateHistory()

early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    min_delta=MIN_DELTA,
    patience=EARLY_STOP_PATIENCE,
    restore_best_weights=True,
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

checkpoint = ModelCheckpoint(
    filepath=str(CHECKPOINT_MODEL_PATH),
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)

csv_log_path = RESULTS_DIR / "training_log.csv"
csv_logger = CSVLogger(csv_log_path)
terminate_nan = TerminateOnNaN()

# =============================================================================
# 21. TRAIN
# =============================================================================

print("\n" + "=" * 80)
print("STARTING CONSERVATIVE FINE-TUNING")
print("=" * 80)
print(f"\nNegative-loss weight: {NEGATIVE_LOSS_WEIGHT}")
print("Checkpoint / early stopping criterion: fixed internal validation loss")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=MAX_EPOCHS,
    callbacks=[
        reduce_lr,
        early_stop,
        checkpoint,
        csv_logger,
        terminate_nan,
        lr_history,
    ],
    verbose=1,
)

# =============================================================================
# 22. TRAINING RESULT
# =============================================================================

epochs_completed = len(history.history["loss"])
val_loss_values = np.asarray(history.history["val_loss"], dtype=float)
best_val_index = int(np.argmin(val_loss_values))
best_val_epoch = best_val_index + 1
best_val_loss = float(val_loss_values[best_val_index])
training_loss_at_best = float(history.history["loss"][best_val_index])

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Epochs completed: {epochs_completed}")
print(f"Best validation-loss epoch: {best_val_epoch}")
print(f"Best validation loss: {best_val_loss:.8f}")
print(f"Training loss at best validation epoch: {training_loss_at_best:.8f}")

# =============================================================================
# 23. SAVE RESTORED BEST MODEL
# =============================================================================

model.save(FINAL_MODEL_PATH)

print(f"\nFine-tuned model:\n{FINAL_MODEL_PATH}")
print(f"\nValidation checkpoint:\n{CHECKPOINT_MODEL_PATH}")

# =============================================================================
# 24. SAVE HISTORY
# =============================================================================

history_df = pd.DataFrame(history.history)
history_df.insert(
    0,
    "epoch",
    np.arange(1, len(history_df) + 1),
)
history_df["learning_rate"] = lr_history.learning_rates

training_history_path = RESULTS_DIR / "training_history.csv"
history_df.to_csv(training_history_path, index=False)

# =============================================================================
# 25. SAVE NEGATIVE SELECTION HISTORY
# =============================================================================

selection_history_df = pd.DataFrame(train_gen.selection_history)
selection_history_df = selection_history_df[
    selection_history_df["epoch"] <= epochs_completed
].copy()

selection_history_path = RESULTS_DIR / "negative_selection_history.csv"
selection_history_df.to_csv(selection_history_path, index=False)

# =============================================================================
# 26. SAVE SPLIT SUMMARY
# =============================================================================

split_summary = pd.DataFrame(
    [
        {
            "component": "original_train",
            "n_records": len(original_train_records),
        },
        {
            "component": "original_validation",
            "n_records": len(original_validation_records),
        },
        {
            "component": "stage1_negative_train_unique_objects",
            "n_records": negative_train_df.loc[
                negative_train_df["negative_source"] == "stage1_rejected",
                "sample_id",
            ].nunique(),
        },
        {
            "component": "stage1_negative_validation_unique_objects",
            "n_records": N_STAGE1_VALIDATION_OBJECTS,
        },
        {
            "component": "stage2_negative_train_unique_objects",
            "n_records": negative_train_df.loc[
                negative_train_df["negative_source"] == "stage2_not_waste",
                "sample_id",
            ].nunique(),
        },
        {
            "component": "stage2_negative_validation_unique_objects",
            "n_records": N_STAGE2_VALIDATION_OBJECTS,
        },
    ]
)

split_summary_path = RESULTS_DIR / "training_validation_split_summary.csv"
split_summary.to_csv(split_summary_path, index=False)

# =============================================================================
# 27. PLOTS
# =============================================================================

loss_plot_path = RESULTS_DIR / "training_validation_loss.png"

plt.figure(figsize=(8, 5))
plt.plot(
    history_df["epoch"],
    history_df["loss"],
    label="Training loss",
)
plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    label="Fixed validation loss",
)
plt.axvline(
    best_val_epoch,
    linestyle="--",
    label=f"Best validation epoch = {best_val_epoch}",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Conservative complementary fine-tuning")
plt.legend()
plt.tight_layout()
plt.savefig(
    loss_plot_path,
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# =============================================================================
# 28. SAVE SUMMARY
# =============================================================================

summary_path = RESULTS_DIR / "fine_tuning_summary.txt"

with open(summary_path, "w") as f:
    f.write("CONSERVATIVE COMPLEMENTARY HARD-NEGATIVE FINE-TUNING\n")
    f.write("=" * 76 + "\n\n")
    f.write(f"Starting model:\n{BASE_MODEL_PATH}\n\n")
    f.write(f"Output model:\n{FINAL_MODEL_PATH}\n\n")
    f.write(f"Negative-loss weight: {NEGATIVE_LOSS_WEIGHT}\n")
    f.write(
        "Stage-1 negative objects / training epoch: "
        f"{N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH}\n"
    )
    f.write(
        "Stage-2 negative objects / training epoch: "
        f"{N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH}\n\n"
    )
    f.write(
        f"Original training replay records: {len(original_train_records)}\n"
    )
    f.write(
        f"Original fixed validation records: {len(original_validation_records)}\n"
    )
    f.write(
        f"Stage-1 validation negative objects: {N_STAGE1_VALIDATION_OBJECTS}\n"
    )
    f.write(
        f"Stage-2 validation negative objects: {N_STAGE2_VALIDATION_OBJECTS}\n\n"
    )
    f.write(f"Epochs completed: {epochs_completed}\n")
    f.write(f"Best validation epoch: {best_val_epoch}\n")
    f.write(f"Best validation loss: {best_val_loss:.8f}\n")
    f.write(
        "Training loss at best validation epoch: "
        f"{training_loss_at_best:.8f}\n\n"
    )
    f.write("Model selection criterion: fixed internal validation loss\n")
    f.write("Independent test area: not used during this fine-tuning run\n")

# =============================================================================
# 29. FINAL REPORT
# =============================================================================

print("\n" + "=" * 80)
print("OUTPUTS")
print("=" * 80)
print(f"\nFine-tuned model:\n{FINAL_MODEL_PATH}")
print(f"\nBest validation checkpoint:\n{CHECKPOINT_MODEL_PATH}")
print(f"\nTraining history:\n{training_history_path}")
print(f"\nNegative selection history:\n{selection_history_path}")
print(
    "\nFixed validation records:\n"
    f"{RESULTS_DIR / 'fixed_internal_validation_records.csv'}"
)
print(f"\nSplit summary:\n{split_summary_path}")
print(f"\nSummary:\n{summary_path}")

print("\n" + "=" * 80)
print("NEXT STEP")
print("=" * 80)
print(
    "\nUse the fixed internal validation loss for model selection. "
    "Do not use the previously inspected independent AOI to choose "
    "lambda, sampling ratios, epochs or threshold."
)
print(
    "For a final unbiased estimate after iterative tuning, use another "
    "spatial area that has remained untouched."
)
print("\nDone.")
