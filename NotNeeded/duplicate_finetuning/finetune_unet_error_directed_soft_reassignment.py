# =============================================================================
# ERROR-DIRECTED SOFT-REASSIGNMENT FINE-TUNING
# =============================================================================
#
# Starts from the ORIGINAL March 10-class U-Net.
#
# Original fully labelled pixels:
#     ordinary 10-class categorical cross-entropy.
#
# Confirmed false-positive pixels:
#     (1) suppress P(Waste):
#             -log(1 - P_student(Waste))
#
#     (2) when the frozen original model would still call the pixel Waste
#         (argmax Waste OR P_teacher(Waste) >= 0.50), remove Waste from the
#         teacher distribution, renormalise the other 9 probabilities, and
#         train the student's conditional non-waste distribution toward that
#         SOFT target.
#
# The alternative class is therefore not asserted as a hard manual label.
# The only certain manual statement is: "this pixel is NOT waste".
#
# The locked independent test AOI is not used during training or checkpointing.
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
    / "error_directed_soft_reassignment"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_error_directed_soft_reassignment_best.keras"
)

FINAL_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_error_directed_soft_reassignment_finetuned.keras"
)

# =============================================================================
# 3. SETTINGS
# =============================================================================

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 4  # two U-Nets are resident: frozen teacher + trainable student

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

# Rotating training negatives.
N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH = 300
N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH = 200

# Fixed internal validation data, all outside the locked independent test AOI.
ORIGINAL_VALIDATION_FRACTION = 0.15
N_STAGE1_VALIDATION_OBJECTS = 150
N_STAGE2_VALIDATION_OBJECTS = 100

# A confirmed-negative pixel receives soft reassignment if the frozen teacher
# still regards Waste as the likely class.
TEACHER_WASTE_PROBABILITY_TRIGGER = 0.50

# "Not waste" is certain; the alternative class is inferred, so use a more
# conservative weight for soft reassignment.
NEGATIVE_LOSS_WEIGHT = 1.0
REDISTRIBUTION_LOSS_WEIGHT = 0.50

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
print("ERROR-DIRECTED SOFT-REASSIGNMENT FINE-TUNING")
print("=" * 80)
print(f"\nOriginal teacher/student model:\n{BASE_MODEL_PATH}")
print(f"\nHard-negative library:\n{NEGATIVE_DATA_DIR}")
print(f"\nTeacher P(Waste) trigger: {TEACHER_WASTE_PROBABILITY_TRIGGER}")
print(f"Negative-loss weight: {NEGATIVE_LOSS_WEIGHT}")
print(f"Redistribution-loss weight: {REDISTRIBUTION_LOSS_WEIGHT}")

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

class ErrorDirectedFineTuner(tf.keras.Model):

    def __init__(self, student, teacher, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.supervised_loss_tracker = tf.keras.metrics.Mean(
            name="supervised_loss"
        )
        self.negative_loss_tracker = tf.keras.metrics.Mean(
            name="negative_loss"
        )
        self.redistribution_loss_tracker = tf.keras.metrics.Mean(
            name="redistribution_loss"
        )

        self.student_neg_p_tracker = tf.keras.metrics.Mean(
            name="student_negative_mean_waste_probability"
        )
        self.student_neg_argmax_tracker = tf.keras.metrics.Mean(
            name="student_negative_waste_argmax_rate"
        )
        self.teacher_neg_p_tracker = tf.keras.metrics.Mean(
            name="teacher_negative_mean_waste_probability"
        )
        self.teacher_neg_argmax_tracker = tf.keras.metrics.Mean(
            name="teacher_negative_waste_argmax_rate"
        )
        self.trigger_tracker = tf.keras.metrics.Mean(
            name="redistribution_trigger_rate"
        )
        self.supervised_accuracy_tracker = tf.keras.metrics.Mean(
            name="supervised_accuracy"
        )

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.supervised_loss_tracker,
            self.negative_loss_tracker,
            self.redistribution_loss_tracker,
            self.student_neg_p_tracker,
            self.student_neg_argmax_tracker,
            self.teacher_neg_p_tracker,
            self.teacher_neg_argmax_tracker,
            self.trigger_tracker,
            self.supervised_accuracy_tracker,
        ]

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def compute_components(self, x, y_true, training):
        supervised_target = y_true[..., :N_CLASSES]
        negative_mask = y_true[..., N_CLASSES]

        supervised_mask = tf.cast(
            tf.reduce_sum(
                supervised_target,
                axis=-1,
            ) > 0.5,
            tf.float32,
        )

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

        # A. Ordinary 10-class supervision.
        ce = tf.keras.losses.categorical_crossentropy(
            supervised_target,
            student_pred,
        )

        supervised_count = tf.reduce_sum(supervised_mask)

        supervised_loss = (
            tf.reduce_sum(
                ce * supervised_mask
            )
            / tf.maximum(
                supervised_count,
                1.0,
            )
        )

        # B. Manually certain "not waste" supervision.
        student_p_waste = tf.clip_by_value(
            student_pred[..., WASTE_CLASS_IDX],
            EPSILON,
            1.0 - EPSILON,
        )

        teacher_p_waste = tf.clip_by_value(
            teacher_pred[..., WASTE_CLASS_IDX],
            EPSILON,
            1.0 - EPSILON,
        )

        negative_count = tf.reduce_sum(negative_mask)

        negative_loss = (
            tf.reduce_sum(
                (
                    -tf.math.log(
                        1.0 - student_p_waste
                    )
                )
                * negative_mask
            )
            / tf.maximum(
                negative_count,
                1.0,
            )
        )

        # C. Which confirmed negatives were actually teacher waste errors?
        teacher_argmax_is_waste = tf.cast(
            tf.argmax(
                teacher_pred,
                axis=-1,
                output_type=tf.int32,
            ) == WASTE_CLASS_IDX,
            tf.float32,
        )

        teacher_high_waste = tf.cast(
            teacher_p_waste
            >= TEACHER_WASTE_PROBABILITY_TRIGGER,
            tf.float32,
        )

        trigger = tf.maximum(
            teacher_argmax_is_waste,
            teacher_high_waste,
        )

        redistribution_mask = (
            negative_mask
            * trigger
        )

        # Higher-confidence teacher waste mistakes get more redistribution weight.
        redistribution_weight = (
            redistribution_mask
            * tf.stop_gradient(
                teacher_p_waste
            )
        )

        # D. Remove Waste and renormalise the OTHER NINE classes.
        # Waste is class 0, so classes 1:10 are the non-waste classes.
        teacher_nonwaste = tf.clip_by_value(
            teacher_pred[..., 1:],
            EPSILON,
            1.0,
        )

        student_nonwaste = tf.clip_by_value(
            student_pred[..., 1:],
            EPSILON,
            1.0,
        )

        teacher_q = (
            teacher_nonwaste
            / tf.maximum(
                tf.reduce_sum(
                    teacher_nonwaste,
                    axis=-1,
                    keepdims=True,
                ),
                EPSILON,
            )
        )

        teacher_q = tf.stop_gradient(
            teacher_q
        )

        student_q = (
            student_nonwaste
            / tf.maximum(
                tf.reduce_sum(
                    student_nonwaste,
                    axis=-1,
                    keepdims=True,
                ),
                EPSILON,
            )
        )

        redistribution_ce = (
            -tf.reduce_sum(
                teacher_q
                * tf.math.log(
                    tf.clip_by_value(
                        student_q,
                        EPSILON,
                        1.0,
                    )
                ),
                axis=-1,
            )
        )

        redistribution_weight_sum = tf.reduce_sum(
            redistribution_weight
        )

        redistribution_loss = (
            tf.reduce_sum(
                redistribution_ce
                * redistribution_weight
            )
            / tf.maximum(
                redistribution_weight_sum,
                1.0,
            )
        )

        total_loss = (
            supervised_loss
            + NEGATIVE_LOSS_WEIGHT
            * negative_loss
            + REDISTRIBUTION_LOSS_WEIGHT
            * redistribution_loss
        )

        # Diagnostics.
        student_waste_argmax = tf.cast(
            tf.argmax(
                student_pred,
                axis=-1,
                output_type=tf.int32,
            ) == WASTE_CLASS_IDX,
            tf.float32,
        )

        student_neg_mean_p = (
            tf.reduce_sum(
                student_p_waste
                * negative_mask
            )
            / tf.maximum(
                negative_count,
                1.0,
            )
        )

        teacher_neg_mean_p = (
            tf.reduce_sum(
                teacher_p_waste
                * negative_mask
            )
            / tf.maximum(
                negative_count,
                1.0,
            )
        )

        student_neg_argmax = (
            tf.reduce_sum(
                student_waste_argmax
                * negative_mask
            )
            / tf.maximum(
                negative_count,
                1.0,
            )
        )

        teacher_neg_argmax = (
            tf.reduce_sum(
                teacher_argmax_is_waste
                * negative_mask
            )
            / tf.maximum(
                negative_count,
                1.0,
            )
        )

        trigger_rate = (
            tf.reduce_sum(
                redistribution_mask
            )
            / tf.maximum(
                negative_count,
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

        return {
            "total_loss": total_loss,
            "supervised_loss": supervised_loss,
            "negative_loss": negative_loss,
            "redistribution_loss": redistribution_loss,
            "student_neg_mean_p": student_neg_mean_p,
            "student_neg_argmax": student_neg_argmax,
            "teacher_neg_mean_p": teacher_neg_mean_p,
            "teacher_neg_argmax": teacher_neg_argmax,
            "trigger_rate": trigger_rate,
            "supervised_accuracy": supervised_accuracy,
            "negative_count": negative_count,
            "supervised_count": supervised_count,
        }

    def update_trackers(self, c):
        self.loss_tracker.update_state(c["total_loss"])
        self.supervised_loss_tracker.update_state(c["supervised_loss"])
        self.negative_loss_tracker.update_state(c["negative_loss"])
        self.redistribution_loss_tracker.update_state(
            c["redistribution_loss"]
        )

        self.student_neg_p_tracker.update_state(
            c["student_neg_mean_p"],
            sample_weight=c["negative_count"],
        )

        self.student_neg_argmax_tracker.update_state(
            c["student_neg_argmax"],
            sample_weight=c["negative_count"],
        )

        self.teacher_neg_p_tracker.update_state(
            c["teacher_neg_mean_p"],
            sample_weight=c["negative_count"],
        )

        self.teacher_neg_argmax_tracker.update_state(
            c["teacher_neg_argmax"],
            sample_weight=c["negative_count"],
        )

        self.trigger_tracker.update_state(
            c["trigger_rate"],
            sample_weight=c["negative_count"],
        )

        self.supervised_accuracy_tracker.update_state(
            c["supervised_accuracy"],
            sample_weight=c["supervised_count"],
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
            c["total_loss"],
            self.student.trainable_variables,
        )

        self.optimizer.apply_gradients(
            zip(
                grads,
                self.student.trainable_variables,
            )
        )

        self.update_trackers(c)

        return {
            metric.name: metric.result()
            for metric in self.metrics
        }

    def test_step(self, data):
        x, y_true = data

        c = self.compute_components(
            x,
            y_true,
            training=False,
        )

        self.update_trackers(c)

        return {
            metric.name: metric.result()
            for metric in self.metrics
        }


trainer = ErrorDirectedFineTuner(
    student=student,
    teacher=teacher,
    name="error_directed_finetuner",
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
        self.learning_rates.append(lr)
        print(f" - learning_rate: {lr:.8g}")


class BestStudentCheckpoint(Callback):

    def __init__(
        self,
        filepath,
        monitor="val_loss",
    ):
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.best = np.inf
        self.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return

        if current < self.best:
            self.best = float(current)
            self.best_epoch = epoch + 1
            self.model.student.save(
                self.filepath
            )
            print(
                f"\nSaved best student at epoch "
                f"{self.best_epoch} "
                f"({self.monitor}={self.best:.6f}) to:\n"
                f"{self.filepath}"
            )


class NegativeSelectionWriter(Callback):

    def __init__(self, generator, output_path):
        super().__init__()
        self.generator = generator
        self.output_path = Path(output_path)

    def on_epoch_end(self, epoch, logs=None):
        completed_epoch = epoch + 1
        df = pd.DataFrame(
            self.generator.selection_history
        )

        df = df[
            df["epoch"]
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

best_checkpoint = BestStudentCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_loss",
)

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
    "\nOriginal labelled pixels:"
    "\n  10-class categorical cross-entropy"
)
print(
    "\nAll confirmed false-positive pixels:"
    "\n  suppress P(Waste)"
)
print(
    "\nTeacher waste-error pixels:"
    "\n  suppress Waste + soft reassignment among the other 9 classes"
)
print(
    "\nCheckpoint / early stopping:"
    "\n  fixed internal validation set"
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
    history.history["loss"]
)

history_df = pd.DataFrame(
    history.history
)

history_df.insert(
    0,
    "epoch",
    np.arange(
        1,
        len(history_df) + 1,
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

val_loss_values = np.asarray(
    history_df["val_loss"],
    dtype=float,
)

best_index = int(
    np.argmin(
        val_loss_values
    )
)

best_epoch = int(
    history_df.iloc[
        best_index
    ]["epoch"]
)

best_val_loss = float(
    val_loss_values[
        best_index
    ]
)

# EarlyStopping restored the best wrapper weights; save the student only.
trainer.student.save(
    FINAL_MODEL_PATH
)

best_epoch_metrics_path = (
    RESULTS_DIR
    / "best_epoch_metrics.csv"
)

history_df.iloc[
    [
        best_index
    ]
].to_csv(
    best_epoch_metrics_path,
    index=False,
)

# Negative coverage through the selected best epoch.
selection_df = pd.DataFrame(
    train_gen.selection_history
)

selection_df = selection_df[
    selection_df["epoch"]
    <= epochs_completed
].copy()

selection_df.to_csv(
    selection_history_path,
    index=False,
)

selection_best = selection_df[
    selection_df["epoch"]
    <= best_epoch
].copy()

selection_best.to_csv(
    RESULTS_DIR
    / "negative_selection_through_best_epoch.csv",
    index=False,
)

coverage_records = []

for source_name in [
    "stage1_rejected",
    "stage2_not_waste",
]:
    available = set(
        negative_train_df.loc[
            negative_train_df["negative_source"]
            == source_name,
            "sample_id",
        ].astype(str)
    )

    seen = set(
        selection_best.loc[
            selection_best["negative_source"]
            == source_name,
            "sample_id",
        ].astype(str)
    )

    coverage_records.append(
        {
            "negative_source": source_name,
            "available_training_objects": len(available),
            "seen_through_best_epoch": len(seen),
            "coverage_percent_through_best_epoch": (
                100.0 * len(seen) / len(available)
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
    / "negative_object_coverage_through_best_epoch.csv"
)

coverage_df.to_csv(
    coverage_path,
    index=False,
)

# =============================================================================
# 16. PLOTS
# =============================================================================

plt.figure(figsize=(8, 5))
plt.plot(
    history_df["epoch"],
    history_df["loss"],
    label="Training loss",
)
plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    label="Validation loss",
)
plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best validation epoch = {best_epoch}",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Error-directed soft-reassignment fine-tuning")
plt.legend()
plt.tight_layout()
plt.savefig(
    RESULTS_DIR
    / "training_validation_loss.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

val_diag = [
    "val_student_negative_mean_waste_probability",
    "val_teacher_negative_mean_waste_probability",
    "val_student_negative_waste_argmax_rate",
    "val_teacher_negative_waste_argmax_rate",
    "val_redistribution_trigger_rate",
]

available_val_diag = [
    c
    for c in val_diag
    if c in history_df.columns
]

if available_val_diag:
    plt.figure(figsize=(9, 5))

    for c in available_val_diag:
        plt.plot(
            history_df["epoch"],
            history_df[c],
            label=c,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Probability / rate")
    plt.title("Fixed-validation false-positive diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR
        / "validation_negative_error_diagnostics.png",
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
        "ERROR-DIRECTED SOFT-REASSIGNMENT FINE-TUNING\n"
    )
    f.write(
        "=" * 72
        + "\n\n"
    )

    f.write(
        f"Original teacher/student model:\n"
        f"{BASE_MODEL_PATH}\n\n"
    )

    f.write(
        f"Final restored-best student:\n"
        f"{FINAL_MODEL_PATH}\n\n"
    )

    f.write(
        f"Best checkpoint:\n"
        f"{BEST_MODEL_PATH}\n\n"
    )

    f.write(
        "Loss design:\n"
        "  Original labels -> 10-class categorical cross-entropy\n"
        "  Confirmed false positives -> -log(1-P_student(Waste))\n"
        "  Teacher waste errors -> soft redistribution across the other "
        "nine classes\n\n"
    )

    f.write(
        f"Teacher P(Waste) trigger: "
        f"{TEACHER_WASTE_PROBABILITY_TRIGGER}\n"
    )

    f.write(
        f"Negative loss weight: "
        f"{NEGATIVE_LOSS_WEIGHT}\n"
    )

    f.write(
        f"Redistribution loss weight: "
        f"{REDISTRIBUTION_LOSS_WEIGHT}\n\n"
    )

    f.write(
        f"Epochs completed: "
        f"{epochs_completed}\n"
    )

    f.write(
        f"Best validation epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        f"Best validation loss: "
        f"{best_val_loss:.8f}\n\n"
    )

    f.write(
        "Hard-negative coverage through best epoch:\n"
    )

    for _, row in coverage_df.iterrows():
        f.write(
            f"  {row['negative_source']}: "
            f"{int(row['seen_through_best_epoch'])} / "
            f"{int(row['available_training_objects'])} "
            f"({row['coverage_percent_through_best_epoch']:.2f}%)\n"
        )

    f.write(
        "\nThe locked independent test AOI was not used for training, "
        "validation, checkpointing, or early stopping.\n"
    )

# =============================================================================
# 18. FINAL REPORT
# =============================================================================

print("\n" + "=" * 80)
print("ERROR-DIRECTED FINE-TUNING COMPLETE")
print("=" * 80)

print(f"\nEpochs completed: {epochs_completed}")
print(f"Best validation epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.8f}")

print("\nNegative-object coverage through selected best epoch:")
print(
    coverage_df.to_string(
        index=False
    )
)

print(f"\nFine-tuned model:\n{FINAL_MODEL_PATH}")
print(f"\nBest checkpoint:\n{BEST_MODEL_PATH}")
print(f"\nTraining history:\n{history_path}")
print(f"\nBest-epoch metrics:\n{best_epoch_metrics_path}")
print(f"\nSummary:\n{summary_path}")

print("\n" + "=" * 80)
print("NEXT STEP")
print("=" * 80)
print(
    "\nCompare on the SAME locked independent test area:"
    "\n  1. Original March model"
    "\n  2. Complementary-only model"
    "\n  3. Error-directed soft-reassignment model"
)
print("\nDone.")
