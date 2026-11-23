# =============================================================================
# FINE-TUNE EXISTING 10-CLASS U-NET WITH COMPLEMENTARY HARD NEGATIVES
# =============================================================================
#
# PURPOSE
# -------
# Continue training the existing multiclass U-Net using two supervision streams:
#
#   1. Original fully labelled image-mask pairs:
#        ordinary 10-class categorical cross-entropy.
#
#   2. Manually verified false-positive regions:
#        "this pixel is definitely NOT waste", while the true alternative
#        class is unknown.
#
# For stream 2, the loss penalises high P(waste) only:
#
#       L_negative = -log(1 - P(waste))
#
# It does NOT force the pixel into Soil, Vegetation, Built-up, etc.
#
# HARD-NEGATIVE MASK FORMAT
# -------------------------
# Each hard-negative image requires a binary negative mask:
#
#       1 or 255 = manually confirmed NOT-WASTE pixel
#       0        = unknown / ignore this pixel
#
# IMPORTANT:
# Do not mark the whole chip as negative unless you have confirmed that the
# whole chip contains no waste. Normally the binary mask should cover only the
# false-positive region(s) that you manually verified.
#
# The independent test area remains untouched during fine-tuning.
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

# Existing trained model. Change this if you want to start from a different
# checkpoint.
BASE_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_march_finetuned_best.keras"
)

# Original fully labelled replay data.
ORIGINAL_DATA_DIR = (
    BASE_PATH
    / "data"
    / "tiled_data"
    / "partitioned"
    / "modelDevelopment"
)

ORIGINAL_IMAGE_DIR = ORIGINAL_DATA_DIR / "images"
ORIGINAL_MASK_DIR = ORIGINAL_DATA_DIR / "masks"

# Complementary hard-negative data.
#
# Re-use the hard-negative RGB images you already created, but create NEW
# binary masks in the "negative_masks" folder.
NEW_DATA_DIR = (
    BASE_PATH
    / "data"
    / "tiled_data"
    / "finetuning_hard_negatives_complete"
)

NEW_IMAGE_DIR = NEW_DATA_DIR / "images"

# NEW: these are NOT 10-class masks.
# They are binary masks: 1/255 = confirmed not waste; 0 = ignore.
NEW_NEGATIVE_MASK_DIR = NEW_DATA_DIR / "negative_masks"

NEW_MANIFEST_PATH = NEW_DATA_DIR / "finetuning_manifest.csv"

RESULTS_DIR = (
    BASE_PATH
    / "results"
    / "complementary_hard_negative_finetuning"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_complementary_hardnegative_checkpoint.keras"
)

FINAL_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_complementary_hardnegative_finetuned.keras"
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

CLASS_NAMES = {
    1: "Waste",
    2: "Soil",
    3: "Vegetation",
    4: "Built-up surface",
    5: "Built-up surface (paved area)",
    6: "Water Body",
    7: "Natural Surface - rocks",
    8: "Non-persistent object",
    9: "Ancillary class",
    10: "Unidentified",
}


# =============================================================================
# 5. FINE-TUNING SETTINGS
# =============================================================================

MAX_EPOCHS = 500

INITIAL_LEARNING_RATE = 1e-5
MIN_LEARNING_RATE = 1e-7

EARLY_STOP_PATIENCE = 30
REDUCE_LR_PATIENCE = 10
MIN_DELTA = 1e-4

# Present each hard-negative chip this many times per nominal epoch.
# Keep your existing value initially, then assess the independent test area.
HARD_NEGATIVE_REPEAT = 3

# Weight of the complementary "definitely not waste" term.
#
# Start at 1.0. Do not increase this simply because false positives remain:
# first check whether waste recall is being reduced on the independent test.
NEGATIVE_LOSS_WEIGHT = 1.0

EPSILON = 1e-7


# =============================================================================
# 6. INPUT CHECKS
# =============================================================================

required_paths = [
    BASE_MODEL_PATH,
    ORIGINAL_IMAGE_DIR,
    ORIGINAL_MASK_DIR,
    NEW_IMAGE_DIR,
    NEW_NEGATIVE_MASK_DIR,
    NEW_MANIFEST_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired input not found:\n{path}"
        )

print("\n" + "=" * 80)
print("COMPLEMENTARY HARD-NEGATIVE FINE-TUNING")
print("=" * 80)

print(f"\nStarting model:\n{BASE_MODEL_PATH}")
print(f"\nOriginal replay data:\n{ORIGINAL_DATA_DIR}")
print(f"\nHard-negative images:\n{NEW_IMAGE_DIR}")
print(f"\nBinary negative masks:\n{NEW_NEGATIVE_MASK_DIR}")
print(f"\nNegative-loss weight: {NEGATIVE_LOSS_WEIGHT}")
print(f"Hard-negative repeat: {HARD_NEGATIVE_REPEAT}x")


# =============================================================================
# 7. UTILITIES
# =============================================================================

def encode_raw_mask(mask):
    """
    Convert original raw labels 1-10 to model channels 0-9.
    """

    mapped = np.full(
        mask.shape,
        -1,
        dtype=np.int32,
    )

    for raw_label, model_idx in LABEL_MAP.items():
        mapped[mask == raw_label] = model_idx

    if np.any(mapped < 0):
        bad = np.unique(
            mask[mapped < 0]
        )

        raise ValueError(
            "Original mask contains values outside 1-10: "
            f"{bad.tolist()}"
        )

    return mapped


def validate_negative_mask(mask, mask_path):
    """
    Complementary mask:
      0     = ignore / unknown
      1/255 = confirmed NOT WASTE
    """

    values = set(
        int(v)
        for v in np.unique(mask)
    )

    allowed = {0, 1, 255}

    invalid = sorted(
        values - allowed
    )

    if invalid:
        raise ValueError(
            "\nA complementary hard-negative mask contains invalid values.\n"
            "Expected only 0, 1, or 255.\n"
            "0 = ignore; 1/255 = confirmed NOT waste.\n\n"
            f"Mask:\n{mask_path}\n\n"
            f"Observed values: {sorted(values)}\n"
            f"Invalid values: {invalid}\n\n"
            "If this is one of your old 10-class masks, do NOT use it here. "
            "Create a binary negative mask instead."
        )

    return (mask > 0).astype(
        np.float32
    )


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

    image = (
        image / 255.0
    ).astype(np.float32)

    return image


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

    mask = np.rint(
        mask
    ).astype(np.int32)

    return mask


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

    mask = np.rint(
        mask
    ).astype(np.int32)

    return validate_negative_mask(
        mask,
        path,
    )


def augment_pair(image, target_mask):
    """
    Apply the same geometry to image and either:
      - original multiclass mask, or
      - binary complementary mask.
    """

    if random.random() < 0.5:
        image = np.flipud(image)
        target_mask = np.flipud(target_mask)

    if random.random() < 0.5:
        angle = random.uniform(
            -15,
            15,
        )

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
# 8. DISCOVER / AUDIT ORIGINAL FULLY LABELLED DATA
# =============================================================================

print("\n" + "=" * 80)
print("AUDITING ORIGINAL REPLAY DATA")
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

common_original = sorted(
    set(original_images).intersection(
        original_masks
    )
)

if not common_original:
    raise FileNotFoundError(
        "No matching original image-mask pairs found."
    )

original_records = []
excluded_original_records = []

for counter, filename in enumerate(
    common_original,
    start=1,
):

    image_path = original_images[filename]
    mask_path = original_masks[filename]

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    observed = set(
        int(v)
        for v in np.unique(mask)
    )

    invalid = sorted(
        observed - VALID_RAW_LABEL_SET
    )

    if invalid:
        excluded_original_records.append(
            {
                "filename": filename,
                "invalid_labels": ",".join(
                    str(v)
                    for v in invalid
                ),
            }
        )
        continue

    original_records.append(
        {
            "image_path": image_path,
            "mask_path": mask_path,
            "negative_mask_path": None,
            "source": "original",
            "sample_id": -1,
            "tile_id": filename,
        }
    )

print(
    f"Original pairs examined: {len(common_original):,}"
)

print(
    f"Clean original replay pairs: {len(original_records):,}"
)

print(
    "Excluded incompatible original masks: "
    f"{len(excluded_original_records):,}"
)

pd.DataFrame(
    excluded_original_records
).to_csv(
    RESULTS_DIR / "excluded_original_masks.csv",
    index=False,
)


# =============================================================================
# 9. LOAD / AUDIT COMPLEMENTARY HARD NEGATIVES
# =============================================================================

print("\n" + "=" * 80)
print("AUDITING COMPLEMENTARY HARD NEGATIVES")
print("=" * 80)

manifest = pd.read_csv(
    NEW_MANIFEST_PATH
)

base_required = [
    "tile_id",
    "sample_id",
    "image_file",
]

missing = [
    c
    for c in base_required
    if c not in manifest.columns
]

if missing:
    raise ValueError(
        "Fine-tuning manifest is missing columns: "
        f"{missing}"
    )

# Preferred new column.
#
# To make transition easier, if your manifest still uses "mask_file",
# the script will use it as the negative-mask filename. The file itself
# must still be binary (0/1/255); an old 10-class mask will be rejected.
if "negative_mask_file" not in manifest.columns:

    if "mask_file" in manifest.columns:
        print(
            "\nNOTE: manifest has 'mask_file' rather than "
            "'negative_mask_file'. Treating mask_file as the binary "
            "complementary mask filename."
        )

        manifest[
            "negative_mask_file"
        ] = manifest[
            "mask_file"
        ]

    else:
        raise ValueError(
            "Manifest must contain 'negative_mask_file' "
            "(or legacy column 'mask_file')."
        )

manifest["sample_id"] = pd.to_numeric(
    manifest["sample_id"],
    errors="raise",
).astype(int)

new_records = []
negative_audit = []

for _, row in manifest.iterrows():

    image_path = (
        NEW_IMAGE_DIR
        / str(row["image_file"])
    )

    negative_mask_path = (
        NEW_NEGATIVE_MASK_DIR
        / str(row["negative_mask_file"])
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"\nHard-negative image not found:\n{image_path}"
        )

    if not negative_mask_path.exists():
        raise FileNotFoundError(
            "\nBinary hard-negative mask not found:\n"
            f"{negative_mask_path}"
        )

    with rasterio.open(
        negative_mask_path
    ) as src:
        raw_negative = src.read(1)

    binary_negative = validate_negative_mask(
        raw_negative,
        negative_mask_path,
    )

    n_negative = int(
        np.sum(binary_negative > 0)
    )

    if n_negative == 0:
        raise ValueError(
            "\nHard-negative mask contains no confirmed negative pixels:\n"
            f"{negative_mask_path}"
        )

    negative_audit.append(
        {
            "tile_id": str(row["tile_id"]),
            "sample_id": int(row["sample_id"]),
            "image_file": str(row["image_file"]),
            "negative_mask_file": str(
                row["negative_mask_file"]
            ),
            "confirmed_not_waste_pixels": n_negative,
            "mask_pixels": int(binary_negative.size),
            "confirmed_fraction": (
                n_negative
                / binary_negative.size
            ),
        }
    )

    new_records.append(
        {
            "image_path": image_path,
            "mask_path": None,
            "negative_mask_path": negative_mask_path,
            "source": "hard_negative",
            "sample_id": int(row["sample_id"]),
            "tile_id": str(row["tile_id"]),
        }
    )

print(
    f"Unique hard-negative image/mask pairs: {len(new_records):,}"
)

print(
    "Source inspection areas represented: "
    f"{manifest['sample_id'].nunique():,}"
)

negative_audit_df = pd.DataFrame(
    negative_audit
)

negative_audit_df.to_csv(
    RESULTS_DIR / "complementary_negative_mask_audit.csv",
    index=False,
)

print(
    "Confirmed NOT-waste pixels across unique masks: "
    f"{negative_audit_df['confirmed_not_waste_pixels'].sum():,}"
)


# =============================================================================
# 10. BUILD FINE-TUNING RECORDS
# =============================================================================

hard_negative_training_records = (
    new_records
    * HARD_NEGATIVE_REPEAT
)

training_records = (
    original_records
    + hard_negative_training_records
)

random.shuffle(
    training_records
)

print("\n" + "=" * 80)
print("FINE-TUNING DATASET")
print("=" * 80)

print(
    f"Original replay records: {len(original_records):,}"
)

print(
    f"Unique complementary negatives: {len(new_records):,}"
)

print(
    "Effective complementary-negative records / epoch: "
    f"{len(hard_negative_training_records):,}"
)

print(
    f"Total records / epoch: {len(training_records):,}"
)

if training_records:
    print(
        "Complementary-negative record share: "
        f"{100 * len(hard_negative_training_records) / len(training_records):.2f}%"
    )

dataset_manifest = pd.DataFrame(
    [
        {
            "record_number": i,
            "tile_id": r["tile_id"],
            "source": r["source"],
            "sample_id": r["sample_id"],
            "image_path": str(r["image_path"]),
            "full_mask_path": (
                ""
                if r["mask_path"] is None
                else str(r["mask_path"])
            ),
            "negative_mask_path": (
                ""
                if r["negative_mask_path"] is None
                else str(
                    r["negative_mask_path"]
                )
            ),
        }
        for i, r in enumerate(
            training_records,
            start=1,
        )
    ]
)

dataset_manifest.to_csv(
    RESULTS_DIR / "finetuning_dataset_manifest.csv",
    index=False,
)


# =============================================================================
# 11. MIXED-SUPERVISION GENERATOR
# =============================================================================
#
# y_true has 11 channels:
#
#   channels 0-9 : ordinary one-hot 10-class target
#   channel 10   : complementary NOT-WASTE mask
#
# For original records:
#   channels 0-9 = one-hot label
#   channel 10   = 0
#
# For hard-negative records:
#   channels 0-9 = 0 (unknown true class; therefore ignored by CE)
#   channel 10   = 1 only where the pixel is confirmed NOT waste
# =============================================================================

class ComplementaryFineTuneGenerator(
    Sequence
):

    def __init__(
        self,
        records,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True,
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )

        self.records = list(
            records
        )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        self.indices = np.arange(
            len(self.records)
        )

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(
                len(self.indices)
                / self.batch_size
            )
        )

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(
                self.indices
            )

    def __getitem__(
        self,
        idx,
    ):

        batch_indices = self.indices[
            idx * self.batch_size:
            (idx + 1) * self.batch_size
        ]

        images = []
        targets = []

        for record_index in batch_indices:

            record = self.records[
                record_index
            ]

            image = read_rgb(
                record["image_path"]
            )

            # ---------------------------------------------------------
            # Fully labelled original replay example
            # ---------------------------------------------------------

            if record["source"] == "original":

                mask = read_original_mask(
                    record["mask_path"]
                )

                if self.augment:
                    image, mask = augment_pair(
                        image,
                        mask,
                    )

                mask = np.rint(
                    mask
                ).astype(np.int32)

                mapped = encode_raw_mask(
                    mask
                )

                supervised_target = to_categorical(
                    mapped,
                    num_classes=N_CLASSES,
                ).astype(np.float32)

                complementary_negative = np.zeros(
                    (IMG_HEIGHT, IMG_WIDTH, 1),
                    dtype=np.float32,
                )

            # ---------------------------------------------------------
            # Complementary hard-negative example
            # ---------------------------------------------------------

            elif record["source"] == "hard_negative":

                negative_mask = read_negative_mask(
                    record[
                        "negative_mask_path"
                    ]
                )

                if self.augment:
                    image, negative_mask = augment_pair(
                        image,
                        negative_mask,
                    )

                negative_mask = (
                    negative_mask > 0.5
                ).astype(np.float32)

                # No positive class is supplied for these pixels.
                supervised_target = np.zeros(
                    (
                        IMG_HEIGHT,
                        IMG_WIDTH,
                        N_CLASSES,
                    ),
                    dtype=np.float32,
                )

                complementary_negative = negative_mask[
                    ...,
                    np.newaxis
                ].astype(
                    np.float32
                )

            else:
                raise ValueError(
                    "Unknown record source: "
                    f"{record['source']}"
                )

            target = np.concatenate(
                [
                    supervised_target,
                    complementary_negative,
                ],
                axis=-1,
            )

            images.append(
                image.astype(np.float32)
            )

            targets.append(
                target.astype(np.float32)
            )

        return (
            np.asarray(
                images,
                dtype=np.float32,
            ),
            np.asarray(
                targets,
                dtype=np.float32,
            ),
        )


train_gen = ComplementaryFineTuneGenerator(
    training_records,
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment=True,
)

print(
    f"\nTraining batches / epoch: {len(train_gen):,}"
)


# =============================================================================
# 12. COMPLEMENTARY LOSS
# =============================================================================

@tf.keras.utils.register_keras_serializable(
    package="WasteFineTuning"
)
def complementary_mixed_loss(
    y_true,
    y_pred,
):
    """
    L = supervised 10-class CE
        + lambda * complementary negative loss

    Complementary negative loss:
        -log(1 - P(waste))

    Only manually confirmed NOT-WASTE pixels contribute to that term.
    Unknown pixels in a hard-negative chip contribute no loss.
    """

    supervised_target = y_true[
        ...,
        :N_CLASSES
    ]

    negative_mask = y_true[
        ...,
        N_CLASSES
    ]

    # Pixels with a one-hot target are fully supervised.
    supervised_mask = tf.cast(
        tf.reduce_sum(
            supervised_target,
            axis=-1,
        ) > 0.5,
        tf.float32,
    )

    # ---------------------------------------------------------
    # Ordinary multiclass loss on original replay pixels
    # ---------------------------------------------------------

    per_pixel_ce = (
        tf.keras.losses.categorical_crossentropy(
            supervised_target,
            y_pred,
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

    # ---------------------------------------------------------
    # "Definitely not waste" loss
    # ---------------------------------------------------------

    p_waste = tf.clip_by_value(
        y_pred[
            ...,
            WASTE_CLASS_IDX
        ],
        EPSILON,
        1.0 - EPSILON,
    )

    per_pixel_negative_loss = -tf.math.log(
        1.0 - p_waste
    )

    negative_count = tf.reduce_sum(
        negative_mask
    )

    negative_loss = (
        tf.reduce_sum(
            per_pixel_negative_loss
            * negative_mask
        )
        / tf.maximum(
            negative_count,
            1.0,
        )
    )

    return (
        supervised_loss
        + NEGATIVE_LOSS_WEIGHT
        * negative_loss
    )


# =============================================================================
# 13. TRAINING DIAGNOSTICS
# =============================================================================

@tf.keras.utils.register_keras_serializable(
    package="WasteFineTuning"
)
def supervised_accuracy(
    y_true,
    y_pred,
):

    target = y_true[
        ...,
        :N_CLASSES
    ]

    mask = tf.cast(
        tf.reduce_sum(
            target,
            axis=-1,
        ) > 0.5,
        tf.float32,
    )

    true_class = tf.argmax(
        target,
        axis=-1,
        output_type=tf.int32,
    )

    pred_class = tf.argmax(
        y_pred,
        axis=-1,
        output_type=tf.int32,
    )

    correct = tf.cast(
        tf.equal(
            true_class,
            pred_class,
        ),
        tf.float32,
    )

    return (
        tf.reduce_sum(
            correct * mask
        )
        / tf.maximum(
            tf.reduce_sum(mask),
            1.0,
        )
    )


@tf.keras.utils.register_keras_serializable(
    package="WasteFineTuning"
)
def negative_mean_waste_probability(
    y_true,
    y_pred,
):

    negative_mask = y_true[
        ...,
        N_CLASSES
    ]

    p_waste = y_pred[
        ...,
        WASTE_CLASS_IDX
    ]

    return (
        tf.reduce_sum(
            p_waste
            * negative_mask
        )
        / tf.maximum(
            tf.reduce_sum(
                negative_mask
            ),
            1.0,
        )
    )


@tf.keras.utils.register_keras_serializable(
    package="WasteFineTuning"
)
def negative_waste_argmax_rate(
    y_true,
    y_pred,
):

    negative_mask = y_true[
        ...,
        N_CLASSES
    ]

    predicted_waste = tf.cast(
        tf.argmax(
            y_pred,
            axis=-1,
        ) == WASTE_CLASS_IDX,
        tf.float32,
    )

    return (
        tf.reduce_sum(
            predicted_waste
            * negative_mask
        )
        / tf.maximum(
            tf.reduce_sum(
                negative_mask
            ),
            1.0,
        )
    )


# =============================================================================
# 14. LOAD EXISTING MODEL
# =============================================================================

print("\n" + "=" * 80)
print("LOADING EXISTING U-NET")
print("=" * 80)

tf.keras.backend.clear_session()

model = load_model(
    BASE_MODEL_PATH,
    compile=False,
)

print(
    f"Model input shape: {model.input_shape}"
)

print(
    f"Model output shape: {model.output_shape}"
)

if tuple(
    model.input_shape[1:]
) != (
    IMG_HEIGHT,
    IMG_WIDTH,
    3,
):
    raise ValueError(
        "Unexpected model input shape: "
        f"{model.input_shape}"
    )

if int(
    model.output_shape[-1]
) != N_CLASSES:
    raise ValueError(
        "Loaded model is not the expected "
        f"{N_CLASSES}-class U-Net. "
        f"Output: {model.output_shape}"
    )

for layer in model.layers:
    layer.trainable = True

print(
    "Trainable parameters: "
    f"{sum(np.prod(v.shape) for v in model.trainable_weights):,}"
)


# =============================================================================
# 15. COMPILE
# =============================================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=INITIAL_LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss=complementary_mixed_loss,
    metrics=[
        supervised_accuracy,
        negative_mean_waste_probability,
        negative_waste_argmax_rate,
    ],
)


# =============================================================================
# 16. CALLBACKS
# =============================================================================

class LearningRateHistory(
    Callback
):

    def __init__(self):
        super().__init__()
        self.learning_rates = []

    def on_epoch_end(
        self,
        epoch,
        logs=None,
    ):

        lr = float(
            tf.keras.backend.get_value(
                self.model.optimizer.learning_rate
            )
        )

        self.learning_rates.append(
            lr
        )

        print(
            f" - learning_rate: {lr:.8g}"
        )


lr_history = LearningRateHistory()

early_stop = EarlyStopping(
    monitor="loss",
    mode="min",
    min_delta=MIN_DELTA,
    patience=EARLY_STOP_PATIENCE,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="loss",
    mode="min",
    factor=0.5,
    patience=REDUCE_LR_PATIENCE,
    min_delta=MIN_DELTA,
    min_lr=MIN_LEARNING_RATE,
    verbose=1,
)

checkpoint = ModelCheckpoint(
    filepath=str(
        CHECKPOINT_MODEL_PATH
    ),
    monitor="loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)

csv_log_path = (
    RESULTS_DIR
    / "training_log.csv"
)

csv_logger = CSVLogger(
    csv_log_path
)

terminate_nan = TerminateOnNaN()


# =============================================================================
# 17. TRAIN
# =============================================================================

print("\n" + "=" * 80)
print("STARTING COMPLEMENTARY FINE-TUNING")
print("=" * 80)

print(
    "\nTraining objective:"
    "\n  Original pixels  -> ordinary 10-class cross-entropy"
    "\n  Hard negatives   -> penalise P(waste), true class left unspecified"
)

print(
    f"\nInitial LR: {INITIAL_LEARNING_RATE}"
)

print(
    f"Negative-loss weight: {NEGATIVE_LOSS_WEIGHT}"
)

print(
    "\nNo validation/test data are used during this fine-tuning step."
)

print(
    "The independent test area remains untouched."
)

history = model.fit(
    train_gen,
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
# 18. SAVE MODEL / HISTORY
# =============================================================================

epochs_completed = len(
    history.history["loss"]
)

loss_values = np.asarray(
    history.history["loss"],
    dtype=float,
)

lowest_loss_index = int(
    np.argmin(
        loss_values
    )
)

lowest_loss_epoch = (
    lowest_loss_index + 1
)

lowest_training_loss = float(
    loss_values[
        lowest_loss_index
    ]
)

model.save(
    FINAL_MODEL_PATH
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


# =============================================================================
# 19. DIAGNOSTIC PLOTS
# =============================================================================

loss_plot_path = (
    RESULTS_DIR
    / "training_loss.png"
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history_df["epoch"],
    history_df["loss"],
    label="Combined training loss",
)

plt.axvline(
    lowest_loss_epoch,
    linestyle="--",
    label=(
        f"Lowest loss epoch = "
        f"{lowest_loss_epoch}"
    ),
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "Complementary hard-negative fine-tuning"
)
plt.legend()
plt.tight_layout()

plt.savefig(
    loss_plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


diagnostic_columns = [
    "supervised_accuracy",
    "negative_mean_waste_probability",
    "negative_waste_argmax_rate",
]

available = [
    c
    for c in diagnostic_columns
    if c in history_df.columns
]

if available:

    metric_plot_path = (
        RESULTS_DIR
        / "training_diagnostics.png"
    )

    plt.figure(
        figsize=(8, 5)
    )

    for c in available:
        plt.plot(
            history_df["epoch"],
            history_df[c],
            label=c,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training diagnostic")
    plt.title(
        "Complementary fine-tuning diagnostics"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        metric_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# 20. SUMMARY
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
        "COMPLEMENTARY HARD-NEGATIVE U-NET FINE-TUNING\n"
    )

    f.write(
        "=" * 72
        + "\n\n"
    )

    f.write(
        f"Starting model:\n{BASE_MODEL_PATH}\n\n"
    )

    f.write(
        f"Output model:\n{FINAL_MODEL_PATH}\n\n"
    )

    f.write(
        f"Original replay records: {len(original_records)}\n"
    )

    f.write(
        f"Unique hard-negative records: {len(new_records)}\n"
    )

    f.write(
        f"Hard-negative repeat: {HARD_NEGATIVE_REPEAT}\n"
    )

    f.write(
        f"Negative-loss weight: {NEGATIVE_LOSS_WEIGHT}\n"
    )

    f.write(
        "Confirmed NOT-waste pixels across unique masks: "
        f"{negative_audit_df['confirmed_not_waste_pixels'].sum()}\n\n"
    )

    f.write(
        f"Epochs completed: {epochs_completed}\n"
    )

    f.write(
        f"Lowest training-loss epoch: {lowest_loss_epoch}\n"
    )

    f.write(
        f"Lowest training loss: {lowest_training_loss:.8f}\n"
    )

    f.write(
        f"Final learning rate: {lr_history.learning_rates[-1]:.8g}\n\n"
    )

    f.write(
        "Hard-negative supervision:\n"
        "  - confirmed negative pixels penalise P(waste)\n"
        "  - no alternative non-waste class is specified\n"
        "  - pixels coded 0 in complementary masks are ignored\n\n"
    )

    f.write(
        "Validation:\n"
        "  No validation/test data were used during fine-tuning.\n"
        "  Compare the starting and fine-tuned models on the same "
        "independent test area.\n"
    )


# =============================================================================
# 21. FINAL REPORT
# =============================================================================

print("\n" + "=" * 80)
print("FINE-TUNING COMPLETE")
print("=" * 80)

print(
    f"Epochs completed: {epochs_completed}"
)

print(
    f"Lowest training-loss epoch: {lowest_loss_epoch}"
)

print(
    f"Lowest training loss: {lowest_training_loss:.8f}"
)

print(
    f"\nFine-tuned model:\n{FINAL_MODEL_PATH}"
)

print(
    f"\nTraining history:\n{history_path}"
)

print(
    "\nComplementary-mask audit:\n"
    f"{RESULTS_DIR / 'complementary_negative_mask_audit.csv'}"
)

print(
    f"\nSummary:\n{summary_path}"
)

print("\n" + "=" * 80)
print("NEXT STEP")
print("=" * 80)

print(
    "\nRun BOTH the starting and fine-tuned models on the SAME "
    "independent test area."
)

print(
    "Check whether false positives decrease WITHOUT an unacceptable "
    "loss of waste recall or segmentation quality."
)

print("\nDone.")
