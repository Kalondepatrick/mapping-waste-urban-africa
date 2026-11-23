# =============================================================================
# HARD-NEGATIVE FINE-TUNING OF EXISTING MULTICLASS U-NET
# =============================================================================
#
# PURPOSE
# -------
# Refine the previously trained multiclass U-Net using manually verified
# recurrent false-positive landscapes.
#
#
# TRAINING DATA
# -------------
# 1. Original model-development image-mask pairs
#
# 2. Newly created image-mask pairs from the confirmed false-positive
#    inspection areas
#
#
# FINE-TUNING STRATEGY
# --------------------
# - Load existing trained model
# - DO NOT train from scratch
# - Audit original masks and remove corrupted/incompatible masks
# - Use ALL available refinement examples
# - Oversample hard-negative examples 3x
# - Retain original examples as replay data
# - Fine-tune complete network at low learning rate
# - Maximum 500 epochs
# - Reduce learning rate when training loss plateaus
# - Stop when training loss has converged
#
#
# VALIDATION
# ----------
# NO validation/test data are used during this fine-tuning step.
#
# Model performance will subsequently be evaluated using the separately
# defined test area.
#
#
# RAW CLASS LABELS
# ----------------
# 1  Waste
# 2  Soil
# 3  Vegetation
# 4  Built-up surface
# 5  Built-up surface (paved area)
# 6  Water Body
# 7  Natural Surface - rocks
# 8  Non-persistent object
# 9  Ancillary class
# 10 Unidentified
#
# Model channels:
#
# raw label 1  -> channel 0
# raw label 2  -> channel 1
# ...
# raw label 10 -> channel 9
#
# =============================================================================


# =============================================================================
# 0. ENVIRONMENT
# =============================================================================

import os

# Optional:
# make TensorFlow numerical behaviour slightly more reproducible across runs.
#
# Must be set BEFORE importing TensorFlow.
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

from tensorflow.keras.utils import (
    Sequence,
    to_categorical
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
    TerminateOnNaN,
    Callback
)


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

SEED = 20260906


random.seed(
    SEED
)


np.random.seed(
    SEED
)


tf.random.set_seed(
    SEED
)


# =============================================================================
# 3. PATHS
# =============================================================================

BASE_PATH = (
    Path.home()
    / "Documents"
    / "GitHub"
    / "mapping-waste-urban-africa"
)


# -----------------------------------------------------------------------------
# Existing trained U-Net
# -----------------------------------------------------------------------------

BASE_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_march_finetuned_best.keras"
)


# -----------------------------------------------------------------------------
# Original model-development data
# -----------------------------------------------------------------------------

ORIGINAL_DATA_DIR = (
    BASE_PATH
    / "data"
    / "tiled_data"
    / "partitioned"
    / "modelDevelopment"
)


ORIGINAL_IMAGE_DIR = (
    ORIGINAL_DATA_DIR
    / "images"
)


ORIGINAL_MASK_DIR = (
    ORIGINAL_DATA_DIR
    / "masks"
)


# -----------------------------------------------------------------------------
# New hard-negative fine-tuning data
# -----------------------------------------------------------------------------

NEW_DATA_DIR = (
    BASE_PATH
    / "data"
    / "tiled_data"
    / "finetuning_hard_negatives_complete"
)


NEW_IMAGE_DIR = (
    NEW_DATA_DIR
    / "images"
)


NEW_MASK_DIR = (
    NEW_DATA_DIR
    / "masks"
)


NEW_MANIFEST_PATH = (
    NEW_DATA_DIR
    / "finetuning_manifest.csv"
)


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------

RESULTS_DIR = (
    BASE_PATH
    / "results"
    / "hard_negative_finetuning"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------------------------------------------------------
# Output models
# -----------------------------------------------------------------------------

CHECKPOINT_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_hardnegative_checkpoint_lowest_train_loss.keras"
)


FINAL_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_hardnegative_finetuned.keras"
)


# =============================================================================
# 4. IMAGE / MODEL CONFIGURATION
# =============================================================================

IMG_HEIGHT = 256

IMG_WIDTH = 256


BATCH_SIZE = 8


N_CLASSES = 10


WASTE_RAW_LABEL = 1

WASTE_CLASS_IDX = 0


# =============================================================================
# 5. FINE-TUNING SETTINGS
# =============================================================================

# Maximum duration.
#
# The original training was permitted to run for up to 500 epochs.
MAX_EPOCHS = 500


# Much lower than the default Adam learning rate used when training
# the original model.
INITIAL_LEARNING_RATE = 1e-5


# Do not stop simply because of a few noisy epochs.
EARLY_STOP_PATIENCE = 30


# Reduce LR before deciding training has converged.
REDUCE_LR_PATIENCE = 10


# Required improvement in training loss.
MIN_DELTA = 1e-4


MIN_LEARNING_RATE = 1e-7


# =============================================================================
# 6. TARGETED HARD-NEGATIVE SAMPLING
# =============================================================================

# Each unique new hard-negative tile is presented three times per nominal
# epoch. Because augmentation is stochastic, these repetitions can undergo
# different transformations.
#
# Original ~804
# New = 66
#
# Without weighting:
#
#       new examples ~7.6%
#
# 3x weighting:
#
#       ~198 hard-negative entries
#
# producing about 20% targeted refinement exposure.

HARD_NEGATIVE_REPEAT = 3


# =============================================================================
# 7. RAW LABEL SYSTEM
# =============================================================================

LABEL_LIST = list(
    range(
        1,
        11
    )
)


VALID_RAW_LABEL_SET = set(
    LABEL_LIST
)


LABEL_MAP = {

    raw_label:
        model_index

    for model_index, raw_label
    in enumerate(
        LABEL_LIST
    )

}


INDEX_TO_LABEL = {

    model_index:
        raw_label

    for raw_label, model_index
    in LABEL_MAP.items()

}


CLASS_NAMES = {

    1:
        "Waste",

    2:
        "Soil",

    3:
        "Vegetation",

    4:
        "Built-up surface",

    5:
        "Built-up surface (paved area)",

    6:
        "Water Body",

    7:
        "Natural Surface - rocks",

    8:
        "Non-persistent object",

    9:
        "Ancillary class",

    10:
        "Unidentified"

}


# =============================================================================
# 8. PRINT CLASS MAPPING
# =============================================================================

print("\n")
print("=" * 80)
print("CLASS LABEL MAPPING")
print("=" * 80)


print(
    f"Raw labels: "
    f"{LABEL_LIST}"
)


print(
    f"Raw -> model: "
    f"{LABEL_MAP}"
)


print(
    f"Waste raw label: "
    f"{WASTE_RAW_LABEL}"
)


print(
    f"Waste model channel: "
    f"{WASTE_CLASS_IDX}"
)


# =============================================================================
# 9. CHECK REQUIRED INPUTS
# =============================================================================

print("\n")
print("=" * 80)
print("CHECKING INPUTS")
print("=" * 80)


required_paths = [

    BASE_MODEL_PATH,

    ORIGINAL_IMAGE_DIR,
    ORIGINAL_MASK_DIR,

    NEW_IMAGE_DIR,
    NEW_MASK_DIR,

    NEW_MANIFEST_PATH

]


for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired input not found:\n"
            f"{path}"
        )


print(
    f"\nStarting model:\n"
    f"{BASE_MODEL_PATH}"
)


print(
    f"\nOriginal model-development data:\n"
    f"{ORIGINAL_DATA_DIR}"
)


print(
    f"\nHard-negative fine-tuning data:\n"
    f"{NEW_DATA_DIR}"
)


# =============================================================================
# 10. ENCODE RAW MASK
# =============================================================================

def encode_raw_mask(mask):

    """
    Convert raw raster labels 1-10 to model indices 0-9.
    """

    mapped_mask = np.full(
        mask.shape,
        -1,
        dtype=np.int32
    )


    for raw_label, model_idx in (
        LABEL_MAP.items()
    ):

        mapped_mask[
            mask == raw_label
        ] = model_idx


    if np.any(
        mapped_mask < 0
    ):

        unknown_values = np.unique(
            mask[
                mapped_mask < 0
            ]
        )


        raise ValueError(
            "Mask contains values outside the "
            "expected 1-10 class system: "
            f"{unknown_values.tolist()}"
        )


    return mapped_mask


# =============================================================================
# 11. DISCOVER ORIGINAL IMAGE-MASK PAIRS
# =============================================================================

print("\n")
print("=" * 80)
print("DISCOVERING ORIGINAL DEVELOPMENT DATA")
print("=" * 80)


original_images = {

    path.name:
        path

    for path in (
        ORIGINAL_IMAGE_DIR.glob(
            "*.tif"
        )
    )

    if not path.name.endswith(
        ".aux.xml"
    )

}


original_masks = {

    path.name:
        path

    for path in (
        ORIGINAL_MASK_DIR.glob(
            "*.tif"
        )
    )

    if not path.name.endswith(
        ".aux.xml"
    )

}


common_original = sorted(

    set(
        original_images.keys()
    ).intersection(
        original_masks.keys()
    )

)


if not common_original:

    raise FileNotFoundError(
        "No matching original image-mask pairs were found."
    )


missing_original_masks = sorted(

    set(
        original_images.keys()
    )
    - set(
        original_masks.keys()
    )

)


missing_original_images = sorted(

    set(
        original_masks.keys()
    )
    - set(
        original_images.keys()
    )

)


print(
    f"Matching original image-mask pairs: "
    f"{len(common_original):,}"
)


if missing_original_masks:

    print(
        f"Images missing masks: "
        f"{len(missing_original_masks):,}"
    )


if missing_original_images:

    print(
        f"Masks missing images: "
        f"{len(missing_original_images):,}"
    )


# =============================================================================
# 12. AUDIT ORIGINAL MASKS
#
# This is necessary because at least one current replay mask was found to
# contain raw value 99.
#
# DO NOT map 99 to any model class.
#
# Any incompatible pair is excluded.
# =============================================================================

print("\n")
print("=" * 80)
print("AUDITING ORIGINAL MASK LABELS")
print("=" * 80)


original_records = []

excluded_original_records = []


original_class_pixel_counts = {

    raw_label:
        0

    for raw_label in LABEL_LIST

}


for counter, filename in enumerate(
    common_original,
    start=1
):


    image_path = (
        original_images[
            filename
        ]
    )


    mask_path = (
        original_masks[
            filename
        ]
    )


    with rasterio.open(
        mask_path
    ) as src:

        mask = src.read(
            1
        )


    observed_labels = set(

        int(
            value
        )

        for value in np.unique(
            mask
        )

    )


    invalid_labels = sorted(

        observed_labels
        - VALID_RAW_LABEL_SET

    )


    if invalid_labels:


        excluded_original_records.append(
            {

                "filename":
                    filename,

                "invalid_labels":
                    ",".join(

                        str(
                            value
                        )

                        for value in invalid_labels

                    )

            }
        )


        continue


    # -------------------------------------------------------------------------
    # Valid original pair
    # -------------------------------------------------------------------------

    record = {

        "image_path":
            image_path,

        "mask_path":
            mask_path,

        "source":
            "original",

        "sample_id":
            -1,

        "tile_id":
            filename

    }


    original_records.append(
        record
    )


    # -------------------------------------------------------------------------
    # Count original training classes
    # -------------------------------------------------------------------------

    for raw_label in LABEL_LIST:

        original_class_pixel_counts[
            raw_label
        ] += int(

            np.sum(
                mask == raw_label
            )

        )


    if (
        counter % 100
        == 0
    ):

        print(
            f"\rOriginal masks audited: "
            f"{counter:,} / "
            f"{len(common_original):,}",
            end="",
            flush=True
        )


print("\n")


print(
    f"Original pairs examined: "
    f"{len(common_original):,}"
)


print(
    f"Clean original replay pairs: "
    f"{len(original_records):,}"
)


print(
    f"Excluded incompatible original pairs: "
    f"{len(excluded_original_records):,}"
)


# =============================================================================
# 13. SAVE ORIGINAL MASK AUDIT
# =============================================================================

excluded_audit_path = (
    RESULTS_DIR
    / "excluded_original_masks.csv"
)


excluded_audit_df = pd.DataFrame(
    excluded_original_records
)


excluded_audit_df.to_csv(
    excluded_audit_path,
    index=False
)


if excluded_original_records:

    print("\nExcluded original masks:")


    for record in (
        excluded_original_records
    ):

        print(
            f"  {record['filename']} "
            f"-> invalid labels: "
            f"{record['invalid_labels']}"
        )


# =============================================================================
# 14. LOAD NEW HARD-NEGATIVE MANIFEST
# =============================================================================

print("\n")
print("=" * 80)
print("LOADING NEW HARD-NEGATIVE DATA")
print("=" * 80)


manifest = pd.read_csv(
    NEW_MANIFEST_PATH
)


required_manifest_columns = [

    "tile_id",
    "sample_id",
    "image_file",
    "mask_file"

]


missing_manifest_columns = [

    column

    for column
    in required_manifest_columns

    if column not in manifest.columns

]


if missing_manifest_columns:

    raise ValueError(
        "\nFine-tuning manifest is missing required columns:\n"
        f"{missing_manifest_columns}"
    )


manifest[
    "sample_id"
] = pd.to_numeric(

    manifest[
        "sample_id"
    ],

    errors="raise"

).astype(
    int
)


print(
    f"Manifest image-mask pairs: "
    f"{len(manifest):,}"
)


print(
    f"Source inspection areas represented: "
    f"{manifest['sample_id'].nunique()}"
)


print(
    f"Source area IDs: "
    f"{sorted(manifest['sample_id'].unique().tolist())}"
)


# =============================================================================
# 15. AUDIT NEW HARD-NEGATIVE MASKS
# =============================================================================

new_records = []


new_class_pixel_counts = {

    raw_label:
        0

    for raw_label in LABEL_LIST

}


for _, row in (
    manifest.iterrows()
):


    image_path = (
        NEW_IMAGE_DIR
        / str(
            row[
                "image_file"
            ]
        )
    )


    mask_path = (
        NEW_MASK_DIR
        / str(
            row[
                "mask_file"
            ]
        )
    )


    if not image_path.exists():

        raise FileNotFoundError(
            f"\nNew training image not found:\n"
            f"{image_path}"
        )


    if not mask_path.exists():

        raise FileNotFoundError(
            f"\nNew training mask not found:\n"
            f"{mask_path}"
        )


    with rasterio.open(
        mask_path
    ) as src:

        mask = src.read(
            1
        )


    observed_labels = set(

        int(
            value
        )

        for value in np.unique(
            mask
        )

    )


    invalid_labels = sorted(

        observed_labels
        - VALID_RAW_LABEL_SET

    )


    if invalid_labels:

        raise ValueError(
            "\nA NEW hard-negative training mask contains "
            "values outside the 1-10 class system.\n\n"
            f"Mask:\n{mask_path}\n\n"
            f"Invalid values: {invalid_labels}"
        )


    # -------------------------------------------------------------------------
    # Record
    # -------------------------------------------------------------------------

    new_records.append(
        {

            "image_path":
                image_path,

            "mask_path":
                mask_path,

            "source":
                "hard_negative",

            "sample_id":
                int(
                    row[
                        "sample_id"
                    ]
                ),

            "tile_id":
                str(
                    row[
                        "tile_id"
                    ]
                )

        }
    )


    # -------------------------------------------------------------------------
    # Count class composition
    # -------------------------------------------------------------------------

    for raw_label in LABEL_LIST:

        new_class_pixel_counts[
            raw_label
        ] += int(

            np.sum(
                mask == raw_label
            )

        )


print(
    f"Clean new hard-negative pairs: "
    f"{len(new_records):,}"
)


# =============================================================================
# 16. CLASS COMPOSITION BEFORE OVERSAMPLING
# =============================================================================

print("\n")
print("=" * 80)
print("NEW HARD-NEGATIVE CLASS COMPOSITION")
print("=" * 80)


new_total_pixels = sum(
    new_class_pixel_counts.values()
)


for raw_label in LABEL_LIST:


    pixel_count = (
        new_class_pixel_counts[
            raw_label
        ]
    )


    percentage = (

        100.0
        * pixel_count
        / new_total_pixels

        if new_total_pixels > 0

        else 0.0

    )


    print(

        f"{raw_label:2d}  "
        f"{CLASS_NAMES[raw_label]:35s} "
        f"{pixel_count:12,d} pixels "
        f"({percentage:6.2f}%)"

    )


# =============================================================================
# 17. CREATE FINE-TUNING DATASET
#
# All original data +
# three presentations of every new hard-negative tile.
# =============================================================================

hard_negative_training_records = (

    new_records
    * HARD_NEGATIVE_REPEAT

)


training_records = (

    original_records
    + hard_negative_training_records

)


# Shuffle initial order reproducibly.
random.shuffle(
    training_records
)


print("\n")
print("=" * 80)
print("FINAL FINE-TUNING DATASET")
print("=" * 80)


print(
    f"Clean original replay pairs: "
    f"{len(original_records):,}"
)


print(
    f"Unique new hard-negative pairs: "
    f"{len(new_records):,}"
)


print(
    f"Hard-negative repeat factor: "
    f"{HARD_NEGATIVE_REPEAT}x"
)


print(
    f"Effective hard-negative entries per epoch: "
    f"{len(hard_negative_training_records):,}"
)


print(
    f"TOTAL effective training records per epoch: "
    f"{len(training_records):,}"
)


if len(training_records) > 0:

    hardnegative_fraction = (

        len(
            hard_negative_training_records
        )

        / len(
            training_records
        )

    )


    print(
        f"Targeted hard-negative share per epoch: "
        f"{hardnegative_fraction * 100:.2f}%"
    )


# =============================================================================
# 18. SAVE DATASET MANIFEST
# =============================================================================

dataset_manifest_records = []


for record_number, record in enumerate(
    training_records,
    start=1
):


    dataset_manifest_records.append(
        {

            "record_number":
                record_number,

            "tile_id":
                record[
                    "tile_id"
                ],

            "source":
                record[
                    "source"
                ],

            "sample_id":
                record[
                    "sample_id"
                ],

            "image_path":
                str(
                    record[
                        "image_path"
                    ]
                ),

            "mask_path":
                str(
                    record[
                        "mask_path"
                    ]
                )

        }
    )


dataset_manifest_path = (
    RESULTS_DIR
    / "finetuning_dataset_manifest.csv"
)


pd.DataFrame(
    dataset_manifest_records
).to_csv(
    dataset_manifest_path,
    index=False
)


# =============================================================================
# 19. DATA GENERATOR
#
# Closely follows the original model-development generator.
# =============================================================================

class FineTuneGenerator(
    Sequence
):


    def __init__(
        self,
        records,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True,
        **kwargs
    ):


        super().__init__(
            **kwargs
        )


        self.records = list(
            records
        )


        self.batch_size = (
            batch_size
        )


        self.shuffle = (
            shuffle
        )


        self.augment = (
            augment
        )


        self.indices = np.arange(
            len(
                self.records
            )
        )


        self.on_epoch_end()


    # =========================================================================
    # NUMBER OF BATCHES
    # =========================================================================

    def __len__(
        self
    ):


        return int(

            np.ceil(

                len(
                    self.indices
                )

                / self.batch_size

            )

        )


    # =========================================================================
    # SHUFFLE
    # =========================================================================

    def on_epoch_end(
        self
    ):


        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


    # =========================================================================
    # LOAD BATCH
    # =========================================================================

    def __getitem__(
        self,
        idx
    ):


        batch_indices = self.indices[

            idx
            * self.batch_size
            :

            (
                idx + 1
            )
            * self.batch_size

        ]


        images = []

        masks = []


        for record_index in (
            batch_indices
        ):


            record = (
                self.records[
                    record_index
                ]
            )


            # =================================================================
            # LOAD RGB IMAGE
            # =================================================================

            with rasterio.open(
                record[
                    "image_path"
                ]
            ) as src:


                image = (
                    src.read(
                        [1, 2, 3]
                    )
                    .transpose(
                        1,
                        2,
                        0
                    )
                    .astype(
                        np.float32
                    )
                )


            # =================================================================
            # LOAD MASK
            # =================================================================

            with rasterio.open(
                record[
                    "mask_path"
                ]
            ) as src:


                mask = src.read(
                    1
                )


            # =================================================================
            # RESIZE
            #
            # Retained for consistency with original training script.
            # Most current chips are already 256 x 256.
            # =================================================================

            image = resize(

                image,

                (
                    IMG_HEIGHT,
                    IMG_WIDTH
                ),

                preserve_range=True

            )


            mask = resize(

                mask,

                (
                    IMG_HEIGHT,
                    IMG_WIDTH
                ),

                order=0,

                preserve_range=True,

                anti_aliasing=False

            )


            mask = np.rint(
                mask
            ).astype(
                np.int32
            )


            # =================================================================
            # DEFENSIVE LABEL CHECK
            # =================================================================

            observed_labels = set(

                int(
                    value
                )

                for value in np.unique(
                    mask
                )

            )


            invalid_labels = sorted(

                observed_labels
                - VALID_RAW_LABEL_SET

            )


            if invalid_labels:

                raise ValueError(
                    "\nUnexpected mask class encountered "
                    "during fine-tuning.\n"
                    f"Mask:\n{record['mask_path']}\n"
                    f"Invalid values: {invalid_labels}"
                )


            # =================================================================
            # AUGMENTATION
            #
            # Same structure as original training:
            #
            # - random vertical flip
            # - random rotation between -15 and +15 degrees
            # =================================================================

            if self.augment:


                # -------------------------------------------------------------
                # Vertical flip
                # -------------------------------------------------------------

                if random.random() < 0.5:


                    image = np.flipud(
                        image
                    )


                    mask = np.flipud(
                        mask
                    )


                # -------------------------------------------------------------
                # Rotation
                # -------------------------------------------------------------

                if random.random() < 0.5:


                    angle = random.uniform(
                        -15,
                        15
                    )


                    image = rotate(

                        image,

                        angle,

                        reshape=False,

                        order=1,

                        mode="reflect"

                    )


                    mask = rotate(

                        mask,

                        angle,

                        reshape=False,

                        order=0,

                        mode="nearest"

                    )


                    mask = np.rint(
                        mask
                    ).astype(
                        np.int32
                    )


            # =================================================================
            # NORMALISE IMAGE
            #
            # Same preprocessing used during original training.
            # =================================================================

            image = (

                image
                / 255.0

            ).astype(
                np.float32
            )


            # =================================================================
            # RAW LABEL 1-10 -> MODEL CHANNEL 0-9
            # =================================================================

            mapped_mask = encode_raw_mask(
                mask
            )


            one_hot_mask = to_categorical(

                mapped_mask,

                num_classes=N_CLASSES

            ).astype(
                np.float32
            )


            images.append(
                image
            )


            masks.append(
                one_hot_mask
            )


        return (

            np.asarray(
                images,
                dtype=np.float32
            ),

            np.asarray(
                masks,
                dtype=np.float32
            )

        )


# =============================================================================
# 20. TRAINING GENERATOR
# =============================================================================

train_gen = FineTuneGenerator(

    training_records,

    batch_size=BATCH_SIZE,

    shuffle=True,

    augment=True

)


print("\n")
print(
    f"Training batches per epoch: "
    f"{len(train_gen):,}"
)


# =============================================================================
# 21. WASTE-SPECIFIC TRAINING METRICS
#
# These metrics are optimization diagnostics only.
#
# They are NOT the final test-area performance estimates.
# =============================================================================

def precision_waste(
    y_true,
    y_pred,
    smooth=1e-7
):


    y_true_waste = tf.cast(

        y_true[
            ...,
            WASTE_CLASS_IDX
        ],

        tf.float32

    )


    y_pred_waste = tf.cast(

        tf.argmax(
            y_pred,
            axis=-1
        )

        == WASTE_CLASS_IDX,

        tf.float32

    )


    tp = tf.reduce_sum(

        y_true_waste
        * y_pred_waste

    )


    fp = tf.reduce_sum(

        (
            1.0
            - y_true_waste
        )

        * y_pred_waste

    )


    return (

        tp
        + smooth

    ) / (

        tp
        + fp
        + smooth

    )


def recall_waste(
    y_true,
    y_pred,
    smooth=1e-7
):


    y_true_waste = tf.cast(

        y_true[
            ...,
            WASTE_CLASS_IDX
        ],

        tf.float32

    )


    y_pred_waste = tf.cast(

        tf.argmax(
            y_pred,
            axis=-1
        )

        == WASTE_CLASS_IDX,

        tf.float32

    )


    tp = tf.reduce_sum(

        y_true_waste
        * y_pred_waste

    )


    fn = tf.reduce_sum(

        y_true_waste

        * (
            1.0
            - y_pred_waste
        )

    )


    return (

        tp
        + smooth

    ) / (

        tp
        + fn
        + smooth

    )


def f1_waste(
    y_true,
    y_pred,
    smooth=1e-7
):


    precision = precision_waste(
        y_true,
        y_pred,
        smooth
    )


    recall = recall_waste(
        y_true,
        y_pred,
        smooth
    )


    return (

        2.0
        * precision
        * recall

    ) / (

        precision
        + recall
        + smooth

    )


def iou_waste(
    y_true,
    y_pred,
    smooth=1e-7
):


    y_true_waste = tf.cast(

        y_true[
            ...,
            WASTE_CLASS_IDX
        ],

        tf.float32

    )


    y_pred_waste = tf.cast(

        tf.argmax(
            y_pred,
            axis=-1
        )

        == WASTE_CLASS_IDX,

        tf.float32

    )


    intersection = tf.reduce_sum(

        y_true_waste
        * y_pred_waste

    )


    union = (

        tf.reduce_sum(
            y_true_waste
        )

        + tf.reduce_sum(
            y_pred_waste
        )

        - intersection

    )


    return (

        intersection
        + smooth

    ) / (

        union
        + smooth

    )


# =============================================================================
# 22. LEARNING RATE HISTORY CALLBACK
# =============================================================================

class LearningRateHistory(
    Callback
):


    def __init__(
        self
    ):


        super().__init__()


        self.learning_rates = []


    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):


        learning_rate = float(

            tf.keras.backend.get_value(
                self.model.optimizer.learning_rate
            )

        )


        self.learning_rates.append(
            learning_rate
        )


        print(
            f" - learning_rate: "
            f"{learning_rate:.8g}"
        )


lr_history = LearningRateHistory()


# =============================================================================
# 23. LOAD EXISTING MODEL
# =============================================================================

print("\n")
print("=" * 80)
print("LOADING EXISTING U-NET")
print("=" * 80)


tf.keras.backend.clear_session()


model = load_model(

    BASE_MODEL_PATH,

    compile=False

)


print(
    f"Model input shape: "
    f"{model.input_shape}"
)


print(
    f"Model output shape: "
    f"{model.output_shape}"
)


# =============================================================================
# 24. VERIFY MODEL COMPATIBILITY
# =============================================================================

if tuple(
    model.input_shape[
        1:
    ]
) != (
    IMG_HEIGHT,
    IMG_WIDTH,
    3
):

    raise ValueError(
        "\nUnexpected model input dimensions.\n"
        f"Expected: (256, 256, 3)\n"
        f"Found: {model.input_shape}"
    )


if int(
    model.output_shape[
        -1
    ]
) != N_CLASSES:

    raise ValueError(
        "\nLoaded model is not the expected "
        "10-class multiclass U-Net.\n"
        f"Output shape: {model.output_shape}"
    )


print(
    "\nModel architecture is compatible "
    "with current training masks."
)


# =============================================================================
# 25. MAKE COMPLETE MODEL TRAINABLE
# =============================================================================

for layer in model.layers:

    layer.trainable = True


n_trainable = int(

    np.sum(
        [
            np.prod(
                variable.shape
            )

            for variable
            in model.trainable_weights
        ]
    )

)


print(
    f"Trainable parameters: "
    f"{n_trainable:,}"
)


# =============================================================================
# 26. COMPILE FOR FINE-TUNING
# =============================================================================

optimizer = (
    tf.keras.optimizers.Adam(

        learning_rate=INITIAL_LEARNING_RATE

    )
)


model.compile(

    optimizer=optimizer,

    loss="categorical_crossentropy",

    metrics=[

        "accuracy",

        precision_waste,

        recall_waste,

        f1_waste,

        iou_waste

    ]

)


# =============================================================================
# 27. CALLBACKS
#
# IMPORTANT:
#
# No val_loss exists in this training design.
#
# These callbacks monitor TRAINING LOSS only.
# =============================================================================

early_stop = EarlyStopping(

    monitor="loss",

    mode="min",

    min_delta=MIN_DELTA,

    patience=EARLY_STOP_PATIENCE,

    restore_best_weights=True,

    verbose=1

)


reduce_lr = ReduceLROnPlateau(

    monitor="loss",

    mode="min",

    factor=0.5,

    patience=REDUCE_LR_PATIENCE,

    min_delta=MIN_DELTA,

    min_lr=MIN_LEARNING_RATE,

    verbose=1

)


checkpoint = ModelCheckpoint(

    filepath=str(
        CHECKPOINT_MODEL_PATH
    ),

    monitor="loss",

    mode="min",

    save_best_only=True,

    verbose=1

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
# 28. PRINT TRAINING DESIGN
# =============================================================================

print("\n")
print("=" * 80)
print("FINE-TUNING DESIGN")
print("=" * 80)


print(
    f"Starting model:\n"
    f"{BASE_MODEL_PATH}"
)


print(
    f"\nOriginal replay pairs: "
    f"{len(original_records):,}"
)


print(
    f"Unique hard-negative pairs: "
    f"{len(new_records):,}"
)


print(
    f"Hard-negative repeat factor: "
    f"{HARD_NEGATIVE_REPEAT}x"
)


print(
    f"Effective hard-negative records: "
    f"{len(hard_negative_training_records):,}"
)


print(
    f"Effective total records / epoch: "
    f"{len(training_records):,}"
)


print(
    f"Batches / epoch: "
    f"{len(train_gen):,}"
)


print(
    f"\nInitial learning rate: "
    f"{INITIAL_LEARNING_RATE}"
)


print(
    f"Minimum learning rate: "
    f"{MIN_LEARNING_RATE}"
)


print(
    f"Maximum epochs: "
    f"{MAX_EPOCHS}"
)


print(
    f"LR reduction patience: "
    f"{REDUCE_LR_PATIENCE}"
)


print(
    f"Early-stop patience: "
    f"{EARLY_STOP_PATIENCE}"
)


print(
    f"Minimum loss improvement: "
    f"{MIN_DELTA}"
)


print(
    "\nNO validation data are used."
)


print(
    "The independent test area remains untouched."
)


# =============================================================================
# 29. FINE-TUNE
# =============================================================================

print("\n")
print("=" * 80)
print("STARTING FINE-TUNING")
print("=" * 80)


history = model.fit(

    train_gen,

    epochs=MAX_EPOCHS,

    callbacks=[

        reduce_lr,

        early_stop,

        checkpoint,

        csv_logger,

        terminate_nan,

        lr_history

    ],

    verbose=1

)


# =============================================================================
# 30. NUMBER OF COMPLETED EPOCHS
# =============================================================================

epochs_completed = len(
    history.history[
        "loss"
    ]
)


print("\n")
print(
    f"Training completed after "
    f"{epochs_completed} epochs."
)


# =============================================================================
# 31. LOWEST TRAINING LOSS
# =============================================================================

loss_values = np.asarray(
    history.history[
        "loss"
    ],
    dtype=float
)


lowest_loss_index = int(
    np.argmin(
        loss_values
    )
)


lowest_loss_epoch = (
    lowest_loss_index
    + 1
)


lowest_training_loss = float(
    loss_values[
        lowest_loss_index
    ]
)


print(
    f"Lowest recorded training loss: "
    f"{lowest_training_loss:.6f}"
)


print(
    f"Epoch of lowest training loss: "
    f"{lowest_loss_epoch}"
)


# =============================================================================
# 32. SAVE FINAL MODEL
#
# EarlyStopping restore_best_weights=True means that if stopping occurred,
# the in-memory model has been returned to the lowest monitored training-loss
# weights.
# =============================================================================

print("\n")
print("=" * 80)
print("SAVING FINE-TUNED MODEL")
print("=" * 80)


model.save(
    FINAL_MODEL_PATH
)


print(
    f"\nFine-tuned model saved:\n"
    f"{FINAL_MODEL_PATH}"
)


print(
    f"\nLowest-training-loss checkpoint:\n"
    f"{CHECKPOINT_MODEL_PATH}"
)


# =============================================================================
# 33. SAVE TRAINING HISTORY
# =============================================================================

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
        ) + 1
    )

)


# Add actual learning rates
history_df[
    "learning_rate"
] = lr_history.learning_rates


training_history_path = (
    RESULTS_DIR
    / "training_history.csv"
)


history_df.to_csv(
    training_history_path,
    index=False
)


# =============================================================================
# 34. SAVE CLASS COMPOSITION
# =============================================================================

class_composition_records = []


for raw_label in LABEL_LIST:


    original_pixels = (
        original_class_pixel_counts[
            raw_label
        ]
    )


    new_unique_pixels = (
        new_class_pixel_counts[
            raw_label
        ]
    )


    effective_new_pixels = (

        new_unique_pixels
        * HARD_NEGATIVE_REPEAT

    )


    effective_total_pixels = (

        original_pixels
        + effective_new_pixels

    )


    class_composition_records.append(
        {

            "raw_label":
                raw_label,

            "class_name":
                CLASS_NAMES[
                    raw_label
                ],

            "original_pixels":
                original_pixels,

            "new_unique_pixels":
                new_unique_pixels,

            "hard_negative_repeat":
                HARD_NEGATIVE_REPEAT,

            "new_effective_pixels":
                effective_new_pixels,

            "effective_training_pixels":
                effective_total_pixels

        }
    )


class_composition_df = pd.DataFrame(
    class_composition_records
)


effective_all_pixels = (
    class_composition_df[
        "effective_training_pixels"
    ].sum()
)


if effective_all_pixels > 0:


    class_composition_df[
        "effective_percentage"
    ] = (

        100.0

        * class_composition_df[
            "effective_training_pixels"
        ]

        / effective_all_pixels

    )


class_composition_path = (
    RESULTS_DIR
    / "effective_training_class_composition.csv"
)


class_composition_df.to_csv(
    class_composition_path,
    index=False
)


# =============================================================================
# 35. LOSS CURVE
# =============================================================================

loss_plot_path = (
    RESULTS_DIR
    / "finetuning_training_loss.png"
)


plt.figure(
    figsize=(
        8,
        5
    )
)


plt.plot(

    history_df[
        "epoch"
    ],

    history_df[
        "loss"
    ],

    label="Training loss"

)


plt.axvline(

    lowest_loss_epoch,

    linestyle="--",

    label=(
        f"Lowest loss epoch = "
        f"{lowest_loss_epoch}"
    )

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Categorical cross-entropy loss"
)


plt.title(
    "Hard-negative U-Net fine-tuning"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    loss_plot_path,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# =============================================================================
# 36. TRAINING METRICS PLOT
# =============================================================================

metrics_plot_path = (
    RESULTS_DIR
    / "finetuning_waste_training_metrics.png"
)


metric_columns = [

    "precision_waste",
    "recall_waste",
    "f1_waste",
    "iou_waste"

]


available_metrics = [

    metric

    for metric in metric_columns

    if metric in history_df.columns

]


if available_metrics:


    plt.figure(
        figsize=(
            8,
            5
        )
    )


    for metric in (
        available_metrics
    ):


        plt.plot(

            history_df[
                "epoch"
            ],

            history_df[
                metric
            ],

            label=metric

        )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Training metric"
    )


    plt.title(
        "Waste-class training diagnostics"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        metrics_plot_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


# =============================================================================
# 37. SAVE TRAINING SUMMARY
# =============================================================================

summary_path = (
    RESULTS_DIR
    / "fine_tuning_summary.txt"
)


with open(
    summary_path,
    "w"
) as f:


    f.write(
        "HARD-NEGATIVE U-NET FINE-TUNING SUMMARY\n"
    )


    f.write(
        "=" * 75
        + "\n\n"
    )


    f.write(
        f"Starting model:\n"
        f"{BASE_MODEL_PATH}\n\n"
    )


    f.write(
        f"Output model:\n"
        f"{FINAL_MODEL_PATH}\n\n"
    )


    f.write(
        "TRAINING DATA\n"
    )


    f.write(
        "-" * 75
        + "\n"
    )


    f.write(
        f"Original matching pairs examined: "
        f"{len(common_original)}\n"
    )


    f.write(
        f"Clean original replay pairs: "
        f"{len(original_records)}\n"
    )


    f.write(
        f"Excluded incompatible original masks: "
        f"{len(excluded_original_records)}\n"
    )


    f.write(
        f"Unique new hard-negative pairs: "
        f"{len(new_records)}\n"
    )


    f.write(
        f"Hard-negative source areas represented: "
        f"{manifest['sample_id'].nunique()}\n"
    )


    f.write(
        f"Hard-negative repeat factor: "
        f"{HARD_NEGATIVE_REPEAT}\n"
    )


    f.write(
        f"Effective hard-negative records per epoch: "
        f"{len(hard_negative_training_records)}\n"
    )


    f.write(
        f"Effective total records per epoch: "
        f"{len(training_records)}\n\n"
    )


    f.write(
        "TRAINING SETTINGS\n"
    )


    f.write(
        "-" * 75
        + "\n"
    )


    f.write(
        f"Batch size: "
        f"{BATCH_SIZE}\n"
    )


    f.write(
        f"Initial learning rate: "
        f"{INITIAL_LEARNING_RATE}\n"
    )


    f.write(
        f"Minimum learning rate: "
        f"{MIN_LEARNING_RATE}\n"
    )


    f.write(
        f"Maximum epochs: "
        f"{MAX_EPOCHS}\n"
    )


    f.write(
        f"Reduce LR patience: "
        f"{REDUCE_LR_PATIENCE}\n"
    )


    f.write(
        f"Early stopping patience: "
        f"{EARLY_STOP_PATIENCE}\n"
    )


    f.write(
        f"Minimum loss improvement: "
        f"{MIN_DELTA}\n\n"
    )


    f.write(
        "RESULT\n"
    )


    f.write(
        "-" * 75
        + "\n"
    )


    f.write(
        f"Epochs completed: "
        f"{epochs_completed}\n"
    )


    f.write(
        f"Lowest training-loss epoch: "
        f"{lowest_loss_epoch}\n"
    )


    f.write(
        f"Lowest training loss: "
        f"{lowest_training_loss:.8f}\n"
    )


    final_learning_rate = float(

        lr_history.learning_rates[
            -1
        ]

    )


    f.write(
        f"Final learning rate: "
        f"{final_learning_rate}\n\n"
    )


    f.write(
        "VALIDATION\n"
    )


    f.write(
        "-" * 75
        + "\n"
    )


    f.write(
        "No validation or test data were used during fine-tuning.\n"
    )


    f.write(
        "Generalization performance will subsequently be evaluated "
        "using the separately defined test area.\n"
    )


# =============================================================================
# 38. FINAL CONSOLE REPORT
# =============================================================================

print("\n")
print("=" * 80)
print("FINE-TUNING COMPLETE")
print("=" * 80)


print(
    f"\nEpochs completed: "
    f"{epochs_completed}"
)


print(
    f"Epoch of lowest training loss: "
    f"{lowest_loss_epoch}"
)


print(
    f"Lowest training loss: "
    f"{lowest_training_loss:.8f}"
)


print(
    f"Final learning rate: "
    f"{lr_history.learning_rates[-1]:.8g}"
)


print("\n")
print("-" * 80)


print(
    f"Clean original replay pairs: "
    f"{len(original_records):,}"
)


print(
    f"Original pairs excluded: "
    f"{len(excluded_original_records):,}"
)


print(
    f"Unique new hard-negative pairs: "
    f"{len(new_records):,}"
)


print(
    f"Effective hard-negative records: "
    f"{len(hard_negative_training_records):,}"
)


print(
    f"Total records per epoch: "
    f"{len(training_records):,}"
)


print("\n")
print("=" * 80)
print("OUTPUT MODEL")
print("=" * 80)


print(
    FINAL_MODEL_PATH
)


print("\n")
print("Training history:")

print(
    training_history_path
)


print("\n")
print("Training dataset manifest:")

print(
    dataset_manifest_path
)


print("\n")
print("Original-mask audit:")

print(
    excluded_audit_path
)


print("\n")
print("Class composition:")

print(
    class_composition_path
)


print("\n")
print("Training loss plot:")

print(
    loss_plot_path
)


print("\n")
print("Summary:")

print(
    summary_path
)


print("\n")
print("=" * 80)
print("NEXT STEP")
print("=" * 80)


print(
    "\nDo NOT fine-tune again based on the training metrics alone."
)


print(
    "The next step is to run the original and refined models on "
    "the same independent test area and compare their performance."
)


print("\nDone.")