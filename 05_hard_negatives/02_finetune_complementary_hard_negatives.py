# =============================================================================
# FINE-TUNE EXISTING 10-CLASS U-NET WITH ROTATING COMPLEMENTARY HARD NEGATIVES
# =============================================================================
#
# PURPOSE
# -------
# Fine-tune the existing March multiclass U-Net using:
#
#   1. ALL clean original fully labelled model-development chips each epoch.
#      These retain ordinary 10-class categorical cross-entropy supervision.
#
#   2. A ROTATING, OBJECT-BALANCED subset of manually verified false-positive
#      chips each epoch.
#
#      For these pixels:
#          1 = definitely NOT waste
#          0 = unknown / ignore
#
#      The complementary loss penalises high P(waste) but does NOT assign the
#      pixel to Soil, Vegetation, Built-up, etc.
#
# WHY ROTATE?
# -----------
# The complementary-negative library now contains >26,000 chips. Feeding the
# whole library into every epoch would allow the negative stream to dominate
# the ~hundreds of original fully labelled replay chips.
#
# This script therefore uses, by default:
#
#      300 Stage-1 rejected OBJECTS per epoch
#      200 Stage-2 explicitly "not_waste" OBJECTS per epoch
#
# One chip is selected per sampled object in a given epoch. Objects are cycled
# without replacement within each source before being reshuffled, so coverage
# of the hard-negative library increases across epochs without making any one
# epoch overwhelmingly negative.
#
# INDEPENDENT TEST AREA
# ---------------------
# The hard-negative creation script already excluded the locked independent
# test AOI. This fine-tuning script does not read or use test data.
#
# NEXT STEP
# ---------
# After fine-tuning, compare the starting model and the refined model on the
# SAME independent test area.
# =============================================================================


# =============================================================================
# 0. ENVIRONMENT
# =============================================================================

import os

# Keep numerical behaviour consistent with the previous workflow.
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


# -----------------------------------------------------------------------------
# Existing trained U-Net
# -----------------------------------------------------------------------------

BASE_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_march_finetuned_best.keras"
)


# -----------------------------------------------------------------------------
# Original fully labelled replay data
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
# Complementary hard-negative library created by the new data-preparation
# script.
# -----------------------------------------------------------------------------

NEGATIVE_DATA_DIR = (
    BASE_PATH
    / "data"
    / "tiled_data"
    / "complementary_hard_negatives_v2"
)

NEGATIVE_IMAGE_DIR = (
    NEGATIVE_DATA_DIR
    / "images"
)

NEGATIVE_MASK_DIR = (
    NEGATIVE_DATA_DIR
    / "negative_masks"
)

NEGATIVE_MANIFEST_PATH = (
    NEGATIVE_DATA_DIR
    / "finetuning_manifest.csv"
)


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------

RESULTS_DIR = (
    BASE_PATH
    / "results"
    / "complementary_hard_negative_finetuning_v2"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------------------------------------------------------
# Output models
# -----------------------------------------------------------------------------

CHECKPOINT_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_complementary_rotating_checkpoint.keras"
)

FINAL_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_complementary_rotating_finetuned.keras"
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


LABEL_LIST = list(
    range(
        1,
        11,
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

# Maximum training duration.
MAX_EPOCHS = 500

# Low learning rate because we are refining an already trained model.
INITIAL_LEARNING_RATE = 1e-5
MIN_LEARNING_RATE = 1e-7

EARLY_STOP_PATIENCE = 30
REDUCE_LR_PATIENCE = 10
MIN_DELTA = 1e-4


# -----------------------------------------------------------------------------
# Complementary-negative sampling per epoch
# -----------------------------------------------------------------------------
#
# The whole >26k-chip library is NOT used in every epoch.
#
# Sampling is by UNIQUE OBJECT (sample_id / focal_candidate_id), not simply by
# chip, so large false-positive objects that produced several chips do not
# automatically receive more weight.
#
# Stage 2 is given relatively strong representation because it reflects direct
# object-level "not_waste" decisions.
# -----------------------------------------------------------------------------

N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH = 300

N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH = 200


# -----------------------------------------------------------------------------
# Complementary negative loss weight
# -----------------------------------------------------------------------------
#
# Start conservatively at 1.0. The independent test-area comparison determines
# whether this reduces commission error without excessive loss of waste recall.
# -----------------------------------------------------------------------------

NEGATIVE_LOSS_WEIGHT = 1.0

EPSILON = 1e-7


# =============================================================================
# 6. CHECK REQUIRED INPUTS
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

        raise FileNotFoundError(
            f"\nRequired input not found:\n"
            f"{path}"
        )


print("\n")
print("=" * 80)
print("ROTATING COMPLEMENTARY HARD-NEGATIVE FINE-TUNING")
print("=" * 80)

print(
    f"\nStarting model:\n"
    f"{BASE_MODEL_PATH}"
)

print(
    f"\nOriginal replay data:\n"
    f"{ORIGINAL_DATA_DIR}"
)

print(
    f"\nComplementary-negative library:\n"
    f"{NEGATIVE_DATA_DIR}"
)

print(
    f"\nStage-1 negatives / epoch: "
    f"{N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH}"
)

print(
    f"Stage-2 negatives / epoch: "
    f"{N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH}"
)

print(
    f"Negative-loss weight: "
    f"{NEGATIVE_LOSS_WEIGHT}"
)


# =============================================================================
# 7. UTILITIES
# =============================================================================

def encode_raw_mask(
    mask
):

    """
    Convert original raw labels 1-10 to model channels 0-9.
    """

    mapped = np.full(
        mask.shape,
        -1,
        dtype=np.int32,
    )


    for raw_label, model_idx in (
        LABEL_MAP.items()
    ):

        mapped[
            mask == raw_label
        ] = model_idx


    if np.any(
        mapped < 0
    ):

        bad = np.unique(
            mask[
                mapped < 0
            ]
        )

        raise ValueError(
            "Original replay mask contains values "
            "outside the expected 1-10 class system: "
            f"{bad.tolist()}"
        )


    return mapped


def validate_negative_mask(
    mask,
    mask_path,
):

    """
    Complementary hard-negative mask:
        0     = unknown / ignore
        1/255 = confirmed NOT waste
    """

    values = set(
        int(
            value
        )
        for value in np.unique(
            mask
        )
    )


    allowed = {
        0,
        1,
        255,
    }


    invalid = sorted(
        values
        - allowed
    )


    if invalid:

        raise ValueError(
            "\nInvalid complementary hard-negative mask.\n"
            f"Mask:\n{mask_path}\n"
            f"Observed values: {sorted(values)}\n"
            f"Invalid values: {invalid}\n\n"
            "Expected only:\n"
            "  0 = unknown / ignore\n"
            "  1 or 255 = confirmed NOT waste"
        )


    binary = (
        mask > 0
    ).astype(
        np.float32
    )


    if np.sum(
        binary
    ) == 0:

        raise ValueError(
            "\nComplementary hard-negative mask contains "
            "no confirmed negative pixels:\n"
            f"{mask_path}"
        )


    return binary


def read_rgb(
    path
):

    with rasterio.open(
        path
    ) as src:

        image = (
            src.read(
                [
                    1,
                    2,
                    3,
                ]
            )
            .transpose(
                1,
                2,
                0,
            )
            .astype(
                np.float32
            )
        )


    image = resize(
        image,
        (
            IMG_HEIGHT,
            IMG_WIDTH,
        ),
        preserve_range=True,
    )


    # Retain the preprocessing used by the original model.
    image = (
        image
        / 255.0
    ).astype(
        np.float32
    )


    return image


def read_original_mask(
    path
):

    with rasterio.open(
        path
    ) as src:

        mask = src.read(
            1
        )


    mask = resize(
        mask,
        (
            IMG_HEIGHT,
            IMG_WIDTH,
        ),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )


    mask = np.rint(
        mask
    ).astype(
        np.int32
    )


    return mask


def read_negative_mask(
    path
):

    with rasterio.open(
        path
    ) as src:

        mask = src.read(
            1
        )


    mask = resize(
        mask,
        (
            IMG_HEIGHT,
            IMG_WIDTH,
        ),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )


    mask = np.rint(
        mask
    ).astype(
        np.int32
    )


    return validate_negative_mask(
        mask,
        path,
    )


def augment_pair(
    image,
    target_mask,
):

    """
    Apply exactly the same geometric augmentation to the image and target.
    """

    # Vertical flip.
    if random.random() < 0.5:

        image = np.flipud(
            image
        )

        target_mask = np.flipud(
            target_mask
        )


    # Small random rotation.
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


    return (
        image,
        target_mask,
    )


# =============================================================================
# 8. DISCOVER / AUDIT ORIGINAL FULLY LABELLED REPLAY DATA
# =============================================================================

print("\n")
print("=" * 80)
print("AUDITING ORIGINAL REPLAY DATA")
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
        "No matching original image-mask pairs found."
    )


original_records = []

excluded_original_records = []


for counter, filename in enumerate(
    common_original,
    start=1,
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


    observed = set(

        int(
            value
        )

        for value in np.unique(
            mask
        )

    )


    invalid = sorted(

        observed
        - VALID_RAW_LABEL_SET

    )


    if invalid:

        excluded_original_records.append(
            {
                "filename":
                    filename,

                "invalid_labels":
                    ",".join(
                        str(
                            value
                        )
                        for value
                        in invalid
                    ),
            }
        )

        continue


    original_records.append(
        {
            "image_path":
                image_path,

            "mask_path":
                mask_path,

            "negative_mask_path":
                None,

            "source":
                "original",

            "negative_source":
                None,

            "sample_id":
                f"original::{filename}",

            "tile_id":
                filename,
        }
    )


print(
    f"Original matching pairs examined: "
    f"{len(common_original):,}"
)

print(
    f"Clean original replay pairs: "
    f"{len(original_records):,}"
)

print(
    f"Excluded incompatible original masks: "
    f"{len(excluded_original_records):,}"
)


pd.DataFrame(
    excluded_original_records
).to_csv(
    RESULTS_DIR
    / "excluded_original_masks.csv",
    index=False,
)


# =============================================================================
# 9. LOAD COMPLEMENTARY HARD-NEGATIVE MANIFEST
# =============================================================================

print("\n")
print("=" * 80)
print("LOADING COMPLEMENTARY HARD-NEGATIVE LIBRARY")
print("=" * 80)


manifest = pd.read_csv(
    NEGATIVE_MANIFEST_PATH,
    dtype=str,
)


# -----------------------------------------------------------------------------
# Required fields
# -----------------------------------------------------------------------------

required_manifest_columns = [
    "tile_id",
    "image_file",
    "negative_mask_file",
]


missing_manifest_columns = [

    column

    for column
    in required_manifest_columns

    if column
    not in manifest.columns

]


if missing_manifest_columns:

    raise ValueError(
        "\nHard-negative manifest is missing required columns:\n"
        f"{missing_manifest_columns}"
    )


# -----------------------------------------------------------------------------
# Resolve object identifier.
#
# The v2 creation script contains both sample_id and focal_candidate_id.
# Prefer focal_candidate_id when available.
# -----------------------------------------------------------------------------

if "focal_candidate_id" in manifest.columns:

    manifest[
        "object_id"
    ] = manifest[
        "focal_candidate_id"
    ].astype(
        str
    )

elif "sample_id" in manifest.columns:

    manifest[
        "object_id"
    ] = manifest[
        "sample_id"
    ].astype(
        str
    )

else:

    raise ValueError(
        "\nManifest needs either "
        "'focal_candidate_id' or 'sample_id'."
    )


# -----------------------------------------------------------------------------
# Resolve source.
# -----------------------------------------------------------------------------

if "focal_source" in manifest.columns:

    manifest[
        "negative_source"
    ] = manifest[
        "focal_source"
    ].astype(
        str
    )

elif "negative_source" in manifest.columns:

    manifest[
        "negative_source"
    ] = manifest[
        "negative_source"
    ].astype(
        str
    )

elif "source_type" in manifest.columns:

    # Compatibility with an earlier preparation script.
    source_map = {
        "stage1_rejected_full_negative":
            "stage1_rejected",

        "stage2_not_waste_partial_negative":
            "stage2_not_waste",
    }

    manifest[
        "negative_source"
    ] = (
        manifest[
            "source_type"
        ]
        .map(
            source_map
        )
        .fillna(
            manifest[
                "source_type"
            ]
        )
        .astype(
            str
        )
    )

else:

    raise ValueError(
        "\nManifest does not contain a source field.\n"
        "Expected one of: focal_source, negative_source, source_type."
    )


# -----------------------------------------------------------------------------
# Keep only the two intended negative sources.
# -----------------------------------------------------------------------------

allowed_negative_sources = {
    "stage1_rejected",
    "stage2_not_waste",
}


unexpected_sources = sorted(

    set(
        manifest[
            "negative_source"
        ].dropna()
    )
    - allowed_negative_sources

)


if unexpected_sources:

    print(
        "\nWARNING: manifest contains additional source labels:"
    )

    print(
        unexpected_sources
    )

    print(
        "Those rows will not be used for complementary fine-tuning."
    )


manifest = manifest[
    manifest[
        "negative_source"
    ].isin(
        allowed_negative_sources
    )
].copy()


if len(
    manifest
) == 0:

    raise ValueError(
        "No usable Stage-1 / Stage-2 hard negatives remain."
    )


# =============================================================================
# 10. BUILD HARD-NEGATIVE RECORD LIBRARY
# =============================================================================

negative_records = []


for _, row in (
    manifest.iterrows()
):

    image_path = (
        NEGATIVE_IMAGE_DIR
        / str(
            row[
                "image_file"
            ]
        )
    )


    negative_mask_path = (
        NEGATIVE_MASK_DIR
        / str(
            row[
                "negative_mask_file"
            ]
        )
    )


    if not image_path.exists():

        raise FileNotFoundError(
            "\nHard-negative image listed in manifest "
            "does not exist:\n"
            f"{image_path}"
        )


    if not negative_mask_path.exists():

        raise FileNotFoundError(
            "\nHard-negative mask listed in manifest "
            "does not exist:\n"
            f"{negative_mask_path}"
        )


    negative_records.append(
        {
            "image_path":
                image_path,

            "mask_path":
                None,

            "negative_mask_path":
                negative_mask_path,

            "source":
                "hard_negative",

            "negative_source":
                str(
                    row[
                        "negative_source"
                    ]
                ),

            "sample_id":
                str(
                    row[
                        "object_id"
                    ]
                ),

            "tile_id":
                str(
                    row[
                        "tile_id"
                    ]
                ),
        }
    )


negative_library_df = pd.DataFrame(
    negative_records
)


# -----------------------------------------------------------------------------
# Source / object counts
# -----------------------------------------------------------------------------

print(
    f"Hard-negative chips available: "
    f"{len(negative_library_df):,}"
)


for source_name in [
    "stage1_rejected",
    "stage2_not_waste",
]:

    subset = negative_library_df[
        negative_library_df[
            "negative_source"
        ] == source_name
    ]

    print(
        f"{source_name}: "
        f"{len(subset):,} chips from "
        f"{subset['sample_id'].nunique():,} unique objects"
    )


# -----------------------------------------------------------------------------
# Save library audit
# -----------------------------------------------------------------------------

library_audit_path = (
    RESULTS_DIR
    / "hard_negative_library_audit.csv"
)


negative_library_df[
    [
        "sample_id",
        "tile_id",
        "negative_source",
        "image_path",
        "negative_mask_path",
    ]
].to_csv(
    library_audit_path,
    index=False,
)


# =============================================================================
# 11. QUICK MASK AUDIT
#
# Validate all masks that will ever enter training lazily when they are loaded.
# Here we additionally inspect a reproducible sample before training starts.
# =============================================================================

print("\n")
print("=" * 80)
print("PRE-FLIGHT COMPLEMENTARY-MASK AUDIT")
print("=" * 80)


audit_n = min(
    200,
    len(
        negative_library_df
    ),
)


audit_rows = negative_library_df.sample(
    n=audit_n,
    random_state=SEED,
)


audit_negative_pixels = []


for _, row in (
    audit_rows.iterrows()
):

    with rasterio.open(
        row[
            "negative_mask_path"
        ]
    ) as src:

        audit_mask = src.read(
            1
        )


    binary = validate_negative_mask(
        audit_mask,
        row[
            "negative_mask_path"
        ],
    )


    audit_negative_pixels.append(
        int(
            np.sum(
                binary
            )
        )
    )


print(
    f"Random masks audited: "
    f"{audit_n:,}"
)

print(
    f"Median confirmed-negative pixels in audit: "
    f"{np.median(audit_negative_pixels):,.0f}"
)

print(
    "Mask audit passed."
)


# =============================================================================
# 12. CYCLING OBJECT SAMPLER
# =============================================================================

class CyclingObjectSampler:

    """
    Cycle through unique hard-negative object IDs without replacement.

    Once every object in a source has been consumed, reshuffle that source
    and begin another cycle.

    If an object has multiple image chips, one of its chips is chosen randomly
    each time that object is selected.
    """

    def __init__(
        self,
        records_df,
        source_name,
        seed,
    ):

        self.source_name = (
            source_name
        )


        self.df = records_df[
            records_df[
                "negative_source"
            ] == source_name
        ].copy()


        if len(
            self.df
        ) == 0:

            raise ValueError(
                f"No hard-negative records found for "
                f"source '{source_name}'."
            )


        self.rng = np.random.default_rng(
            seed
        )


        self.object_to_indices = {

            str(
                object_id
            ):
                group.index.to_numpy()

            for object_id, group
            in self.df.groupby(
                "sample_id"
            )

        }


        self.object_ids = np.array(
            list(
                self.object_to_indices.keys()
            ),
            dtype=object,
        )


        self.order = np.array(
            [],
            dtype=object,
        )

        self.position = 0

        self.cycle_number = 0

        self._reshuffle()


    def _reshuffle(
        self
    ):

        self.order = (
            self.object_ids.copy()
        )


        self.rng.shuffle(
            self.order
        )


        self.position = 0

        self.cycle_number += 1


    def sample(
        self,
        n_objects,
    ):

        if n_objects <= 0:

            return []


        selected_object_ids = []


        while (
            len(
                selected_object_ids
            )
            < n_objects
        ):

            remaining = (
                len(
                    self.order
                )
                - self.position
            )


            needed = (
                n_objects
                - len(
                    selected_object_ids
                )
            )


            take = min(
                remaining,
                needed,
            )


            if take > 0:

                selected_object_ids.extend(
                    self.order[
                        self.position:
                        self.position
                        + take
                    ].tolist()
                )


                self.position += take


            if (
                len(
                    selected_object_ids
                )
                < n_objects
            ):

                self._reshuffle()


        selected_records = []


        for object_id in (
            selected_object_ids
        ):

            candidate_indices = (
                self.object_to_indices[
                    str(
                        object_id
                    )
                ]
            )


            selected_index = int(
                self.rng.choice(
                    candidate_indices
                )
            )


            row = self.df.loc[
                selected_index
            ]


            selected_records.append(
                {
                    "image_path":
                        row[
                            "image_path"
                        ],

                    "mask_path":
                        None,

                    "negative_mask_path":
                        row[
                            "negative_mask_path"
                        ],

                    "source":
                        "hard_negative",

                    "negative_source":
                        row[
                            "negative_source"
                        ],

                    "sample_id":
                        str(
                            row[
                                "sample_id"
                            ]
                        ),

                    "tile_id":
                        str(
                            row[
                                "tile_id"
                            ]
                        ),
                }
            )


        return selected_records


# =============================================================================
# 13. MIXED-SUPERVISION ROTATING GENERATOR
# =============================================================================
#
# y_true contains 11 channels:
#
#   channels 0-9 : ordinary one-hot 10-class supervision
#   channel 10   : complementary NOT-WASTE mask
#
# Original replay example:
#   channels 0-9 = one-hot class labels
#   channel 10   = 0
#
# Complementary negative example:
#   channels 0-9 = 0 (true alternative class unknown)
#   channel 10   = 1 only for confirmed NOT-waste pixels
# =============================================================================

class RotatingComplementaryGenerator(
    Sequence
):

    def __init__(
        self,
        original_records,
        negative_library_df,
        batch_size=BATCH_SIZE,
        n_stage1=N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH,
        n_stage2=N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH,
        shuffle=True,
        augment=True,
        seed=SEED,
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )


        self.original_records = list(
            original_records
        )

        self.negative_library_df = (
            negative_library_df.copy()
        )

        self.batch_size = (
            batch_size
        )

        self.n_stage1 = (
            n_stage1
        )

        self.n_stage2 = (
            n_stage2
        )

        self.shuffle = (
            shuffle
        )

        self.augment = (
            augment
        )

        self.seed = (
            seed
        )

        self.epoch_number = 0


        self.stage1_sampler = (
            CyclingObjectSampler(
                self.negative_library_df,
                "stage1_rejected",
                seed
                + 1001,
            )
        )


        self.stage2_sampler = (
            CyclingObjectSampler(
                self.negative_library_df,
                "stage2_not_waste",
                seed
                + 2001,
            )
        )


        self.current_records = []

        self.indices = np.array(
            [],
            dtype=int,
        )


        self.selection_history = []


        # Prepare epoch 1.
        self._prepare_epoch()


    def _prepare_epoch(
        self
    ):

        self.epoch_number += 1


        stage1_records = (
            self.stage1_sampler.sample(
                self.n_stage1
            )
        )


        stage2_records = (
            self.stage2_sampler.sample(
                self.n_stage2
            )
        )


        selected_negative_records = (
            stage1_records
            + stage2_records
        )


        # Log exactly which negative objects/chips were selected.
        for record in (
            selected_negative_records
        ):

            self.selection_history.append(
                {
                    "epoch":
                        self.epoch_number,

                    "negative_source":
                        record[
                            "negative_source"
                        ],

                    "sample_id":
                        record[
                            "sample_id"
                        ],

                    "tile_id":
                        record[
                            "tile_id"
                        ],
                }
            )


        self.current_records = (
            list(
                self.original_records
            )
            + selected_negative_records
        )


        self.indices = np.arange(
            len(
                self.current_records
            )
        )


        if self.shuffle:

            rng = np.random.default_rng(
                self.seed
                + 100000
                + self.epoch_number
            )


            rng.shuffle(
                self.indices
            )


        if (
            self.epoch_number == 1
            or self.epoch_number % 10 == 0
        ):

            print(
                "\n"
                f"[Epoch pool {self.epoch_number}] "
                f"original={len(self.original_records):,}, "
                f"stage1_neg={len(stage1_records):,}, "
                f"stage2_neg={len(stage2_records):,}, "
                f"total={len(self.current_records):,}"
            )


    def __len__(
        self
    ):

        return int(
            math.ceil(
                len(
                    self.current_records
                )
                / self.batch_size
            )
        )


    def on_epoch_end(
        self
    ):

        # Keras calls this after each completed epoch.
        self._prepare_epoch()


    def __getitem__(
        self,
        idx,
    ):

        batch_indices = (
            self.indices[
                idx
                * self.batch_size
                :
                (
                    idx
                    + 1
                )
                * self.batch_size
            ]
        )


        images = []

        targets = []


        for record_index in (
            batch_indices
        ):

            record = (
                self.current_records[
                    record_index
                ]
            )


            image = read_rgb(
                record[
                    "image_path"
                ]
            )


            # -----------------------------------------------------------------
            # Original fully labelled replay example
            # -----------------------------------------------------------------

            if (
                record[
                    "source"
                ]
                == "original"
            ):

                mask = read_original_mask(
                    record[
                        "mask_path"
                    ]
                )


                if self.augment:

                    image, mask = (
                        augment_pair(
                            image,
                            mask,
                        )
                    )


                mask = np.rint(
                    mask
                ).astype(
                    np.int32
                )


                mapped = encode_raw_mask(
                    mask
                )


                supervised_target = (
                    to_categorical(
                        mapped,
                        num_classes=N_CLASSES,
                    )
                    .astype(
                        np.float32
                    )
                )


                complementary_negative = (
                    np.zeros(
                        (
                            IMG_HEIGHT,
                            IMG_WIDTH,
                            1,
                        ),
                        dtype=np.float32,
                    )
                )


            # -----------------------------------------------------------------
            # Complementary hard-negative example
            # -----------------------------------------------------------------

            elif (
                record[
                    "source"
                ]
                == "hard_negative"
            ):

                negative_mask = (
                    read_negative_mask(
                        record[
                            "negative_mask_path"
                        ]
                    )
                )


                if self.augment:

                    image, negative_mask = (
                        augment_pair(
                            image,
                            negative_mask,
                        )
                    )


                negative_mask = (
                    negative_mask
                    > 0.5
                ).astype(
                    np.float32
                )


                # Alternative class is unknown.
                supervised_target = (
                    np.zeros(
                        (
                            IMG_HEIGHT,
                            IMG_WIDTH,
                            N_CLASSES,
                        ),
                        dtype=np.float32,
                    )
                )


                complementary_negative = (
                    negative_mask[
                        ...,
                        np.newaxis
                    ].astype(
                        np.float32
                    )
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
                image.astype(
                    np.float32
                )
            )


            targets.append(
                target.astype(
                    np.float32
                )
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


# =============================================================================
# 14. CREATE TRAINING GENERATOR
# =============================================================================

train_gen = (
    RotatingComplementaryGenerator(
        original_records=original_records,
        negative_library_df=negative_library_df,
        batch_size=BATCH_SIZE,
        n_stage1=N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH,
        n_stage2=N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH,
        shuffle=True,
        augment=True,
        seed=SEED,
    )
)


print("\n")
print("=" * 80)
print("PER-EPOCH TRAINING COMPOSITION")
print("=" * 80)

print(
    f"Original fully labelled replay records: "
    f"{len(original_records):,}"
)

print(
    f"Stage-1 complementary negatives: "
    f"{N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH:,}"
)

print(
    f"Stage-2 complementary negatives: "
    f"{N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH:,}"
)

print(
    f"Total records per epoch: "
    f"{len(train_gen.current_records):,}"
)

print(
    f"Hard-negative record share per epoch: "
    f"{100.0 * (N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH + N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH) / len(train_gen.current_records):.2f}%"
)

print(
    f"Training batches per epoch: "
    f"{len(train_gen):,}"
)


# =============================================================================
# 15. COMPLEMENTARY MIXED LOSS
# =============================================================================

@tf.keras.utils.register_keras_serializable(
    package="WasteFineTuning"
)
def complementary_mixed_loss(
    y_true,
    y_pred,
):

    """
    Total loss:

        supervised 10-class CE
        +
        lambda * complementary NOT-WASTE loss

    For confirmed negative pixels:

        L_negative = -log(1 - P(waste))

    Unknown pixels in complementary chips contribute no loss.
    """


    supervised_target = (
        y_true[
            ...,
            :N_CLASSES
        ]
    )


    negative_mask = (
        y_true[
            ...,
            N_CLASSES
        ]
    )


    # -------------------------------------------------------------------------
    # Supervised 10-class component
    # -------------------------------------------------------------------------

    supervised_mask = tf.cast(

        tf.reduce_sum(
            supervised_target,
            axis=-1,
        )
        > 0.5,

        tf.float32,

    )


    per_pixel_ce = (
        tf.keras.losses.categorical_crossentropy(
            supervised_target,
            y_pred,
        )
    )


    supervised_count = (
        tf.reduce_sum(
            supervised_mask
        )
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


    # -------------------------------------------------------------------------
    # Complementary negative component
    # -------------------------------------------------------------------------

    p_waste = tf.clip_by_value(

        y_pred[
            ...,
            WASTE_CLASS_IDX
        ],

        EPSILON,

        1.0
        - EPSILON,

    )


    per_pixel_negative_loss = (
        -tf.math.log(
            1.0
            - p_waste
        )
    )


    negative_count = (
        tf.reduce_sum(
            negative_mask
        )
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
        +
        NEGATIVE_LOSS_WEIGHT
        * negative_loss
    )


# =============================================================================
# 16. EPOCH-LEVEL METRICS
# =============================================================================
#
# These metrics accumulate numerators and denominators across the full epoch.
# This avoids the misleading behaviour of averaging zero-valued metrics from
# batches that contain none of the relevant supervision type.
# =============================================================================

@tf.keras.utils.register_keras_serializable(
    package="WasteFineTuning"
)
class SupervisedPixelAccuracy(
    tf.keras.metrics.Metric
):

    def __init__(
        self,
        name="supervised_accuracy",
        **kwargs,
    ):

        super().__init__(
            name=name,
            **kwargs,
        )

        self.correct = self.add_weight(
            name="correct",
            initializer="zeros",
        )

        self.count = self.add_weight(
            name="count",
            initializer="zeros",
        )


    def update_state(
        self,
        y_true,
        y_pred,
        sample_weight=None,
    ):

        target = (
            y_true[
                ...,
                :N_CLASSES
            ]
        )


        mask = tf.cast(

            tf.reduce_sum(
                target,
                axis=-1,
            )
            > 0.5,

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


        self.correct.assign_add(
            tf.reduce_sum(
                correct
                * mask
            )
        )


        self.count.assign_add(
            tf.reduce_sum(
                mask
            )
        )


    def result(
        self
    ):

        return (
            self.correct
            / tf.maximum(
                self.count,
                1.0,
            )
        )


    def reset_state(
        self
    ):

        self.correct.assign(
            0.0
        )

        self.count.assign(
            0.0
        )


@tf.keras.utils.register_keras_serializable(
    package="WasteFineTuning"
)
class NegativeMeanWasteProbability(
    tf.keras.metrics.Metric
):

    def __init__(
        self,
        name="negative_mean_waste_probability",
        **kwargs,
    ):

        super().__init__(
            name=name,
            **kwargs,
        )

        self.probability_sum = self.add_weight(
            name="probability_sum",
            initializer="zeros",
        )

        self.count = self.add_weight(
            name="count",
            initializer="zeros",
        )


    def update_state(
        self,
        y_true,
        y_pred,
        sample_weight=None,
    ):

        negative_mask = (
            y_true[
                ...,
                N_CLASSES
            ]
        )


        p_waste = (
            y_pred[
                ...,
                WASTE_CLASS_IDX
            ]
        )


        self.probability_sum.assign_add(
            tf.reduce_sum(
                p_waste
                * negative_mask
            )
        )


        self.count.assign_add(
            tf.reduce_sum(
                negative_mask
            )
        )


    def result(
        self
    ):

        return (
            self.probability_sum
            / tf.maximum(
                self.count,
                1.0,
            )
        )


    def reset_state(
        self
    ):

        self.probability_sum.assign(
            0.0
        )

        self.count.assign(
            0.0
        )


@tf.keras.utils.register_keras_serializable(
    package="WasteFineTuning"
)
class NegativeWasteArgmaxRate(
    tf.keras.metrics.Metric
):

    def __init__(
        self,
        name="negative_waste_argmax_rate",
        **kwargs,
    ):

        super().__init__(
            name=name,
            **kwargs,
        )

        self.waste_predictions = self.add_weight(
            name="waste_predictions",
            initializer="zeros",
        )

        self.count = self.add_weight(
            name="count",
            initializer="zeros",
        )


    def update_state(
        self,
        y_true,
        y_pred,
        sample_weight=None,
    ):

        negative_mask = (
            y_true[
                ...,
                N_CLASSES
            ]
        )


        predicted_waste = tf.cast(

            tf.argmax(
                y_pred,
                axis=-1,
            )
            == WASTE_CLASS_IDX,

            tf.float32,

        )


        self.waste_predictions.assign_add(
            tf.reduce_sum(
                predicted_waste
                * negative_mask
            )
        )


        self.count.assign_add(
            tf.reduce_sum(
                negative_mask
            )
        )


    def result(
        self
    ):

        return (
            self.waste_predictions
            / tf.maximum(
                self.count,
                1.0,
            )
        )


    def reset_state(
        self
    ):

        self.waste_predictions.assign(
            0.0
        )

        self.count.assign(
            0.0
        )


# =============================================================================
# 17. LOAD EXISTING MODEL
# =============================================================================

print("\n")
print("=" * 80)
print("LOADING EXISTING U-NET")
print("=" * 80)


tf.keras.backend.clear_session()


model = load_model(
    BASE_MODEL_PATH,
    compile=False,
)


print(
    f"Model input shape: "
    f"{model.input_shape}"
)

print(
    f"Model output shape: "
    f"{model.output_shape}"
)


if tuple(
    model.input_shape[
        1:
    ]
) != (
    IMG_HEIGHT,
    IMG_WIDTH,
    3,
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
# 18. COMPILE
# =============================================================================

optimizer = (
    tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE
    )
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
# 19. CALLBACKS
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
        logs=None,
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


class NegativeSelectionHistoryWriter(
    Callback
):

    """
    Persist the rotating negative selection after every epoch.
    """

    def __init__(
        self,
        generator,
        output_path,
    ):

        super().__init__()

        self.generator = (
            generator
        )

        self.output_path = (
            Path(
                output_path
            )
        )


    def on_epoch_end(
        self,
        epoch,
        logs=None,
    ):

        history_df = pd.DataFrame(
            self.generator.selection_history
        )


        # on_epoch_end() in the Sequence prepares the NEXT epoch.
        # Retain only selection rows up to the epoch just completed.
        completed_epoch = (
            epoch
            + 1
        )


        history_df = history_df[
            history_df[
                "epoch"
            ]
            <= completed_epoch
        ]


        history_df.to_csv(
            self.output_path,
            index=False,
        )


lr_history = (
    LearningRateHistory()
)


negative_selection_history_path = (
    RESULTS_DIR
    / "negative_selection_history.csv"
)


selection_writer = (
    NegativeSelectionHistoryWriter(
        train_gen,
        negative_selection_history_path,
    )
)


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


terminate_nan = (
    TerminateOnNaN()
)


# =============================================================================
# 20. PRINT TRAINING DESIGN
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
    f"\nOriginal replay pairs / epoch: "
    f"{len(original_records):,}"
)


print(
    f"Stage-1 negative objects / epoch: "
    f"{N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH:,}"
)


print(
    f"Stage-2 negative objects / epoch: "
    f"{N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH:,}"
)


print(
    f"Total records / epoch: "
    f"{len(train_gen.current_records):,}"
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
    f"Negative-loss weight: "
    f"{NEGATIVE_LOSS_WEIGHT}"
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
    "\nNegative objects are rotated without replacement "
    "within each source before reshuffling."
)


print(
    "\nNO independent test data are used during fine-tuning."
)


# =============================================================================
# 21. FINE-TUNE
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
        lr_history,
        selection_writer,
    ],

    verbose=1,

)


# =============================================================================
# 22. TRAINING RESULT
# =============================================================================

epochs_completed = len(
    history.history[
        "loss"
    ]
)


loss_values = np.asarray(
    history.history[
        "loss"
    ],
    dtype=float,
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


print("\n")
print(
    f"Training completed after "
    f"{epochs_completed} epoch(s)."
)


print(
    f"Lowest recorded training loss: "
    f"{lowest_training_loss:.8f}"
)


print(
    f"Epoch of lowest training loss: "
    f"{lowest_loss_epoch}"
)


# =============================================================================
# 23. SAVE FINAL MODEL
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
# 24. SAVE TRAINING HISTORY
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
        )
        + 1
    )
)


history_df[
    "learning_rate"
] = lr_history.learning_rates


training_history_path = (
    RESULTS_DIR
    / "training_history.csv"
)


history_df.to_csv(
    training_history_path,
    index=False,
)


# =============================================================================
# 25. SAVE FINAL NEGATIVE-SELECTION HISTORY / COVERAGE
# =============================================================================

selection_history_df = pd.DataFrame(
    train_gen.selection_history
)


# Because the generator prepares the next epoch after the last completed epoch,
# retain only rows corresponding to epochs that actually ran.
selection_history_df = (
    selection_history_df[
        selection_history_df[
            "epoch"
        ]
        <= epochs_completed
    ]
    .copy()
)


selection_history_df.to_csv(
    negative_selection_history_path,
    index=False,
)


coverage_records = []


for source_name in [
    "stage1_rejected",
    "stage2_not_waste",
]:

    available_objects = set(

        negative_library_df.loc[
            negative_library_df[
                "negative_source"
            ]
            == source_name,
            "sample_id",
        ].astype(
            str
        )

    )


    seen_objects = set(

        selection_history_df.loc[
            selection_history_df[
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

            "available_unique_objects":
                len(
                    available_objects
                ),

            "seen_unique_objects":
                len(
                    seen_objects
                ),

            "coverage_percent":
                (
                    100.0
                    * len(
                        seen_objects
                    )
                    / len(
                        available_objects
                    )
                    if len(
                        available_objects
                    )
                    > 0
                    else 0.0
                ),
        }
    )


coverage_df = pd.DataFrame(
    coverage_records
)


coverage_path = (
    RESULTS_DIR
    / "hard_negative_object_coverage.csv"
)


coverage_df.to_csv(
    coverage_path,
    index=False,
)


# =============================================================================
# 26. DIAGNOSTIC PLOTS
# =============================================================================

loss_plot_path = (
    RESULTS_DIR
    / "training_loss.png"
)


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


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


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


available_metrics = [

    metric

    for metric
    in diagnostic_columns

    if metric
    in history_df.columns

]


if available_metrics:

    metrics_plot_path = (
        RESULTS_DIR
        / "training_diagnostics.png"
    )


    plt.figure(
        figsize=(
            8,
            5,
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
            label=metric,
        )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Training diagnostic"
    )


    plt.title(
        "Complementary fine-tuning diagnostics"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        metrics_plot_path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


# =============================================================================
# 27. SAVE SUMMARY
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
        "ROTATING COMPLEMENTARY HARD-NEGATIVE U-NET FINE-TUNING\n"
    )


    f.write(
        "=" * 76
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
        "-" * 76
        + "\n"
    )


    f.write(
        f"Clean original replay pairs / epoch: "
        f"{len(original_records)}\n"
    )


    f.write(
        f"Hard-negative library chips: "
        f"{len(negative_library_df)}\n"
    )


    f.write(
        f"Stage-1 negative objects / epoch: "
        f"{N_STAGE1_NEGATIVE_OBJECTS_PER_EPOCH}\n"
    )


    f.write(
        f"Stage-2 negative objects / epoch: "
        f"{N_STAGE2_NEGATIVE_OBJECTS_PER_EPOCH}\n"
    )


    f.write(
        f"Total records / epoch: "
        f"{len(train_gen.current_records)}\n\n"
    )


    f.write(
        "LOSS\n"
    )


    f.write(
        "-" * 76
        + "\n"
    )


    f.write(
        "Original replay pixels: ordinary 10-class categorical cross-entropy.\n"
    )


    f.write(
        "Complementary pixels: -log(1 - P(waste)).\n"
    )


    f.write(
        "Alternative non-waste class is not specified.\n"
    )


    f.write(
        f"Negative-loss weight: "
        f"{NEGATIVE_LOSS_WEIGHT}\n\n"
    )


    f.write(
        "TRAINING RESULT\n"
    )


    f.write(
        "-" * 76
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


    f.write(
        f"Final learning rate: "
        f"{lr_history.learning_rates[-1]:.8g}\n\n"
    )


    f.write(
        "HARD-NEGATIVE OBJECT COVERAGE\n"
    )


    f.write(
        "-" * 76
        + "\n"
    )


    for _, row in (
        coverage_df.iterrows()
    ):

        f.write(
            f"{row['negative_source']}: "
            f"{int(row['seen_unique_objects'])} / "
            f"{int(row['available_unique_objects'])} "
            f"({row['coverage_percent']:.2f}%)\n"
        )


    f.write(
        "\nVALIDATION\n"
    )


    f.write(
        "-" * 76
        + "\n"
    )


    f.write(
        "No independent test data were used during fine-tuning.\n"
    )


    f.write(
        "The next step is to compare the starting and fine-tuned models "
        "on the same locked independent test area.\n"
    )


# =============================================================================
# 28. FINAL CONSOLE REPORT
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
    f"Lowest training-loss epoch: "
    f"{lowest_loss_epoch}"
)


print(
    f"Lowest training loss: "
    f"{lowest_training_loss:.8f}"
)


print("\nHard-negative object coverage:")


print(
    coverage_df.to_string(
        index=False
    )
)


print(
    f"\nFine-tuned model:\n"
    f"{FINAL_MODEL_PATH}"
)


print(
    f"\nTraining history:\n"
    f"{training_history_path}"
)


print(
    f"\nNegative selection history:\n"
    f"{negative_selection_history_path}"
)


print(
    f"\nCoverage summary:\n"
    f"{coverage_path}"
)


print(
    f"\nSummary:\n"
    f"{summary_path}"
)


print("\n")
print("=" * 80)
print("NEXT STEP")
print("=" * 80)


print(
    "\nRun the STARTING and FINE-TUNED models on exactly the same "
    "independent test area."
)


print(
    "Evaluate whether commission error falls while waste recall and "
    "segmentation quality are retained."
)


print("\nDone.")
