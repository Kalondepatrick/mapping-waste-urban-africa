#!/usr/bin/env python3
"""EfficientNetB0 + U-Net decoder: 10-class waste segmentation.

Run with the same Python environment used for the original U-Net:
    python fine_tuning_again.py

Edit PROJECT_DIR and TEMPORAL_DIR if needed. Masks must use raw labels 1-10,
with waste = 1. Old masks containing 99 are excluded before sampling; 99 is
never mapped to a class. Any other unknown labels cause an explicit error.

The original .keras model is evaluated only. The new decoder is initialized
from scratch; only the EfficientNet encoder receives ImageNet weights.
Both models accept RGB float32 input scaled to 0-1. EfficientNet's 0-255
conversion is embedded in the saved model. Do not multiply by 255 externally.

Stage 1 trains the decoder; Stage 2 fine-tunes encoder blocks 6-7 and top
convolution, keeping encoder BatchNormalization in inference mode.

Old monitor tiles came from original model-development data, so are a
retention check, not an independent test of the original model. New validation
selects checkpoints and is also not an independent test. With four new tiles,
this is a pilot; use independent months/areas for a generalisation assessment.

Requires tensorflow, numpy, pandas, rasterio, geopandas, scipy,
scikit-image and scikit-learn. First use downloads ImageNet weights unless
EFFICIENTNET_WEIGHTS is set to a compatible local checkpoint.
Reference: https://keras.io/api/applications/efficientnet/efficientnet_models/
Reference: https://keras.io/guides/transfer_learning/
"""
import json
import random
import warnings
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from scipy.ndimage import rotate
from skimage.transform import resize
from sklearn.metrics import precision_recall_curve
from tensorflow.keras.utils import Sequence, to_categorical
# ============================================================
# 1. CONFIGURATION
# ============================================================
PROJECT_DIR = Path(
    r"C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa"
)
TEMPORAL_DIR = Path(
    r"C:\Users\Peter\Documents\GitHub\waste_spatio_temporal"
)
# ------------------------------------------------------------
# Existing trained U-Net
# ------------------------------------------------------------
MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "filters_16_best.keras"
)
# ------------------------------------------------------------
# Newly digitised / newly completed March chips
# ------------------------------------------------------------
NEW_IMAGE_DIR = (
    TEMPORAL_DIR
    / "intermediaries"
    / "march_training_chips"
    / "images"
)
NEW_MASK_DIR = (
    TEMPORAL_DIR
    / "intermediaries"
    / "march_training_chips"
    / "masks"
)
NEW_TILE_INDEX = (
    TEMPORAL_DIR
    / "intermediaries"
    / "march_training_chips"
    / "annotation_diagnostics"
    / "rejected_tiles_now_complete.shp"
)
# ------------------------------------------------------------
# Original 804 model-development tiles
# ------------------------------------------------------------
OLD_IMAGE_DIR = (
    PROJECT_DIR
    / "data"
    / "tiled_data"
    / "partitioned"
    / "modelDevelopment"
    / "images"
)
OLD_MASK_DIR = (
    PROJECT_DIR
    / "data"
    / "tiled_data"
    / "partitioned"
    / "modelDevelopment"
    / "masks"
)
# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "efficientnetb0_march_transfer_v2"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)
BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "efficientnetb0_unet_best.keras"
)
FINAL_MODEL_PATH = (
    OUTPUT_DIR
    / "efficientnetb0_unet_final.keras"
)
COMPARISON_CSV = (
    OUTPUT_DIR
    / "original_vs_efficientnetb0_waste_metrics.csv"
)
PARTITION_CSV = (
    OUTPUT_DIR
    / "finetuning_partition.csv"
)
# ============================================================
# 2. SETTINGS
# ============================================================
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 8
STAGE1_EPOCHS = 30
STAGE2_EPOCHS = 100
SEED = 42
N_CLASSES = 10
WASTE_RAW_LABEL = 1
LABEL_LIST = list(
    range(1, 11)
)
LABEL_MAP = {
    raw_label: model_idx
    for model_idx, raw_label
    in enumerate(LABEL_LIST)
}
WASTE_CLASS_IDX = (
    LABEL_MAP[WASTE_RAW_LABEL]
)
# ------------------------------------------------------------
# Fine-tuning settings
# ------------------------------------------------------------
STAGE1_LEARNING_RATE = 1e-3
STAGE2_LEARNING_RATE = 1e-5
EFFICIENTNET_WEIGHTS = "imagenet"  # Or a compatible local weights file.
IGNORE_RAW_LABELS = {99}
# Separate checkpoints prevent the second stage overwriting a better first stage.
STAGE1_MODEL_PATH = OUTPUT_DIR / "efficientnetb0_unet_stage1_best.keras"
STAGE2_MODEL_PATH = OUTPUT_DIR / "efficientnetb0_unet_stage2_best.keras"
EXCLUDED_MASKS_CSV = OUTPUT_DIR / "excluded_old_masks.csv"
# Repeat each new training tile this many times.
# Augmentation means repeated copies need not look identical.
NEW_REPEAT = 10
# Retain substantial representation of original data.
OLD_TO_NEW_RATIO = 25
# But ensure at least this many old development tiles are replayed.
MIN_OLD_REPLAY = 100
# No need to use all 804.
MAX_OLD_REPLAY = 200
# Separate old tiles used only to check whether fine-tuning
# severely damaged performance on the previous distribution.
OLD_MONITOR_SIZE = 50
# Fraction of NEW tiles withheld from fine-tuning.
NEW_VALIDATION_FRACTION = 0.20
# ============================================================
# 3. REPRODUCIBILITY
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# ============================================================
# 4. LABEL ENCODING
# ============================================================
def encode_raw_mask(mask):
    mapped = np.full(
        mask.shape,
        -1,
        dtype="int32"
    )
    for raw_label, model_idx in LABEL_MAP.items():
        mapped[
            mask == raw_label
        ] = model_idx
    if np.any(mapped < 0):
        invalid = np.unique(
            mask[
                mapped < 0
            ]
        )
        raise ValueError(
            f"Unexpected mask values: "
            f"{invalid.tolist()}"
        )
    return mapped
def inspect_raw_mask(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    if not np.all(np.isfinite(mask)) or not np.all(mask == np.rint(mask)):
        raise ValueError(f"{mask_path} contains non-finite or non-integer labels.")
    values = set(np.unique(mask).astype("int64").tolist())
    unknown = values - set(LABEL_LIST) - IGNORE_RAW_LABELS
    if unknown:
        raise ValueError(f"{mask_path} contains unexpected labels {sorted(unknown)}")
    return values & IGNORE_RAW_LABELS

# ============================================================
# 5. IDENTIFY NEW TILES
# ============================================================
print("\n")
print("=" * 70)
print("IDENTIFYING NEWLY COMPLETED TILES")
print("=" * 70)
if not NEW_TILE_INDEX.exists():
    raise FileNotFoundError(
        f"Cannot find:\n{NEW_TILE_INDEX}"
    )
new_index = gpd.read_file(
    NEW_TILE_INDEX
)
if "tile_id" not in new_index.columns:
    raise ValueError(
        "The newly completed tile shapefile "
        "does not contain 'tile_id'."
    )
new_names = sorted(
    {
        f"{str(tile)}.tif"
        for tile
        in new_index["tile_id"]
    }
)
new_pairs = []
for name in new_names:
    img = (
        NEW_IMAGE_DIR
        / name
    )
    mask = (
        NEW_MASK_DIR
        / name
    )
    if (
        img.exists()
        and
        mask.exists()
    ):
        ignored = inspect_raw_mask(mask)
        if ignored:
            raise ValueError(
                f"New tile {mask} still contains {sorted(ignored)}. "
                "Finish its annotation before adding it to the completed-tile index."
            )
        new_pairs.append(
            (
                str(img),
                str(mask),
                name,
                "new"
            )
        )
print(
    f"New complete image-mask pairs: "
    f"{len(new_pairs)}"
)
if len(new_pairs) < 2:
    raise ValueError(
        "At least two newly completed tiles are required "
        "to retain a new-data validation tile."
    )
# ============================================================
# 6. SPLIT NEW DATA
# ============================================================
random.shuffle(
    new_pairs
)
n_new_val = max(
    1,
    int(
        round(
            len(new_pairs)
            * NEW_VALIDATION_FRACTION
        )
    )
)
# Never let validation consume all new tiles.
n_new_val = min(
    n_new_val,
    len(new_pairs) - 1
)
new_val_pairs = (
    new_pairs[:n_new_val]
)
new_train_pairs = (
    new_pairs[n_new_val:]
)
print(
    f"New tiles for training   : "
    f"{len(new_train_pairs)}"
)
print(
    f"New tiles for validation : "
    f"{len(new_val_pairs)}"
)
# ============================================================
# 7. ORIGINAL DEVELOPMENT DATA
# ============================================================
old_image_names = {
    p.name
    for p
    in OLD_IMAGE_DIR.glob("*.tif")
}
old_mask_names = {
    p.name
    for p
    in OLD_MASK_DIR.glob("*.tif")
}
old_names = sorted(
    old_image_names.intersection(
        old_mask_names
    )
)
if not old_names:
    raise FileNotFoundError(
        "No matching original development "
        "image-mask pairs were found."
    )
# Filter BEFORE splitting, so neither monitor nor replay can contain 99.
valid_old_names = []
excluded_old_rows = []
for name in old_names:
    ignored = inspect_raw_mask(OLD_MASK_DIR / name)
    if ignored:
        excluded_old_rows.append({"tile": name, "ignored_labels": str(sorted(ignored))})
    else:
        valid_old_names.append(name)

pd.DataFrame(excluded_old_rows, columns=["tile", "ignored_labels"]).to_csv(
    EXCLUDED_MASKS_CSV, index=False
)
print("\n" + "=" * 70)
print("CHECKING ORIGINAL MASK LABELS")
print("=" * 70)
print(f"Old masks before filtering : {len(old_names)}")
print(f"Old masks containing 99    : {len(excluded_old_rows)}")
print(f"Old masks retained         : {len(valid_old_names)}")
for row in excluded_old_rows[:20]:
    print(f"  Excluded: {row['tile']} labels={row['ignored_labels']}")
if len(excluded_old_rows) / len(old_names) >= 0.10:
    warnings.warn(
        "At least 10% of old tiles contain 99 and were excluded. "
        "Consider pixel-level ignore masking to retain their valid annotations."
    )
old_names = valid_old_names
if len(old_names) < 2:
    raise ValueError("Need at least two valid old tiles for separate replay and monitor sets.")
random.shuffle(old_names)
# ============================================================
# 8. OLD MONITOR SAMPLE
# ============================================================
n_old_monitor = min(
    OLD_MONITOR_SIZE,
    len(old_names) - 1  # Reserve at least one replay tile.
)
old_monitor_names = (
    old_names[:n_old_monitor]
)
remaining_old_names = (
    old_names[n_old_monitor:]
)
old_monitor_pairs = [
    (
        str(
            OLD_IMAGE_DIR
            / name
        ),
        str(
            OLD_MASK_DIR
            / name
        ),
        name,
        "old_monitor"
    )
    for name
    in old_monitor_names
]
# ============================================================
# 9. OLD REPLAY SAMPLE
# ============================================================
target_old = max(
    MIN_OLD_REPLAY,
    len(new_train_pairs)
    * OLD_TO_NEW_RATIO
)
n_old_replay = min(
    target_old,
    MAX_OLD_REPLAY,
    len(remaining_old_names)
)
old_replay_names = (
    remaining_old_names[
        :n_old_replay
    ]
)
old_replay_pairs = [
    (
        str(
            OLD_IMAGE_DIR
            / name
        ),
        str(
            OLD_MASK_DIR
            / name
        ),
        name,
        "old_replay"
    )
    for name
    in old_replay_names
]
print(
    f"Original replay tiles     : "
    f"{len(old_replay_pairs)}"
)
print(
    f"Original monitor tiles    : "
    f"{len(old_monitor_pairs)}"
)
# ============================================================
# 10. OVERSAMPLE NEW TRAINING TILES
# ============================================================
repeated_new_pairs = (
    new_train_pairs
    * NEW_REPEAT
)
training_pairs = (
    repeated_new_pairs
    + old_replay_pairs
)
random.shuffle(
    training_pairs
)
print("\n")
print("=" * 70)
print("FINAL TRANSFER-LEARNING DATASET")
print("=" * 70)
print(
    f"Unique new training tiles : "
    f"{len(new_train_pairs)}"
)
print(
    f"New repetitions           : "
    f"{NEW_REPEAT}x"
)
print(
    f"Repeated new observations : "
    f"{len(repeated_new_pairs)}"
)
print(
    f"Old replay observations   : "
    f"{len(old_replay_pairs)}"
)
print(
    f"Total training records    : "
    f"{len(training_pairs)}"
)
print(
    f"Batch size                : "
    f"{BATCH_SIZE}"
)
print(
    f"Expected steps/epoch      : "
    f"{int(np.ceil(len(training_pairs) / BATCH_SIZE))}"
)
# ============================================================
# 11. SAVE PARTITION INFORMATION
# ============================================================
partition_rows = []
for p in new_train_pairs:
    partition_rows.append(
        {
            "tile": p[2],
            "group": "new_training"
        }
    )
for p in new_val_pairs:
    partition_rows.append(
        {
            "tile": p[2],
            "group": "new_validation"
        }
    )
for p in old_replay_pairs:
    partition_rows.append(
        {
            "tile": p[2],
            "group": "old_replay"
        }
    )
for p in old_monitor_pairs:
    partition_rows.append(
        {
            "tile": p[2],
            "group": "old_monitor"
        }
    )
pd.DataFrame(
    partition_rows
).to_csv(
    PARTITION_CSV,
    index=False
)
# ============================================================
# 12. DATA GENERATOR
# ============================================================
class RasterGenerator(Sequence):
    def __init__(
        self,
        pairs,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pairs = list(
            pairs
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
            len(self.pairs)
        )
        self.on_epoch_end()
    def __len__(self):
        return int(
            np.ceil(
                len(self.indices)
                /
                self.batch_size
            )
        )
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(
                self.indices
            )
    def __getitem__(
        self,
        idx
    ):
        batch_indices = (
            self.indices[
                idx * self.batch_size:
                (idx + 1) * self.batch_size
            ]
        )
        images = []
        masks = []
        for i in batch_indices:
            image_path = (
                self.pairs[i][0]
            )
            mask_path = (
                self.pairs[i][1]
            )
            # ----------------------------------------------
            # Read RGB image
            # ----------------------------------------------
            with rasterio.open(
                image_path
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
                        "float32"
                    )
                )
            # ----------------------------------------------
            # Read mask
            # ----------------------------------------------
            with rasterio.open(
                mask_path
            ) as src:
                mask = src.read(
                    1
                )
            # ----------------------------------------------
            # Check native labels before nearest-neighbour resizing.
            if not np.all(np.isfinite(mask)) or not np.all(mask == np.rint(mask)):
                raise ValueError(f"{mask_path} contains non-integer or non-finite labels")
            native_invalid = np.setdiff1d(np.unique(mask), LABEL_LIST)
            if len(native_invalid):
                raise ValueError(f"{mask_path} contains unexpected labels {native_invalid.tolist()}")
            if image.shape[:2] != mask.shape:
                raise ValueError(f"Image/mask dimensions differ: {image_path}, {mask_path}")
            if not np.all(np.isfinite(image)) or image.min() < 0 or image.max() > 255:
                raise ValueError(f"Expected RGB values in 0-255: {image_path}")
            # Same resizing as original model
            # ----------------------------------------------
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
                "int32"
            )
            # ----------------------------------------------
            # Check classes
            # ----------------------------------------------
            invalid = (
                np.setdiff1d(
                    np.unique(mask),
                    LABEL_LIST
                )
            )
            if len(invalid) > 0:
                raise ValueError(
                    f"{mask_path} contains "
                    f"unexpected labels "
                    f"{invalid.tolist()}"
                )
            # ----------------------------------------------
            # SAME augmentation as original training
            # ----------------------------------------------
            if self.augment:
                if random.random() < 0.5:
                    image = np.flipud(
                        image
                    ).copy()
                    mask = np.flipud(
                        mask
                    ).copy()
                if random.random() < 0.5:
                    angle = (
                        random.uniform(
                            -15,
                            15
                        )
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
                        "int32"
                    )
            # ----------------------------------------------
            # SAME normalisation
            # ----------------------------------------------
            image = (
                image
                / 255.0
            )
            # ----------------------------------------------
            # SAME 1-10 -> 0-9 -> one-hot encoding
            # ----------------------------------------------
            mapped_mask = (
                encode_raw_mask(
                    mask
                )
            )
            one_hot = (
                to_categorical(
                    mapped_mask,
                    num_classes=N_CLASSES
                )
            )
            images.append(
                image
            )
            masks.append(
                one_hot
            )
        return (
            np.asarray(
                images,
                dtype="float32"
            ),
            np.asarray(
                masks,
                dtype="float32"
            )
        )
# ============================================================
# 13. GENERATORS
# ============================================================
train_gen = RasterGenerator(
    training_pairs,
    shuffle=True,
    augment=True
)
new_val_gen = RasterGenerator(
    new_val_pairs,
    shuffle=False,
    augment=False
)
old_monitor_gen = RasterGenerator(
    old_monitor_pairs,
    shuffle=False,
    augment=False
)
# ============================================================
# 14. WASTE METRIC FUNCTIONS
# ============================================================
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
        (1.0 - y_true_waste)
        * y_pred_waste
    )
    return (
        (tp + smooth)
        /
        (tp + fp + smooth)
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
        * (1.0 - y_pred_waste)
    )
    return (
        (tp + smooth)
        /
        (tp + fn + smooth)
    )
def f1_waste(
    y_true,
    y_pred,
    smooth=1e-7
):
    p = precision_waste(
        y_true,
        y_pred,
        smooth
    )
    r = recall_waste(
        y_true,
        y_pred,
        smooth
    )
    return (
        2.0 * p * r
        /
        (p + r + smooth)
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
        +
        tf.reduce_sum(
            y_pred_waste
        )
        -
        intersection
    )
    return (
        (intersection + smooth)
        /
        (union + smooth)
    )
# ============================================================
# 15. EXTERNAL WASTE EVALUATION
# ============================================================
def evaluate_waste(
    model,
    generator,
    dataset_name,
    probability_threshold=0.50
):
    if len(generator) == 0:
        raise ValueError(f"No tiles available for {dataset_name}")
    y_true_all = []
    waste_probability_all = []
    argmax_all = []
    for batch_i in range(
        len(generator)
    ):
        x, y = (
            generator[batch_i]
        )
        pred = model.predict(
            x,
            verbose=0
        )
        # True waste
        true_waste = (
            y[
                ...,
                WASTE_CLASS_IDX
            ]
            .reshape(-1)
            .astype("uint8")
        )
        # Probability of waste
        waste_prob = (
            pred[
                ...,
                WASTE_CLASS_IDX
            ]
            .reshape(-1)
        )
        # Multiclass argmax -> waste/non-waste
        pred_argmax = (
            np.argmax(
                pred,
                axis=-1
            )
            == WASTE_CLASS_IDX
        ).reshape(-1).astype(
            "uint8"
        )
        y_true_all.append(
            true_waste
        )
        waste_probability_all.append(
            waste_prob
        )
        argmax_all.append(
            pred_argmax
        )
    y_true = np.concatenate(
        y_true_all
    )
    waste_prob = np.concatenate(
        waste_probability_all
    )
    pred_argmax = np.concatenate(
        argmax_all
    )
    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------
    def binary_metrics(
        truth,
        pred
    ):
        tp = np.sum(
            (truth == 1)
            &
            (pred == 1)
        )
        fp = np.sum(
            (truth == 0)
            &
            (pred == 1)
        )
        fn = np.sum(
            (truth == 1)
            &
            (pred == 0)
        )
        precision = (
            tp
            /
            (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )
        recall = (
            tp
            /
            (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall
            /
            (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        iou = (
            tp
            /
            (tp + fp + fn)
            if (tp + fp + fn) > 0
            else 0.0
        )
        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "IoU": iou,
            "TP_pixels": int(tp),
            "FP_pixels": int(fp),
            "FN_pixels": int(fn)
        }
    # --------------------------------------------------------
    # A. Multiclass argmax decision
    # --------------------------------------------------------
    argmax_metrics = (
        binary_metrics(
            y_true,
            pred_argmax
        )
    )
    # --------------------------------------------------------
    # B. Fixed 0.50 waste probability threshold
    # --------------------------------------------------------
    pred_fixed = (
        waste_prob
        >= probability_threshold
    ).astype(
        "uint8"
    )
    fixed_metrics = (
        binary_metrics(
            y_true,
            pred_fixed
        )
    )
    # --------------------------------------------------------
    # C. Optimal threshold on THIS evaluation dataset
    #
    # Diagnostic only — do not use as independent test tuning.
    # --------------------------------------------------------
    best_threshold = np.nan
    optimal_metrics = {
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "IoU": np.nan,
        "TP_pixels": np.nan,
        "FP_pixels": np.nan,
        "FN_pixels": np.nan
    }
    if (
        np.any(y_true == 1)
        and
        np.any(y_true == 0)
    ):
        precision_curve, recall_curve, thresholds = (
            precision_recall_curve(
                y_true,
                waste_prob
            )
        )
        if len(thresholds) > 0:
            p = precision_curve[:-1]
            r = recall_curve[:-1]
            f1 = (
                2 * p * r
                /
                (
                    p
                    + r
                    + 1e-7
                )
            )
            best_index = (
                np.argmax(
                    f1
                )
            )
            best_threshold = float(
                thresholds[
                    best_index
                ]
            )
            pred_optimal = (
                waste_prob
                >= best_threshold
            ).astype(
                "uint8"
            )
            optimal_metrics = (
                binary_metrics(
                    y_true,
                    pred_optimal
                )
            )
    results = []
    for decision, metrics, threshold in [
        (
            "multiclass_argmax",
            argmax_metrics,
            np.nan
        ),
        (
            "waste_probability_0.50",
            fixed_metrics,
            probability_threshold
        ),
        (
            "diagnostic_optimal_threshold",
            optimal_metrics,
            best_threshold
        )
    ]:
        results.append(
            {
                "Dataset": dataset_name,
                "Decision": decision,
                "Threshold": threshold,
                **metrics
            }
        )
    return results
# ============================================================
# 16. LOAD ORIGINAL MODEL
# ============================================================
print("\n")
print("=" * 70)
print("LOADING ORIGINAL MODEL")
print("=" * 70)
original_model = (
    tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
)
print(
    "Input shape :",
    original_model.input_shape
)
print(
    "Output shape:",
    original_model.output_shape
)
if (
    original_model.output_shape[-1]
    != N_CLASSES
):
    raise ValueError(
        "Loaded model does not have "
        "10 output classes."
    )
# ============================================================
# 17. BASELINE BEFORE FINE-TUNING
# ============================================================
print("\n")
print("=" * 70)
print("ORIGINAL MODEL: PRE-FINETUNING PERFORMANCE")
print("=" * 70)
comparison_results = []
baseline_new = evaluate_waste(
    original_model,
    new_val_gen,
    "new_validation"
)
for row in baseline_new:
    row["Model"] = (
        "original_custom_unet"
    )
comparison_results.extend(
    baseline_new
)
baseline_old = evaluate_waste(
    original_model,
    old_monitor_gen,
    "old_monitor"
)
for row in baseline_old:
    row["Model"] = (
        "original_custom_unet"
    )
comparison_results.extend(
    baseline_old
)
# ============================================================
# 18. BUILD EFFICIENTNETB0 U-NET
# ============================================================
# Save baseline results before training, in case a later stage is interrupted.
pd.DataFrame(comparison_results).to_csv(COMPARISON_CSV, index=False)
del original_model


def conv_block(x, filters, name):
    for i in (1, 2):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False,
            kernel_initializer="he_normal", name=f"{name}_conv{i}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = tf.keras.layers.Activation("relu", name=f"{name}_relu{i}")(x)
    return x


def build_efficientnet_unet():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=EFFICIENTNET_WEIGHTS,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    # Last outputs before each spatial downsample: 128, 64, 32, 16 pixels.
    feature_names = ["block1a_project_bn", "block2b_add", "block3b_add", "block5c_add"]
    encoder = tf.keras.Model(
        base.input,
        [base.get_layer(name).output for name in feature_names] + [base.output],
        name="efficientnetb0_encoder"
    )
    encoder.trainable = False
    inputs = tf.keras.Input((IMG_HEIGHT, IMG_WIDTH, 3), name="rgb_0_to_1")
    # EfficientNet has its own built-in preprocessing and requires 0-255.
    pixels = tf.keras.layers.Rescaling(255.0, name="efficientnet_input_0_to_255")(inputs)
    features = encoder(pixels, training=False)
    x = features[-1]
    for i, (skip, filters) in enumerate(zip(reversed(features[:-1]), (256, 128, 64, 32)), 1):
        x = tf.keras.layers.Conv2DTranspose(
            filters, 2, strides=2, padding="same", name=f"decoder_up{i}"
        )(x)
        x = tf.keras.layers.Concatenate(name=f"decoder_skip{i}")([x, skip])
        x = conv_block(x, filters, f"decoder_block{i}")
    x = tf.keras.layers.Conv2DTranspose(
        16, 2, strides=2, padding="same", name="decoder_up_full_resolution"
    )(x)
    x = conv_block(x, 16, "decoder_full_resolution")
    outputs = tf.keras.layers.Conv2D(
        N_CLASSES, 1, padding="same", activation="softmax", name="class_probabilities"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="efficientnetb0_unet")
    if model.output_shape[1:] != (IMG_HEIGHT, IMG_WIDTH, N_CLASSES):
        raise ValueError(f"Unexpected output shape: {model.output_shape}")
    return model, encoder


def compile_model(model, learning_rate):
    # Recompile after every trainability change, with a fresh optimizer.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", precision_waste, recall_waste, f1_waste, iou_waste]
    )


def stage_callbacks(checkpoint, history_path, patience):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, patience // 2),
            min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint), monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(history_path)),
        tf.keras.callbacks.TerminateOnNaN()
    ]


print("\n" + "=" * 70)
print("BUILDING EFFICIENTNETB0 U-NET")
print("=" * 70)
model, encoder = build_efficientnet_unet()
print("Input shape:", model.input_shape, "Output shape:", model.output_shape)
print("\nSTAGE 1: FROZEN ENCODER / TRAIN NEW DECODER")
compile_model(model, STAGE1_LEARNING_RATE)
history1 = model.fit(
    train_gen, validation_data=new_val_gen, epochs=STAGE1_EPOCHS,
    callbacks=stage_callbacks(
        STAGE1_MODEL_PATH, OUTPUT_DIR / "stage1_decoder_history.csv", patience=10
    ), verbose=1
)
losses1 = np.asarray(history1.history.get("val_loss", []), dtype=float)
finite1 = losses1[np.isfinite(losses1)]
if not STAGE1_MODEL_PATH.exists() or not len(finite1):
    raise RuntimeError("Stage 1 produced no finite validation checkpoint.")
# Explicitly restore the saved best stage, even when all requested epochs ran.
model = tf.keras.models.load_model(STAGE1_MODEL_PATH, compile=False)
encoder = model.get_layer("efficientnetb0_encoder")

print("\nSTAGE 2: FINE-TUNE UPPER ENCODER BLOCKS AT LOW LEARNING RATE")
encoder.trainable = True
for layer in encoder.layers:
    layer.trainable = (
        layer.name.startswith(("block6", "block7", "top_conv"))
        and not isinstance(layer, tf.keras.layers.BatchNormalization)
    )
print("Trainable encoder layers:", sum(layer.trainable for layer in encoder.layers))
compile_model(model, STAGE2_LEARNING_RATE)
history2 = model.fit(
    train_gen, validation_data=new_val_gen, epochs=STAGE2_EPOCHS,
    callbacks=stage_callbacks(
        STAGE2_MODEL_PATH, OUTPUT_DIR / "stage2_finetuning_history.csv", patience=20
    ), verbose=1
)
# This state can have early-stopping restoration; it is not necessarily the last epoch.
model.save(FINAL_MODEL_PATH)
losses2 = np.asarray(history2.history.get("val_loss", []), dtype=float)
finite2 = losses2[np.isfinite(losses2)]
best1 = float(finite1.min())
best2 = float(finite2.min()) if len(finite2) else float("inf")
selected_path = STAGE2_MODEL_PATH if best2 < best1 else STAGE1_MODEL_PATH
if not selected_path.exists():
    raise RuntimeError(f"Selected checkpoint missing: {selected_path}")
best_model = tf.keras.models.load_model(selected_path, compile=False)
best_model.save(BEST_MODEL_PATH)
print(f"Best stage 1 val_loss: {best1:.6f}; stage 2: {best2:.6f}")
print(f"Selected checkpoint: {selected_path.name}")
with (OUTPUT_DIR / "experiment_settings.json").open("w", encoding="utf-8") as f:
    json.dump({
        "encoder": "EfficientNetB0", "encoder_weights": str(EFFICIENTNET_WEIGHTS),
        "input_range": [0, 1], "internal_encoder_input_range": [0, 255],
        "labels_raw_to_output_index": LABEL_MAP, "waste_class_index": WASTE_CLASS_IDX,
        "seed": SEED, "ignored_raw_labels_excluded_per_tile": sorted(IGNORE_RAW_LABELS),
        "old_masks_excluded": len(excluded_old_rows),
        "new_train_unique": len(new_train_pairs), "new_validation_unique": len(new_val_pairs),
        "old_replay_unique": len(old_replay_pairs), "old_monitor_unique": len(old_monitor_pairs),
        "stage1_best_val_loss": best1,
        "stage2_best_val_loss": best2 if np.isfinite(best2) else None,
        "selected_checkpoint": selected_path.name,
        "old_monitor_is_independent_test": False
    }, f, indent=2)

# ============================================================
# 24. POST-FINETUNING EVALUATION
# ============================================================
print("\n")
print("=" * 70)
print("BEST FINE-TUNED MODEL: PERFORMANCE")
print("=" * 70)
finetuned_new = evaluate_waste(
    best_model,
    new_val_gen,
    "new_validation"
)
for row in finetuned_new:
    row["Model"] = (
        "efficientnetb0_imagenet_unet"
    )
comparison_results.extend(
    finetuned_new
)
finetuned_old = evaluate_waste(
    best_model,
    old_monitor_gen,
    "old_monitor"
)
for row in finetuned_old:
    row["Model"] = (
        "efficientnetb0_imagenet_unet"
    )
comparison_results.extend(
    finetuned_old
)
# ============================================================
# 25. SAVE COMPARISON
# ============================================================
comparison_df = pd.DataFrame(
    comparison_results
)
comparison_df = comparison_df[
    [
        "Model",
        "Dataset",
        "Decision",
        "Threshold",
        "Precision",
        "Recall",
        "F1",
        "IoU",
        "TP_pixels",
        "FP_pixels",
        "FN_pixels"
    ]
]
comparison_df.to_csv(
    COMPARISON_CSV,
    index=False
)
# ============================================================
# 26. DISPLAY RESULTS
# ============================================================
print("\n")
print("=" * 70)
print("ORIGINAL U-NET vs EFFICIENTNETB0 WASTE PERFORMANCE")
print("=" * 70)
print(
    comparison_df.to_string(
        index=False
    )
)
# ============================================================
# 27. FINAL OUTPUT
# ============================================================
print("\n")
print("=" * 70)
print("FINE-TUNING COMPLETE")
print("=" * 70)
print(
    "\nBest fine-tuned model:"
)
print(
    BEST_MODEL_PATH
)
print(
    "\nFinal training state (may have restored best weights):"
)
print(
    FINAL_MODEL_PATH
)
print(
    "\nWaste comparison:"
)
print(
    COMPARISON_CSV
)
print(
    "\nPartition record:"
)
print(
    PARTITION_CSV
)
print(
    "\nTraining history:"
)
print(
    OUTPUT_DIR / "stage1_decoder_history.csv"
)
print(
    OUTPUT_DIR / "stage2_finetuning_history.csv"
)
print("=" * 70)

print("Excluded-mask audit:", EXCLUDED_MASKS_CSV)
print("The old monitor checks retention; it is not an independent baseline test.")
print("RGB inference input remains 0-1. Output index 0 = raw class 1 (waste).")
print("Evaluate both models on independent months/areas before claiming wider improvement.")
