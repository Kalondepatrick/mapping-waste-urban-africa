#!/usr/bin/env python3

import random
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
    / "march_finetuning_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "filters_16_march_finetuned_best.keras"
)

FINAL_MODEL_PATH = (
    OUTPUT_DIR
    / "filters_16_march_finetuned_final.keras"
)

COMPARISON_CSV = (
    OUTPUT_DIR
    / "original_vs_finetuned_waste_metrics.csv"
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
EPOCHS = 100

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

LEARNING_RATE = 1e-5

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

# Freeze some early layers during fine-tuning.
FREEZE_FRACTION = 0.40


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

random.shuffle(
    old_names
)


# ============================================================
# 8. OLD MONITOR SAMPLE
# ============================================================

n_old_monitor = min(
    OLD_MONITOR_SIZE,
    len(old_names)
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
print("FINAL FINE-TUNING DATASET")
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
            "IoU": iou
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
        "IoU": np.nan
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
        "original"
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
        "original"
    )

comparison_results.extend(
    baseline_old
)


# ============================================================
# 18. PREPARE MODEL FOR FINE-TUNING
# ============================================================

model = original_model

freeze_until = int(
    len(model.layers)
    * FREEZE_FRACTION
)

for layer in model.layers[
    :freeze_until
]:

    layer.trainable = False


for layer in model.layers[
    freeze_until:
]:

    layer.trainable = True


print("\n")
print(
    f"Total layers  : "
    f"{len(model.layers)}"
)

print(
    f"Frozen layers : "
    f"{freeze_until}"
)

print(
    f"Trainable     : "
    f"{len(model.layers) - freeze_until}"
)


# ============================================================
# 19. COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy",
        precision_waste,
        recall_waste,
        f1_waste,
        iou_waste
    ]
)


# ============================================================
# 20. CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(
        OUTPUT_DIR
        / "fine_tuning_history.csv"
    )
]


# ============================================================
# 21. FINE-TUNE
# ============================================================

print("\n")
print("=" * 70)
print("STARTING SECOND FINE-TUNING RUN")
print("=" * 70)

history = model.fit(

    train_gen,

    validation_data=new_val_gen,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# 22. SAVE FINAL STATE
# ============================================================

model.save(
    FINAL_MODEL_PATH
)


# ============================================================
# 23. LOAD BEST CHECKPOINT
# ============================================================

best_model = (
    tf.keras.models.load_model(
        BEST_MODEL_PATH,
        compile=False
    )
)


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
        "fine_tuned"
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
        "fine_tuned"
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
        "IoU"
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
print("ORIGINAL vs FINE-TUNED WASTE PERFORMANCE")
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
    "\nFinal epoch model:"
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
    OUTPUT_DIR
    / "fine_tuning_history.csv"
)

print("=" * 70)