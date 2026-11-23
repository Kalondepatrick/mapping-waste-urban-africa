#!/usr/bin/env python3
"""
EfficientNetB0-pretrained U-Net-style segmentation model for the Ndirande
10-class waste mapping workflow.

Purpose
-------
This is a controlled transfer-learning experiment based on the existing
fine-tuning script. It keeps the same:
    * 256 x 256 RGB inputs
    * 10 raw mask labels (1..10 -> model indices 0..9)
    * new-tile validation split
    * old-data replay and old-monitor sample
    * geometric augmentation
    * categorical cross-entropy objective
    * waste-specific external evaluation

The main experimental change is the encoder:
    original custom U-Net encoder -> ImageNet-pretrained EfficientNetB0.

Training is done in two stages:
    Stage 1: freeze EfficientNet; train the new decoder.
    Stage 2: unfreeze only the upper EfficientNet blocks at a low LR,
             while keeping BatchNormalization frozen.

Important preprocessing detail
------------------------------
The existing generator returns images scaled to [0, 1]. Keras EfficientNet
expects [0, 255] because its own preprocessing is included in the model.
To preserve compatibility with the original U-Net baseline, this script keeps
the generator unchanged and multiplies by 255 inside the EfficientNet branch.
"""

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
from tensorflow.keras import layers
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

# Existing custom U-Net: used only as a baseline comparator.
ORIGINAL_MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "filters_16_best.keras"
)

# Newly digitised / newly completed March chips.
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

# Original 804 model-development tiles.
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

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "efficientnetb0_transfer_v1"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_BEST_WEIGHTS = OUTPUT_DIR / "efficientnetb0_stage1_best.weights.h5"
STAGE2_BEST_WEIGHTS = OUTPUT_DIR / "efficientnetb0_stage2_best.weights.h5"
FINAL_MODEL_PATH = OUTPUT_DIR / "efficientnetb0_unet_finetuned_best.keras"
COMPARISON_CSV = OUTPUT_DIR / "original_vs_efficientnet_waste_metrics.csv"
PARTITION_CSV = OUTPUT_DIR / "training_partition.csv"


# ============================================================
# 2. SETTINGS
# ============================================================

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 8
SEED = 42

N_CLASSES = 10
WASTE_RAW_LABEL = 1
LABEL_LIST = list(range(1, 11))
LABEL_MAP = {
    raw_label: model_idx
    for model_idx, raw_label in enumerate(LABEL_LIST)
}
WASTE_CLASS_IDX = LABEL_MAP[WASTE_RAW_LABEL]

# Same sampling strategy as the previous script.
NEW_REPEAT = 10
OLD_TO_NEW_RATIO = 25
MIN_OLD_REPLAY = 100
MAX_OLD_REPLAY = 200
OLD_MONITOR_SIZE = 50
NEW_VALIDATION_FRACTION = 0.20

# Transfer-learning schedule.
STAGE1_EPOCHS = 40
STAGE1_LR = 1e-3

STAGE2_EPOCHS = 80
STAGE2_LR = 1e-5

# Fine-tune only the upper part of EfficientNetB0.
# block6 + block7 + top is a conservative first experiment.
FINE_TUNE_FROM = "block6a_expand_conv"


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
    mapped = np.full(mask.shape, -1, dtype="int32")

    for raw_label, model_idx in LABEL_MAP.items():
        mapped[mask == raw_label] = model_idx

    if np.any(mapped < 0):
        invalid = np.unique(mask[mapped < 0])
        raise ValueError(
            f"Unexpected mask values: {invalid.tolist()}"
        )

    return mapped


# ============================================================
# 5. IDENTIFY NEW TILES
# ============================================================

print("\n" + "=" * 70)
print("IDENTIFYING NEWLY COMPLETED TILES")
print("=" * 70)

if not NEW_TILE_INDEX.exists():
    raise FileNotFoundError(f"Cannot find:\n{NEW_TILE_INDEX}")

new_index = gpd.read_file(NEW_TILE_INDEX)

if "tile_id" not in new_index.columns:
    raise ValueError(
        "The newly completed tile shapefile does not contain 'tile_id'."
    )

new_names = sorted({f"{str(tile)}.tif" for tile in new_index["tile_id"]})

new_pairs = []
for name in new_names:
    img = NEW_IMAGE_DIR / name
    mask = NEW_MASK_DIR / name
    if img.exists() and mask.exists():
        new_pairs.append((str(img), str(mask), name, "new"))

print(f"New complete image-mask pairs: {len(new_pairs)}")

if len(new_pairs) < 2:
    raise ValueError(
        "At least two newly completed tiles are required to retain a validation tile."
    )


# ============================================================
# 6. SPLIT NEW DATA
# ============================================================

random.shuffle(new_pairs)

n_new_val = max(
    1,
    int(round(len(new_pairs) * NEW_VALIDATION_FRACTION))
)
n_new_val = min(n_new_val, len(new_pairs) - 1)

new_val_pairs = new_pairs[:n_new_val]
new_train_pairs = new_pairs[n_new_val:]

print(f"New tiles for training   : {len(new_train_pairs)}")
print(f"New tiles for validation : {len(new_val_pairs)}")


# ============================================================
# 7. ORIGINAL DEVELOPMENT DATA
# ============================================================

old_image_names = {p.name for p in OLD_IMAGE_DIR.glob("*.tif")}
old_mask_names = {p.name for p in OLD_MASK_DIR.glob("*.tif")}
old_names = sorted(old_image_names.intersection(old_mask_names))

if not old_names:
    raise FileNotFoundError(
        "No matching original development image-mask pairs were found."
    )

random.shuffle(old_names)


# ============================================================
# 8. OLD MONITOR SAMPLE
# ============================================================

n_old_monitor = min(OLD_MONITOR_SIZE, len(old_names))
old_monitor_names = old_names[:n_old_monitor]
remaining_old_names = old_names[n_old_monitor:]

old_monitor_pairs = [
    (
        str(OLD_IMAGE_DIR / name),
        str(OLD_MASK_DIR / name),
        name,
        "old_monitor",
    )
    for name in old_monitor_names
]


# ============================================================
# 9. OLD REPLAY SAMPLE
# ============================================================

target_old = max(
    MIN_OLD_REPLAY,
    len(new_train_pairs) * OLD_TO_NEW_RATIO,
)

n_old_replay = min(
    target_old,
    MAX_OLD_REPLAY,
    len(remaining_old_names),
)

old_replay_names = remaining_old_names[:n_old_replay]

old_replay_pairs = [
    (
        str(OLD_IMAGE_DIR / name),
        str(OLD_MASK_DIR / name),
        name,
        "old_replay",
    )
    for name in old_replay_names
]

print(f"Original replay tiles    : {len(old_replay_pairs)}")
print(f"Original monitor tiles   : {len(old_monitor_pairs)}")


# ============================================================
# 10. OVERSAMPLE NEW TRAINING TILES
# ============================================================

repeated_new_pairs = new_train_pairs * NEW_REPEAT
training_pairs = repeated_new_pairs + old_replay_pairs
random.shuffle(training_pairs)

print("\n" + "=" * 70)
print("FINAL TRANSFER-LEARNING DATASET")
print("=" * 70)
print(f"Unique new training tiles : {len(new_train_pairs)}")
print(f"New repetitions           : {NEW_REPEAT}x")
print(f"Repeated new observations : {len(repeated_new_pairs)}")
print(f"Old replay observations   : {len(old_replay_pairs)}")
print(f"Total training records    : {len(training_pairs)}")
print(f"Batch size                : {BATCH_SIZE}")
print(
    "Expected steps/epoch       : "
    f"{int(np.ceil(len(training_pairs) / BATCH_SIZE))}"
)


# ============================================================
# 11. SAVE PARTITION INFORMATION
# ============================================================

partition_rows = []

for p in new_train_pairs:
    partition_rows.append({"tile": p[2], "group": "new_training"})
for p in new_val_pairs:
    partition_rows.append({"tile": p[2], "group": "new_validation"})
for p in old_replay_pairs:
    partition_rows.append({"tile": p[2], "group": "old_replay"})
for p in old_monitor_pairs:
    partition_rows.append({"tile": p[2], "group": "old_monitor"})

pd.DataFrame(partition_rows).to_csv(PARTITION_CSV, index=False)


# ============================================================
# 12. DATA GENERATOR
# ============================================================

class RasterGenerator(Sequence):
    """Same [0,1] generator convention as the original U-Net workflow."""

    def __init__(
        self,
        pairs,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pairs = list(pairs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.pairs))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]

        images = []
        masks = []

        for i in batch_indices:
            image_path = self.pairs[i][0]
            mask_path = self.pairs[i][1]

            with rasterio.open(image_path) as src:
                image = (
                    src.read([1, 2, 3])
                    .transpose(1, 2, 0)
                    .astype("float32")
                )

            with rasterio.open(mask_path) as src:
                mask = src.read(1)

            image = resize(
                image,
                (IMG_HEIGHT, IMG_WIDTH),
                preserve_range=True,
            )

            mask = resize(
                mask,
                (IMG_HEIGHT, IMG_WIDTH),
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )
            mask = np.rint(mask).astype("int32")

            invalid = np.setdiff1d(np.unique(mask), LABEL_LIST)
            if len(invalid) > 0:
                raise ValueError(
                    f"{mask_path} contains unexpected labels {invalid.tolist()}"
                )

            if self.augment:
                if random.random() < 0.5:
                    image = np.flipud(image).copy()
                    mask = np.flipud(mask).copy()

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
                    mask = np.rint(mask).astype("int32")

            # Preserve the original U-Net data convention.
            # EfficientNet conversion to [0,255] happens inside its model branch.
            image = image / 255.0

            mapped_mask = encode_raw_mask(mask)
            one_hot = to_categorical(mapped_mask, num_classes=N_CLASSES)

            images.append(image)
            masks.append(one_hot)

        return (
            np.asarray(images, dtype="float32"),
            np.asarray(masks, dtype="float32"),
        )


train_gen = RasterGenerator(
    training_pairs,
    shuffle=True,
    augment=True,
)

new_val_gen = RasterGenerator(
    new_val_pairs,
    shuffle=False,
    augment=False,
)

old_monitor_gen = RasterGenerator(
    old_monitor_pairs,
    shuffle=False,
    augment=False,
)


# ============================================================
# 13. WASTE METRICS
# ============================================================

def precision_waste(y_true, y_pred, smooth=1e-7):
    y_true_waste = tf.cast(y_true[..., WASTE_CLASS_IDX], tf.float32)
    y_pred_waste = tf.cast(
        tf.argmax(y_pred, axis=-1) == WASTE_CLASS_IDX,
        tf.float32,
    )
    tp = tf.reduce_sum(y_true_waste * y_pred_waste)
    fp = tf.reduce_sum((1.0 - y_true_waste) * y_pred_waste)
    return (tp + smooth) / (tp + fp + smooth)


def recall_waste(y_true, y_pred, smooth=1e-7):
    y_true_waste = tf.cast(y_true[..., WASTE_CLASS_IDX], tf.float32)
    y_pred_waste = tf.cast(
        tf.argmax(y_pred, axis=-1) == WASTE_CLASS_IDX,
        tf.float32,
    )
    tp = tf.reduce_sum(y_true_waste * y_pred_waste)
    fn = tf.reduce_sum(y_true_waste * (1.0 - y_pred_waste))
    return (tp + smooth) / (tp + fn + smooth)


def f1_waste(y_true, y_pred, smooth=1e-7):
    p = precision_waste(y_true, y_pred, smooth)
    r = recall_waste(y_true, y_pred, smooth)
    return 2.0 * p * r / (p + r + smooth)


def iou_waste(y_true, y_pred, smooth=1e-7):
    y_true_waste = tf.cast(y_true[..., WASTE_CLASS_IDX], tf.float32)
    y_pred_waste = tf.cast(
        tf.argmax(y_pred, axis=-1) == WASTE_CLASS_IDX,
        tf.float32,
    )
    intersection = tf.reduce_sum(y_true_waste * y_pred_waste)
    union = (
        tf.reduce_sum(y_true_waste)
        + tf.reduce_sum(y_pred_waste)
        - intersection
    )
    return (intersection + smooth) / (union + smooth)


# ============================================================
# 14. EXTERNAL WASTE EVALUATION
# ============================================================

def evaluate_waste(
    model,
    generator,
    dataset_name,
    probability_threshold=0.50,
):
    y_true_all = []
    waste_probability_all = []
    argmax_all = []

    for batch_i in range(len(generator)):
        x, y = generator[batch_i]
        pred = model.predict(x, verbose=0)

        true_waste = (
            y[..., WASTE_CLASS_IDX]
            .reshape(-1)
            .astype("uint8")
        )

        waste_prob = pred[..., WASTE_CLASS_IDX].reshape(-1)

        pred_argmax = (
            np.argmax(pred, axis=-1) == WASTE_CLASS_IDX
        ).reshape(-1).astype("uint8")

        y_true_all.append(true_waste)
        waste_probability_all.append(waste_prob)
        argmax_all.append(pred_argmax)

    y_true = np.concatenate(y_true_all)
    waste_prob = np.concatenate(waste_probability_all)
    pred_argmax = np.concatenate(argmax_all)

    def binary_metrics(truth, pred):
        tp = np.sum((truth == 1) & (pred == 1))
        fp = np.sum((truth == 0) & (pred == 1))
        fn = np.sum((truth == 1) & (pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "IoU": iou,
            "TP_pixels": int(tp),
            "FP_pixels": int(fp),
            "FN_pixels": int(fn),
        }

    argmax_metrics = binary_metrics(y_true, pred_argmax)

    pred_fixed = (waste_prob >= probability_threshold).astype("uint8")
    fixed_metrics = binary_metrics(y_true, pred_fixed)

    best_threshold = np.nan
    optimal_metrics = {
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "IoU": np.nan,
        "TP_pixels": np.nan,
        "FP_pixels": np.nan,
        "FN_pixels": np.nan,
    }

    if np.any(y_true == 1) and np.any(y_true == 0):
        precision_curve, recall_curve, thresholds = precision_recall_curve(
            y_true,
            waste_prob,
        )

        if len(thresholds) > 0:
            p = precision_curve[:-1]
            r = recall_curve[:-1]
            f1 = 2 * p * r / (p + r + 1e-7)
            best_index = np.argmax(f1)
            best_threshold = float(thresholds[best_index])
            pred_optimal = (waste_prob >= best_threshold).astype("uint8")
            optimal_metrics = binary_metrics(y_true, pred_optimal)

    results = []
    for decision, metrics, threshold in [
        ("multiclass_argmax", argmax_metrics, np.nan),
        ("waste_probability_0.50", fixed_metrics, probability_threshold),
        ("diagnostic_optimal_threshold", optimal_metrics, best_threshold),
    ]:
        results.append(
            {
                "Dataset": dataset_name,
                "Decision": decision,
                "Threshold": threshold,
                **metrics,
            }
        )

    return results


# ============================================================
# 15. EFFICIENTNETB0 U-NET MODEL
# ============================================================

def conv_block(x, filters, name):
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def decoder_block(x, skip, filters, name):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size=2,
        strides=2,
        padding="same",
        name=f"{name}_up",
    )(x)

    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = conv_block(x, filters, name=f"{name}_conv")
    return x


def build_efficientnetb0_unet():
    """
    Build an EfficientNetB0 encoder + U-Net-style decoder.

    Input to the overall segmentation model remains [0,1] to match the
    original workflow. A Rescaling(255) layer is inserted before EfficientNet.
    """

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    )

    # Multi-scale EfficientNet features. For a 256x256 input these are
    # approximately 128, 64, 32, 16 and 8 pixel spatial resolutions.
    feature_encoder = tf.keras.Model(
        inputs=base.input,
        outputs=[
            base.get_layer("block2a_expand_activation").output,
            base.get_layer("block3a_expand_activation").output,
            base.get_layer("block4a_expand_activation").output,
            base.get_layer("block6a_expand_activation").output,
            base.get_layer("top_activation").output,
        ],
        name="efficientnetb0_encoder",
    )

    # Stage 1 starts with the entire pretrained encoder frozen.
    feature_encoder.trainable = False

    inputs = layers.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        name="rgb_input_0_1",
    )

    # Existing generator gives [0,1]. EfficientNet expects [0,255].
    efficientnet_input = layers.Rescaling(
        scale=255.0,
        name="restore_0_255_for_efficientnet",
    )(inputs)

    # training=False is intentional: keep EfficientNet BatchNorm statistics
    # in inference mode even after selected convolutional layers are unfrozen.
    s1, s2, s3, s4, bottleneck = feature_encoder(
        efficientnet_input,
        training=False,
    )

    x = decoder_block(bottleneck, s4, 256, "decoder4")
    x = decoder_block(x, s3, 128, "decoder3")
    x = decoder_block(x, s2, 64, "decoder2")
    x = decoder_block(x, s1, 32, "decoder1")

    # Last 128 -> 256 step has no EfficientNet skip connection.
    x = layers.Conv2DTranspose(
        16,
        kernel_size=2,
        strides=2,
        padding="same",
        name="decoder0_up",
    )(x)
    x = conv_block(x, 16, name="decoder0_conv")

    outputs = layers.Conv2D(
        N_CLASSES,
        kernel_size=1,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="EfficientNetB0_UNet_10class",
    )

    return model, feature_encoder


# ============================================================
# 16. TRAINABILITY HELPERS
# ============================================================

def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            precision_waste,
            recall_waste,
            f1_waste,
            iou_waste,
        ],
    )


def unfreeze_upper_efficientnet(encoder, start_layer_name):
    """Unfreeze only upper EfficientNet layers; keep every BN layer frozen."""

    encoder.trainable = True

    found = False
    trainable_started = False

    for layer in encoder.layers:
        if layer.name == start_layer_name:
            found = True
            trainable_started = True

        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = trainable_started

    if not found:
        raise ValueError(
            f"Fine-tuning start layer '{start_layer_name}' was not found."
        )


def print_trainability(model, encoder, label):
    model_trainable = np.sum(
        [np.prod(v.shape) for v in model.trainable_weights]
    )
    model_nontrainable = np.sum(
        [np.prod(v.shape) for v in model.non_trainable_weights]
    )

    encoder_trainable_layers = sum(layer.trainable for layer in encoder.layers)

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    print(f"Encoder layers              : {len(encoder.layers)}")
    print(f"Trainable encoder layers    : {encoder_trainable_layers}")
    print(f"Trainable model parameters  : {model_trainable:,}")
    print(f"Non-trainable parameters    : {model_nontrainable:,}")


# ============================================================
# 17. BASELINE ORIGINAL U-NET
# ============================================================

comparison_results = []

if ORIGINAL_MODEL_PATH.exists():
    print("\n" + "=" * 70)
    print("LOADING ORIGINAL U-NET BASELINE")
    print("=" * 70)

    original_model = tf.keras.models.load_model(
        ORIGINAL_MODEL_PATH,
        compile=False,
    )

    if original_model.output_shape[-1] != N_CLASSES:
        raise ValueError("Loaded original model does not have 10 output classes.")

    for dataset_name, generator in [
        ("new_validation", new_val_gen),
        ("old_monitor", old_monitor_gen),
    ]:
        rows = evaluate_waste(
            original_model,
            generator,
            dataset_name,
        )
        for row in rows:
            row["Model"] = "original_custom_unet"
        comparison_results.extend(rows)
else:
    print(
        "\nWARNING: Original model not found. "
        "EfficientNet training will continue without baseline evaluation."
    )


# ============================================================
# 18. BUILD EFFICIENTNETB0 MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING IMAGENET-PRETRAINED EFFICIENTNETB0 U-NET")
print("=" * 70)

model, encoder = build_efficientnetb0_unet()

print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

if model.output_shape[-1] != N_CLASSES:
    raise ValueError("EfficientNet model does not have 10 output classes.")


# ============================================================
# 19. STAGE 1: TRAIN DECODER, FREEZE EFFICIENTNET
# ============================================================

compile_model(model, STAGE1_LR)
print_trainability(
    model,
    encoder,
    "STAGE 1: FROZEN EFFICIENTNET / TRAIN DECODER",
)

stage1_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=STAGE1_BEST_WEIGHTS,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        OUTPUT_DIR / "stage1_history.csv"
    ),
]

model.fit(
    train_gen,
    validation_data=new_val_gen,
    epochs=STAGE1_EPOCHS,
    callbacks=stage1_callbacks,
    verbose=1,
)

# Ensure Stage 2 starts from the best Stage 1 state.
model.load_weights(STAGE1_BEST_WEIGHTS)


# ============================================================
# 20. STAGE 2: FINE-TUNE UPPER EFFICIENTNET BLOCKS
# ============================================================

unfreeze_upper_efficientnet(
    encoder,
    FINE_TUNE_FROM,
)

# Recompile after changing trainability.
compile_model(model, STAGE2_LR)

print_trainability(
    model,
    encoder,
    f"STAGE 2: FINE-TUNE FROM {FINE_TUNE_FROM}",
)

stage2_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=STAGE2_BEST_WEIGHTS,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        OUTPUT_DIR / "stage2_history.csv"
    ),
]

model.fit(
    train_gen,
    validation_data=new_val_gen,
    epochs=STAGE2_EPOCHS,
    callbacks=stage2_callbacks,
    verbose=1,
)

model.load_weights(STAGE2_BEST_WEIGHTS)
model.save(FINAL_MODEL_PATH)


# ============================================================
# 21. POST-TRAINING EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("EFFICIENTNETB0 MODEL: PERFORMANCE")
print("=" * 70)

for dataset_name, generator in [
    ("new_validation", new_val_gen),
    ("old_monitor", old_monitor_gen),
]:
    rows = evaluate_waste(
        model,
        generator,
        dataset_name,
    )
    for row in rows:
        row["Model"] = "efficientnetb0_imagenet_unet"
    comparison_results.extend(rows)


# ============================================================
# 22. SAVE COMPARISON
# ============================================================

comparison_df = pd.DataFrame(comparison_results)

comparison_columns = [
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
    "FN_pixels",
]

comparison_df = comparison_df[comparison_columns]
comparison_df.to_csv(COMPARISON_CSV, index=False)

print("\n" + "=" * 70)
print("ORIGINAL vs EFFICIENTNETB0 WASTE PERFORMANCE")
print("=" * 70)
print(comparison_df.to_string(index=False))

print("\n" + "=" * 70)
print("TRANSFER LEARNING COMPLETE")
print("=" * 70)
print("Best EfficientNetB0 model:")
print(FINAL_MODEL_PATH)
print("\nWaste comparison:")
print(COMPARISON_CSV)
print("\nPartition record:")
print(PARTITION_CSV)
print("\nStage 1 history:")
print(OUTPUT_DIR / "stage1_history.csv")
print("\nStage 2 history:")
print(OUTPUT_DIR / "stage2_history.csv")
print("=" * 70)
