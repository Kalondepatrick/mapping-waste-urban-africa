import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import rotate
from skimage.transform import resize
from sklearn.metrics import average_precision_score, precision_recall_curve

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence, to_categorical


# -----------------------------
# Configuration
# -----------------------------
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 8
EPOCHS = 500

BASE_DATA_PATH = (
    r"C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa\data\tiled_data"
    r"\partitioned\modelDevelopment"
)
IMG_DIR = os.path.join(BASE_DATA_PATH, "images")
MASK_DIR = os.path.join(BASE_DATA_PATH, "masks")
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# IMPORTANT:
# The original multiclass mask uses labels 1-10.
# Waste is raw raster label 1.
# TensorFlow / one-hot encoding internally uses channels 0-(N_CLASSES-1).
WASTE_RAW_LABEL = 1


# -----------------------------
# Dynamic Label Discovery
# -----------------------------
mask_files = [
    f
    for f in os.listdir(MASK_DIR)
    if f.endswith(".tif") and not f.endswith(".aux.xml")
]

if not mask_files:
    raise FileNotFoundError(f"No .tif files found in {MASK_DIR}")

unique_labels = set()
for f in mask_files:
    with rasterio.open(os.path.join(MASK_DIR, f)) as src:
        band = src.read(1)
        unique_labels.update(int(v) for v in np.unique(band))

LABEL_LIST = sorted(unique_labels)
N_CLASSES = len(LABEL_LIST)

# Raw raster label -> internal model channel
# Example for labels 1-10:
# {1: 0, 2: 1, ..., 10: 9}
LABEL_MAP = {raw_label: model_idx for model_idx, raw_label in enumerate(LABEL_LIST)}

# Internal model channel -> original raw raster label
INDEX_TO_LABEL = {model_idx: raw_label for raw_label, model_idx in LABEL_MAP.items()}

if WASTE_RAW_LABEL not in LABEL_MAP:
    raise ValueError(
        f"Waste raw label {WASTE_RAW_LABEL} was not found in the masks. "
        f"Detected labels: {LABEL_LIST}"
    )

# Waste is raw raster class 1, but its model channel is determined by LABEL_MAP.
# If raster labels are exactly 1-10, this will correctly be channel 0.
WASTE_CLASS_IDX = LABEL_MAP[WASTE_RAW_LABEL]

print("\n=========================================")
print("          CLASS LABEL MAPPING")
print("=========================================")
print(f"Detected raw raster labels : {LABEL_LIST}")
print(f"Number of classes          : {N_CLASSES}")
print(f"Raw -> model mapping       : {LABEL_MAP}")
print(f"Model -> raw mapping       : {INDEX_TO_LABEL}")
print(f"Waste raw raster label     : {WASTE_RAW_LABEL}")
print(f"Waste model channel        : {WASTE_CLASS_IDX}")
print("=========================================\n")

# Save the class mapping for reproducibility
mapping_csv_path = os.path.join(RESULTS_DIR, "class_label_mapping.csv")
mapping_df = pd.DataFrame(
    [
        {
            "Raw_Raster_Label": raw_label,
            "Model_Channel": LABEL_MAP[raw_label],
            "Class": "Waste" if raw_label == WASTE_RAW_LABEL else "Other",
        }
        for raw_label in LABEL_LIST
    ]
)
mapping_df.to_csv(mapping_csv_path, index=False)


# -----------------------------
# Waste-class Metrics
# -----------------------------
def precision_waste(y_true, y_pred, smooth=1e-7):
    """Precision for waste only."""
    y_true_waste = tf.cast(y_true[..., WASTE_CLASS_IDX], tf.float32)
    y_pred_waste = tf.cast(
        tf.argmax(y_pred, axis=-1) == WASTE_CLASS_IDX,
        tf.float32,
    )

    tp = tf.reduce_sum(y_true_waste * y_pred_waste)
    fp = tf.reduce_sum((1.0 - y_true_waste) * y_pred_waste)
    return (tp + smooth) / (tp + fp + smooth)


def recall_waste(y_true, y_pred, smooth=1e-7):
    """Recall for waste only."""
    y_true_waste = tf.cast(y_true[..., WASTE_CLASS_IDX], tf.float32)
    y_pred_waste = tf.cast(
        tf.argmax(y_pred, axis=-1) == WASTE_CLASS_IDX,
        tf.float32,
    )

    tp = tf.reduce_sum(y_true_waste * y_pred_waste)
    fn = tf.reduce_sum(y_true_waste * (1.0 - y_pred_waste))
    return (tp + smooth) / (tp + fn + smooth)


def f1_waste(y_true, y_pred, smooth=1e-7):
    """F1-score for waste only."""
    p = precision_waste(y_true, y_pred, smooth)
    r = recall_waste(y_true, y_pred, smooth)
    return (2.0 * p * r) / (p + r + smooth)


def iou_waste(y_true, y_pred, smooth=1e-7):
    """Intersection-over-Union for waste only."""
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


# -----------------------------
# Helpers for Encoding / Decoding Labels
# -----------------------------
def encode_raw_mask(mask):
    """
    Convert original raster labels (e.g. 1-10) to model indices (0-9).
    """
    mapped_mask = np.full(mask.shape, -1, dtype="int32")

    for raw_label, model_idx in LABEL_MAP.items():
        mapped_mask[mask == raw_label] = model_idx

    if np.any(mapped_mask < 0):
        unknown_values = np.unique(mask[mapped_mask < 0])
        raise ValueError(
            "Mask contains values that are not present in LABEL_MAP: "
            f"{unknown_values.tolist()}"
        )

    return mapped_mask


def decode_model_indices(index_mask):
    """
    Convert model indices (0-9) back to original raster labels (1-10).
    """
    raw_mask = np.full(index_mask.shape, -1, dtype="int32")

    for model_idx, raw_label in INDEX_TO_LABEL.items():
        raw_mask[index_mask == model_idx] = raw_label

    if np.any(raw_mask < 0):
        unknown_indices = np.unique(index_mask[raw_mask < 0])
        raise ValueError(
            "Prediction contains model indices that are not present in "
            f"INDEX_TO_LABEL: {unknown_indices.tolist()}"
        )

    return raw_mask


def probabilities_to_raw_labels(pred_probs):
    """
    Convert softmax prediction probabilities directly to original raster labels.

    Input shape:
        (..., N_CLASSES)

    Output:
        Raster labels in the original class system, e.g. 1-10.
    """
    predicted_indices = np.argmax(pred_probs, axis=-1)
    return decode_model_indices(predicted_indices)


# -----------------------------
# Data Generator
# -----------------------------
class RasterDataGenerator(Sequence):
    def __init__(
        self,
        img_dir,
        mask_dir,
        img_files,
        mask_files,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = list(img_files)
        self.mask_files = list(mask_files)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        if len(self.img_files) != len(self.mask_files):
            raise ValueError(
                "Image and mask file lists must contain the same number of files."
            )

        self.indices = np.arange(len(self.img_files))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        imgs, masks = [], []

        for i in batch_idx:
            img_file = self.img_files[i]
            mask_file = self.mask_files[i]

            with rasterio.open(os.path.join(self.img_dir, img_file)) as src:
                img = (
                    src.read([1, 2, 3])
                    .transpose(1, 2, 0)
                    .astype("float32")
                )

            with rasterio.open(os.path.join(self.mask_dir, mask_file)) as src:
                mask = src.read(1)

            img = resize(
                img,
                (IMG_HEIGHT, IMG_WIDTH),
                preserve_range=True,
            )

            # Nearest-neighbour resizing preserves categorical class labels.
            mask = resize(
                mask,
                (IMG_HEIGHT, IMG_WIDTH),
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )

            # Make sure the resized categorical mask contains integer labels.
            mask = np.rint(mask).astype("int32")

            if self.augment:
                if random.random() < 0.5:
                    img = np.flipud(img)
                    mask = np.flipud(mask)

                if random.random() < 0.5:
                    angle = random.uniform(-15, 15)

                    img = rotate(
                        img,
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

            img = img / 255.0

            # Convert original raster classes (1-10) to model channels (0-9).
            mapped_mask = encode_raw_mask(mask)

            imgs.append(img)
            masks.append(
                to_categorical(mapped_mask, num_classes=N_CLASSES)
            )

        return np.array(imgs), np.array(masks)


# -----------------------------
# U-Net Model
# -----------------------------
def build_model(n_classes, init_filters=16):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

    def conv_block(x, f):
        x = Conv2D(f, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(f, 3, padding="same", activation="relu")(x)
        return BatchNormalization()(x)

    s1 = conv_block(inputs, init_filters)
    p1 = MaxPooling2D((2, 2))(s1)

    s2 = conv_block(p1, init_filters * 2)
    p2 = MaxPooling2D((2, 2))(s2)

    s3 = conv_block(p2, init_filters * 4)
    p3 = MaxPooling2D((2, 2))(s3)

    b1 = conv_block(p3, init_filters * 8)

    d1_up = Conv2DTranspose(
        init_filters * 4,
        2,
        strides=2,
        padding="same",
    )(b1)
    d1 = Concatenate()([d1_up, s3])
    d1 = conv_block(d1, init_filters * 4)

    d2_up = Conv2DTranspose(
        init_filters * 2,
        2,
        strides=2,
        padding="same",
    )(d1)
    d2 = Concatenate()([d2_up, s2])
    d2 = conv_block(d2, init_filters * 2)

    d3_up = Conv2DTranspose(
        init_filters,
        2,
        strides=2,
        padding="same",
    )(d2)
    d3 = Concatenate()([d3_up, s1])
    d3 = conv_block(d3, init_filters)

    outputs = Conv2D(n_classes, 1, activation="softmax")(d3)

    return Model(inputs, outputs, name="UNet_Filters_16")


# -----------------------------
# Train-Validation Split
# -----------------------------
all_img = sorted(
    [
        f
        for f in os.listdir(IMG_DIR)
        if f.endswith(".tif") and not f.endswith(".aux.xml")
    ]
)

all_msk = sorted(
    [
        f
        for f in os.listdir(MASK_DIR)
        if f.endswith(".tif") and not f.endswith(".aux.xml")
    ]
)

# Pair by filename rather than relying on two independently sorted lists.
common_files = sorted(set(all_img).intersection(all_msk))

if not common_files:
    raise FileNotFoundError(
        "No matching .tif filenames were found between image and mask folders."
    )

missing_masks = sorted(set(all_img) - set(all_msk))
missing_images = sorted(set(all_msk) - set(all_img))

if missing_masks:
    print(f"Warning: {len(missing_masks)} images have no matching mask.")

if missing_images:
    print(f"Warning: {len(missing_images)} masks have no matching image.")

random.shuffle(common_files)

split_idx = int(len(common_files) * 0.8)
train_files = common_files[:split_idx]
val_files = common_files[split_idx:]

if len(train_files) == 0 or len(val_files) == 0:
    raise ValueError(
        "The train-validation split produced an empty training or validation set."
    )

train_imgs = train_files
train_msks = train_files
val_imgs = val_files
val_msks = val_files

print(f"Training image-mask pairs   : {len(train_files)}")
print(f"Validation image-mask pairs : {len(val_files)}")

train_gen = RasterDataGenerator(
    IMG_DIR,
    MASK_DIR,
    train_imgs,
    train_msks,
)

val_gen = RasterDataGenerator(
    IMG_DIR,
    MASK_DIR,
    val_imgs,
    val_msks,
    shuffle=False,
    augment=False,
)


# -----------------------------
# Callbacks & Execution
# -----------------------------
EXP_NAME = "filters_16"
model_save_path = os.path.join(RESULTS_DIR, f"{EXP_NAME}_best.keras")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1,
)

model_checkpoint = ModelCheckpoint(
    filepath=model_save_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

model = build_model(N_CLASSES)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        precision_waste,
        recall_waste,
        f1_waste,
        iou_waste,
    ],
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        reduce_lr,
        model_checkpoint,
    ],
)

final_model_path = os.path.join(RESULTS_DIR, f"{EXP_NAME}_final.keras")
model.save(final_model_path)

print(
    f"Model saved to:\n"
    f" - Best:  {model_save_path}\n"
    f" - Final: {final_model_path}"
)


# -----------------------------
# Waste Probability Extraction
# & Threshold Optimization
# -----------------------------
print(
    "\nExtracting WASTE prediction probabilities "
    "for threshold optimization..."
)
print(
    f"Waste raw raster label = {WASTE_RAW_LABEL}; "
    f"model probability channel = {WASTE_CLASS_IDX}"
)

y_true_list = []
y_pred_probs_list = []

for i in range(len(val_gen)):
    batch_imgs, batch_masks = val_gen[i]
    preds = model.predict(batch_imgs, verbose=0)

    # Ground-truth waste presence from the one-hot waste channel.
    y_true_list.append(
        batch_masks[..., WASTE_CLASS_IDX].flatten()
    )

    # Predicted probability specifically for waste.
    y_pred_probs_list.append(
        preds[..., WASTE_CLASS_IDX].flatten()
    )

y_true_all = np.concatenate(y_true_list)
y_pred_probs_all = np.concatenate(y_pred_probs_list)

# Compute precision-recall curve for WASTE only.
precision, recall, thresholds = precision_recall_curve(
    y_true_all,
    y_pred_probs_all,
)

ap_score = average_precision_score(
    y_true_all,
    y_pred_probs_all,
)

if len(thresholds) == 0:
    raise ValueError(
        "No thresholds were returned. Check whether the validation data "
        "contains both waste and non-waste pixels."
    )

# Precision and recall arrays contain one extra point compared with thresholds.
p_matched = precision[:-1]
r_matched = recall[:-1]

# Compute F1 across thresholds.
f1_scores = (
    2.0 * p_matched * r_matched
) / (
    p_matched + r_matched + 1e-7
)

# Pick threshold with maximum waste F1.
best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])
best_prec = float(p_matched[best_idx])
best_rec = float(r_matched[best_idx])
best_f1 = float(f1_scores[best_idx])

# Calculate waste IoU at the optimal threshold.
opt_pred_binary = (
    y_pred_probs_all >= best_threshold
).astype(np.float32)

intersection = np.sum(y_true_all * opt_pred_binary)
union = (
    np.sum(y_true_all)
    + np.sum(opt_pred_binary)
    - intersection
)

best_iou = float(
    (intersection + 1e-7) / (union + 1e-7)
)

print("\n=========================================")
print("      WASTE THRESHOLD SELECTION")
print("=========================================")
print(f"Waste raw raster label     : {WASTE_RAW_LABEL}")
print(f"Waste model channel        : {WASTE_CLASS_IDX}")
print(f"Optimal Threshold (t*)     : {best_threshold:.4f}")
print(f"Precision @ t*             : {best_prec:.4f}")
print(f"Recall @ t*                : {best_rec:.4f}")
print(f"Max F1-Score @ t*          : {best_f1:.4f}")
print(f"IoU Score @ t*             : {best_iou:.4f}")
print(f"Average Precision          : {ap_score:.4f}")
print("=========================================\n")


# -----------------------------
# Separate Waste Diagnostic Plots
# -----------------------------
# Each diagnostic is saved as its own publication-ready figure:
#   1. Precision-Recall curve
#   2. Precision vs probability threshold
#   3. Recall vs probability threshold
#   4. F1 vs probability threshold

# Waste prevalence is the no-skill baseline for the PR curve.
waste_prevalence = float(np.mean(y_true_all))

# 1) Precision-Recall Curve
fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    recall,
    precision,
    lw=2,
    label=f"Waste PR (AP = {ap_score:.4f})",
)

ax.axhline(
    y=waste_prevalence,
    linestyle="--",
    lw=1.5,
    alpha=0.7,
    label=f"No-skill baseline = {waste_prevalence:.4f}",
)

ax.scatter(
    [best_rec],
    [best_prec],
    s=80,
    zorder=5,
    label=f"Optimal F1 point (t = {best_threshold:.4f})",
)

ax.set_xlabel("Recall (Sensitivity)")
ax.set_ylabel("Precision (PPV)")
ax.set_title("Precision-Recall Curve: Waste")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="lower left")

plt.tight_layout()

pr_curve_path = os.path.join(
    RESULTS_DIR,
    f"{EXP_NAME}_waste_precision_recall_curve.png",
)
plt.savefig(pr_curve_path, dpi=300, bbox_inches="tight")
plt.close()


# 2) Precision vs Threshold
fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    thresholds,
    p_matched,
    lw=2,
    label="Waste Precision",
)

ax.axvline(
    x=best_threshold,
    linestyle="--",
    lw=2,
    label=f"Optimal threshold = {best_threshold:.4f}",
)

ax.scatter(
    [best_threshold],
    [best_prec],
    s=80,
    zorder=5,
    label=f"Precision @ t* = {best_prec:.4f}",
)

ax.set_xlabel("Waste Probability Threshold")
ax.set_ylabel("Precision (PPV)")
ax.set_title("Waste Precision vs Probability Threshold")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="best")

plt.tight_layout()

precision_threshold_path = os.path.join(
    RESULTS_DIR,
    f"{EXP_NAME}_waste_precision_vs_threshold.png",
)
plt.savefig(precision_threshold_path, dpi=300, bbox_inches="tight")
plt.close()


# 3) Recall vs Threshold
fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    thresholds,
    r_matched,
    lw=2,
    label="Waste Recall",
)

ax.axvline(
    x=best_threshold,
    linestyle="--",
    lw=2,
    label=f"Optimal threshold = {best_threshold:.4f}",
)

ax.scatter(
    [best_threshold],
    [best_rec],
    s=80,
    zorder=5,
    label=f"Recall @ t* = {best_rec:.4f}",
)

ax.set_xlabel("Waste Probability Threshold")
ax.set_ylabel("Recall (Sensitivity)")
ax.set_title("Waste Recall vs Probability Threshold")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="best")

plt.tight_layout()

recall_threshold_path = os.path.join(
    RESULTS_DIR,
    f"{EXP_NAME}_waste_recall_vs_threshold.png",
)
plt.savefig(recall_threshold_path, dpi=300, bbox_inches="tight")
plt.close()


# 4) F1 vs Threshold
fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    thresholds,
    f1_scores,
    lw=2,
    label="Waste F1",
)

ax.axvline(
    x=best_threshold,
    linestyle="--",
    lw=2,
    label=f"Optimal threshold = {best_threshold:.4f}",
)

ax.scatter(
    [best_threshold],
    [best_f1],
    s=80,
    zorder=5,
    label=f"Peak F1 = {best_f1:.4f}",
)

ax.set_xlabel("Waste Probability Threshold")
ax.set_ylabel("F1 Score")
ax.set_title("Waste F1 vs Probability Threshold")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="best")

plt.tight_layout()

f1_threshold_path = os.path.join(
    RESULTS_DIR,
    f"{EXP_NAME}_waste_f1_vs_threshold.png",
)
plt.savefig(f1_threshold_path, dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# Save Waste Results
# -----------------------------
csv_path = os.path.join(
    RESULTS_DIR,
    f"{EXP_NAME}_waste_metrics.csv",
)

results_df = pd.DataFrame(
    [
        {
            "Experiment": EXP_NAME,
            "Waste_Raw_Raster_Label": WASTE_RAW_LABEL,
            "Waste_Model_Channel": WASTE_CLASS_IDX,
            "Optimal_Threshold": round(best_threshold, 4),
            "Precision": round(best_prec, 4),
            "Recall": round(best_rec, 4),
            "F1_Score": round(best_f1, 4),
            "IoU": round(best_iou, 4),
            "Average_Precision": round(ap_score, 4),
        }
    ]
)

results_df.to_csv(csv_path, index=False)

txt_path = os.path.join(
    RESULTS_DIR,
    f"{EXP_NAME}_waste_metrics.txt",
)

with open(txt_path, "w") as f:
    f.write(f"--- Waste Evaluation Results: {EXP_NAME} ---\n")
    f.write(f"Waste Raw Raster Label      : {WASTE_RAW_LABEL}\n")
    f.write(f"Waste Model Channel         : {WASTE_CLASS_IDX}\n")
    f.write(f"Optimal Probability Threshold: {best_threshold:.4f}\n")
    f.write(f"Precision @ Optimal Threshold: {best_prec:.4f}\n")
    f.write(f"Recall @ Optimal Threshold   : {best_rec:.4f}\n")
    f.write(f"F1 Score @ Optimal Threshold : {best_f1:.4f}\n")
    f.write(f"IoU @ Optimal Threshold      : {best_iou:.4f}\n")
    f.write(f"Average Precision            : {ap_score:.4f}\n")

print("Waste threshold metrics successfully saved to:")
print(f" - CSV:          {csv_path}")
print(f" - TXT:          {txt_path}")
print(f" - PR curve:      {pr_curve_path}")
print(f" - Precision plot:{precision_threshold_path}")
print(f" - Recall plot:   {recall_threshold_path}")
print(f" - F1 plot:       {f1_threshold_path}")
print(f" - Class mapping:{mapping_csv_path}")


# -----------------------------
# IMPORTANT FOR FUTURE RASTER PREDICTION
# -----------------------------
# The model's argmax prediction is an internal index (0 to N_CLASSES-1).
# DO NOT save that index raster directly if you want the original class labels.
# Convert it back to the raw raster classes first:
#
#     pred_probs = model.predict(image_batch)
#     pred_raw_labels = probabilities_to_raw_labels(pred_probs)
#
# For your 10-class masks this means, for example:
#     model channel 0 -> original raster label 1 (WASTE)
#     model channel 1 -> original raster label 2
#     ...
#     model channel 9 -> original raster label 10
#
# Therefore waste remains CLASS 1 in the original raster label system,
# even though the neural network uses channel 0 internally.
