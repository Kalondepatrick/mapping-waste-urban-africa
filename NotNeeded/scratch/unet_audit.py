"""
Waste-pile data audit for a previously trained multiclass U-Net
===============================================================

Purpose
-------
This script DOES NOT retrain the model.

It:
1. Loads the previously saved Keras U-Net.
2. Runs inference on every matching image/mask pair that was eligible for
   training/validation.
3. Finds every connected ground-truth WASTE object (raw raster class 1).
4. Measures how strongly the saved model supports each labelled waste pile.
5. Identifies the dominant class each difficult pile is confused with.
6. Ranks piles for manual review.
7. Searches for high-confidence predicted waste regions that are NOT labelled
   as waste, which can reveal possible missing waste annotations.
8. Saves visual panels for the most suspicious labelled piles and possible
   missing annotations.
9. Performs basic image/mask consistency checks.

IMPORTANT
---------
The original training script randomly shuffled files before the 80/20 split,
but did not save the split or fix a seed. Therefore the exact historical
train/validation membership cannot be reconstructed from the saved model alone.

This script audits ALL matching pairs. Since every matching pair was assigned
to either training or validation in the original script, this covers both.

Interpretation
--------------
A low-performing labelled pile is NOT automatically a bad label. It is a
"review priority": inspect it for possible annotation mistakes, unusual waste
appearance, shadows, mixed materials, edge effects, etc.

Likewise, a high-confidence prediction outside the waste label is NOT
automatically missing ground truth. It is a candidate for manual review.
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from matplotlib.patches import Rectangle
from scipy.ndimage import label as connected_components
from skimage.transform import resize


# ============================================================
# CONFIGURATION
# ============================================================

IMG_HEIGHT = 256
IMG_WIDTH = 256

BASE_DATA_PATH = (
    r"C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa\data\tiled_data"
    r"\partitioned\modelDevelopment"
)

IMG_DIR = os.path.join(BASE_DATA_PATH, "images")
MASK_DIR = os.path.join(BASE_DATA_PATH, "masks")

RESULTS_DIR = "./results"

# Use the best saved model from the original training script.
MODEL_PATH = os.path.join(RESULTS_DIR, "filters_16_best.keras")

# Original raster label for waste.
WASTE_RAW_LABEL = 1

# Your expected multiclass raster labels.
EXPECTED_RAW_LABELS = list(range(1, 11))

# If None, the script tries to read the optimal threshold from the
# waste metrics CSV created by the training script. If that file is absent,
# it falls back to 0.50.
WASTE_THRESHOLD = None

THRESHOLD_METRICS_CSV = os.path.join(
    RESULTS_DIR,
    "filters_16_waste_metrics.csv",
)

# Connected-component settings.
# 8-connected pixels are treated as belonging to one object.
CONNECTIVITY_STRUCTURE = np.ones((3, 3), dtype=np.uint8)

# Report every labelled waste component, including very small ones.
MIN_GT_PILE_PIXELS = 1

# Review flags for labelled piles.
LOW_MEAN_WASTE_PROB = 0.30
LOW_PILE_RECALL = 0.30
STRONG_NEGATIVE_MARGIN = -0.20
HIGH_NORMALIZED_ENTROPY = 0.65
TINY_PILE_PIXELS = 10

# Search for possible missing waste labels:
# only pixels with very high model confidence are considered.
MISSING_LABEL_PROB_THRESHOLD = 0.90
MIN_MISSING_COMPONENT_PIXELS = 10

# Number of visual panels to save.
TOP_N_DIFFICULT_PILES = 30
TOP_N_MISSING_LABEL_CANDIDATES = 30

# Number of pixels around each object's bounding box in saved diagnostic crops.
VISUAL_PADDING = 25

# Optional human-readable names.
# Fill these in if you want, e.g.
# CLASS_NAMES = {
#     1: "Waste",
#     2: "Vegetation",
#     3: "Built-up",
#     ...
# }
CLASS_NAMES = {
    1: "Waste",
}


# ============================================================
# OUTPUT FOLDERS
# ============================================================

AUDIT_DIR = os.path.join(RESULTS_DIR, "waste_pile_data_audit")
DIFFICULT_PILE_DIR = os.path.join(AUDIT_DIR, "difficult_labelled_piles")
MISSING_LABEL_DIR = os.path.join(AUDIT_DIR, "possible_missing_waste_labels")

os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(DIFFICULT_PILE_DIR, exist_ok=True)
os.makedirs(MISSING_LABEL_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def safe_name(text):
    """Make a filename-safe string."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def class_name(raw_label):
    """Human-readable class label."""
    raw_label = int(raw_label)
    return CLASS_NAMES.get(raw_label, f"Raw class {raw_label}")


def bbox_from_binary(binary_mask):
    """
    Return bbox as:
        min_row, min_col, max_row_exclusive, max_col_exclusive
    """
    rows, cols = np.where(binary_mask)
    if len(rows) == 0:
        return None
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()) + 1,
        int(cols.max()) + 1,
    )


def padded_bbox(bbox, height, width, pad=VISUAL_PADDING):
    """Pad and clip a bounding box."""
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(height, r1 + pad),
        min(width, c1 + pad),
    )


def touches_edge(binary_mask):
    """Whether an object touches any tile boundary."""
    if not np.any(binary_mask):
        return False
    return bool(
        np.any(binary_mask[0, :])
        or np.any(binary_mask[-1, :])
        or np.any(binary_mask[:, 0])
        or np.any(binary_mask[:, -1])
    )


def normalized_entropy(probabilities):
    """
    Mean normalized softmax entropy.
    0 = very confident.
    1 = maximally uncertain across classes.
    """
    p = np.clip(probabilities, 1e-8, 1.0)
    entropy = -np.sum(p * np.log(p), axis=-1)
    denom = np.log(probabilities.shape[-1])
    return entropy / denom


def load_probability_threshold():
    """Load threshold from previous metrics CSV if possible."""
    if WASTE_THRESHOLD is not None:
        return float(WASTE_THRESHOLD), "configured manually"

    if os.path.exists(THRESHOLD_METRICS_CSV):
        try:
            df = pd.read_csv(THRESHOLD_METRICS_CSV)
            if "Optimal_Threshold" in df.columns and len(df) > 0:
                value = float(df.iloc[0]["Optimal_Threshold"])
                return value, THRESHOLD_METRICS_CSV
        except Exception as exc:
            print(
                f"Warning: could not read threshold CSV "
                f"({THRESHOLD_METRICS_CSV}): {exc}"
            )

    return 0.50, "fallback default of 0.50"


def resize_image_and_mask(img, mask):
    """
    Apply the same resizing logic used during training.
    """
    img_resized = resize(
        img,
        (IMG_HEIGHT, IMG_WIDTH),
        preserve_range=True,
    ).astype("float32")

    mask_resized = resize(
        mask,
        (IMG_HEIGHT, IMG_WIDTH),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )

    mask_resized = np.rint(mask_resized).astype("int32")

    # Same normalization as original training script.
    img_resized = img_resized / 255.0

    return img_resized, mask_resized


def get_dominant_value(values):
    """Return mode and its count."""
    values = np.asarray(values)
    if values.size == 0:
        return None, 0

    unique, counts = np.unique(values, return_counts=True)
    idx = int(np.argmax(counts))
    return int(unique[idx]), int(counts[idx])


def infer_split_from_optional_csv(filename):
    """
    Optional helper.

    If you later create:
        results/filters_16_data_split.csv

    with columns:
        filename, split

    where split is train/validation, this script will use it.

    The original training script did not create this file, so normally
    the returned value will be 'historical_split_unknown'.
    """
    split_csv = os.path.join(RESULTS_DIR, "filters_16_data_split.csv")

    if not hasattr(infer_split_from_optional_csv, "_mapping"):
        infer_split_from_optional_csv._mapping = {}

        if os.path.exists(split_csv):
            try:
                sdf = pd.read_csv(split_csv)
                if {"filename", "split"}.issubset(sdf.columns):
                    infer_split_from_optional_csv._mapping = dict(
                        zip(
                            sdf["filename"].astype(str),
                            sdf["split"].astype(str),
                        )
                    )
                    print(f"Using split metadata from: {split_csv}")
            except Exception as exc:
                print(f"Warning: could not read split CSV: {exc}")

    return infer_split_from_optional_csv._mapping.get(
        filename,
        "historical_split_unknown",
    )


# ============================================================
# DISCOVER FILES AND LABEL MAPPING
# ============================================================

all_img = sorted(
    f
    for f in os.listdir(IMG_DIR)
    if f.endswith(".tif") and not f.endswith(".aux.xml")
)

all_msk = sorted(
    f
    for f in os.listdir(MASK_DIR)
    if f.endswith(".tif") and not f.endswith(".aux.xml")
)

common_files = sorted(set(all_img).intersection(all_msk))

missing_masks = sorted(set(all_img) - set(all_msk))
missing_images = sorted(set(all_msk) - set(all_img))

if not common_files:
    raise FileNotFoundError(
        "No matching .tif filenames were found between image and mask folders."
    )

print("\n=========================================")
print("            FILE DISCOVERY")
print("=========================================")
print(f"Matching image-mask pairs : {len(common_files)}")
print(f"Images without masks      : {len(missing_masks)}")
print(f"Masks without images      : {len(missing_images)}")
print("=========================================\n")

# Discover labels exactly as in training.
unique_labels = set()

for filename in all_msk:
    with rasterio.open(os.path.join(MASK_DIR, filename)) as src:
        band = src.read(1)
        unique_labels.update(int(v) for v in np.unique(band))

LABEL_LIST = sorted(unique_labels)
N_CLASSES = len(LABEL_LIST)

LABEL_MAP = {
    raw_label: model_idx
    for model_idx, raw_label in enumerate(LABEL_LIST)
}

INDEX_TO_LABEL = {
    model_idx: raw_label
    for raw_label, model_idx in LABEL_MAP.items()
}

if WASTE_RAW_LABEL not in LABEL_MAP:
    raise ValueError(
        f"Waste raw raster label {WASTE_RAW_LABEL} was not found. "
        f"Detected labels: {LABEL_LIST}"
    )

WASTE_CLASS_IDX = LABEL_MAP[WASTE_RAW_LABEL]

print("=========================================")
print("           CLASS LABEL MAPPING")
print("=========================================")
print(f"Detected raw labels       : {LABEL_LIST}")
print(f"Expected raw labels       : {EXPECTED_RAW_LABELS}")
print(f"Raw -> model mapping      : {LABEL_MAP}")
print(f"Waste raw label           : {WASTE_RAW_LABEL}")
print(f"Waste model channel       : {WASTE_CLASS_IDX}")

if LABEL_LIST != EXPECTED_RAW_LABELS:
    print(
        "WARNING: Detected labels do not exactly match expected labels 1-10."
    )

print("=========================================\n")


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Saved model was not found:\n{MODEL_PATH}"
    )

print(f"Loading model:\n{MODEL_PATH}")

# compile=False avoids requiring the custom training metrics for inference.
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

model_output_channels = int(model.output_shape[-1])

if model_output_channels != N_CLASSES:
    raise ValueError(
        "Model/data class mismatch.\n"
        f"Model output channels: {model_output_channels}\n"
        f"Classes currently discovered in masks: {N_CLASSES}\n"
        f"Detected labels: {LABEL_LIST}\n"
        "This may indicate that the masks have changed since training."
    )

waste_threshold, threshold_source = load_probability_threshold()

print("\n=========================================")
print("           INFERENCE SETTINGS")
print("=========================================")
print(f"Waste probability threshold : {waste_threshold:.4f}")
print(f"Threshold source             : {threshold_source}")
print(f"Model output classes         : {model_output_channels}")
print("=========================================\n")


# ============================================================
# STORAGE
# ============================================================

pile_rows = []
missing_label_rows = []
image_audit_rows = []


# ============================================================
# RUN DATA AUDIT + INFERENCE
# ============================================================

for file_index, filename in enumerate(common_files, start=1):

    img_path = os.path.join(IMG_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename)

    print(
        f"[{file_index:>4}/{len(common_files)}] "
        f"Auditing {filename}"
    )

    # --------------------------------------------------------
    # Read original image and mask
    # --------------------------------------------------------
    with rasterio.open(img_path) as src:
        image_band_count = src.count
        image_dtype = str(src.dtypes[0])

        if src.count < 3:
            raise ValueError(
                f"{filename}: image contains only {src.count} band(s); "
                "the model expects at least 3."
            )

        img = (
            src.read([1, 2, 3])
            .transpose(1, 2, 0)
            .astype("float32")
        )

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        mask_dtype = str(src.dtypes[0])

    original_img_shape = img.shape
    original_mask_shape = mask.shape

    finite_pixels = img[np.isfinite(img)]

    if finite_pixels.size > 0:
        image_min = float(np.min(finite_pixels))
        image_max = float(np.max(finite_pixels))
        image_mean = float(np.mean(finite_pixels))
    else:
        image_min = np.nan
        image_max = np.nan
        image_mean = np.nan

    image_nan_count = int(np.sum(~np.isfinite(img)))

    file_labels = sorted(int(v) for v in np.unique(mask))

    audit_flags = []

    if original_img_shape[:2] != original_mask_shape:
        audit_flags.append("IMAGE_MASK_SHAPE_MISMATCH")

    unexpected_file_labels = sorted(
        set(file_labels) - set(EXPECTED_RAW_LABELS)
    )

    missing_expected_labels = sorted(
        set(EXPECTED_RAW_LABELS) - set(file_labels)
    )

    if unexpected_file_labels:
        audit_flags.append(
            "UNEXPECTED_MASK_LABELS:"
            + ",".join(map(str, unexpected_file_labels))
        )

    if image_nan_count > 0:
        audit_flags.append(f"NONFINITE_IMAGE_VALUES:{image_nan_count}")

    # The training code always divides imagery by 255.
    # These conditions can reveal scaling problems in the source imagery.
    if np.isfinite(image_max):
        if image_max <= 1.5:
            audit_flags.append(
                "IMAGE_APPEARS_ALREADY_NORMALIZED_BUT_TRAINING_DIVIDES_BY_255"
            )
        elif image_max > 255.0:
            audit_flags.append(
                "IMAGE_MAX_GT_255_BUT_TRAINING_DIVIDES_BY_255"
            )

    # --------------------------------------------------------
    # Apply training preprocessing
    # --------------------------------------------------------
    img_model, mask_model = resize_image_and_mask(img, mask)

    resized_labels = sorted(int(v) for v in np.unique(mask_model))

    post_resize_unexpected = sorted(
        set(resized_labels) - set(LABEL_LIST)
    )

    if post_resize_unexpected:
        audit_flags.append(
            "RESIZE_CREATED_UNKNOWN_LABELS:"
            + ",".join(map(str, post_resize_unexpected))
        )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------
    pred_probs = model.predict(
        np.expand_dims(img_model, axis=0),
        verbose=0,
    )[0]

    waste_prob = pred_probs[..., WASTE_CLASS_IDX]
    pred_model_idx = np.argmax(pred_probs, axis=-1)

    # Convert model channels back to original raster labels.
    pred_raw = np.zeros_like(pred_model_idx, dtype=np.int32)

    for idx, raw_label in INDEX_TO_LABEL.items():
        pred_raw[pred_model_idx == idx] = raw_label

    gt_waste = mask_model == WASTE_RAW_LABEL

    # --------------------------------------------------------
    # Find labelled waste piles
    # --------------------------------------------------------
    pile_labels, n_piles = connected_components(
        gt_waste.astype(np.uint8),
        structure=CONNECTIVITY_STRUCTURE,
    )

    pile_sizes = []

    for pile_id in range(1, n_piles + 1):

        pile_mask = pile_labels == pile_id
        n_pixels = int(np.sum(pile_mask))

        if n_pixels < MIN_GT_PILE_PIXELS:
            continue

        pile_sizes.append(n_pixels)

        bbox = bbox_from_binary(pile_mask)
        r0, c0, r1, c1 = bbox

        rows, cols = np.where(pile_mask)
        centroid_row = float(np.mean(rows))
        centroid_col = float(np.mean(cols))

        pile_probs = pred_probs[pile_mask]
        pile_waste_prob = waste_prob[pile_mask]
        pile_pred_idx = pred_model_idx[pile_mask]
        pile_pred_raw = pred_raw[pile_mask]

        # ----------------------------------------------------
        # Waste support metrics
        # ----------------------------------------------------
        mean_waste_prob = float(np.mean(pile_waste_prob))
        median_waste_prob = float(np.median(pile_waste_prob))
        p10_waste_prob = float(np.percentile(pile_waste_prob, 10))
        p90_waste_prob = float(np.percentile(pile_waste_prob, 90))
        min_waste_prob = float(np.min(pile_waste_prob))
        max_waste_prob = float(np.max(pile_waste_prob))

        pile_recall_threshold = float(
            np.mean(pile_waste_prob >= waste_threshold)
        )

        pile_recall_argmax = float(
            np.mean(pile_pred_idx == WASTE_CLASS_IDX)
        )

        # ----------------------------------------------------
        # Competing class
        # ----------------------------------------------------
        other_indices = [
            idx for idx in range(N_CLASSES)
            if idx != WASTE_CLASS_IDX
        ]

        other_probs = pile_probs[:, other_indices]

        top_other_local_idx = np.argmax(other_probs, axis=-1)
        top_other_model_idx = np.array(
            [other_indices[i] for i in top_other_local_idx],
            dtype=np.int32,
        )

        top_other_prob = np.max(other_probs, axis=-1)

        mean_top_competing_prob = float(np.mean(top_other_prob))

        mean_margin = float(
            np.mean(pile_waste_prob - top_other_prob)
        )

        # Dominant competing class based on strongest non-waste class
        # at each ground-truth waste pixel.
        dominant_competing_idx, competing_count = get_dominant_value(
            top_other_model_idx
        )

        dominant_competing_raw = (
            INDEX_TO_LABEL[dominant_competing_idx]
            if dominant_competing_idx is not None
            else None
        )

        dominant_competing_fraction = (
            float(competing_count / n_pixels)
            if n_pixels > 0
            else np.nan
        )

        # Dominant actual argmax mistake among pixels the model called non-waste.
        wrong_pixels = pile_pred_idx != WASTE_CLASS_IDX

        if np.any(wrong_pixels):
            dominant_wrong_idx, wrong_count = get_dominant_value(
                pile_pred_idx[wrong_pixels]
            )
            dominant_wrong_raw = INDEX_TO_LABEL[dominant_wrong_idx]

            dominant_wrong_fraction_of_pile = float(
                wrong_count / n_pixels
            )

            dominant_wrong_fraction_of_errors = float(
                wrong_count / np.sum(wrong_pixels)
            )
        else:
            dominant_wrong_raw = None
            dominant_wrong_fraction_of_pile = 0.0
            dominant_wrong_fraction_of_errors = 0.0

        # ----------------------------------------------------
        # Uncertainty
        # ----------------------------------------------------
        pile_entropy = normalized_entropy(pile_probs)
        mean_entropy = float(np.mean(pile_entropy))

        # ----------------------------------------------------
        # Review priority
        #
        # This is deliberately called a REVIEW score, not an error
        # probability. High score = inspect this labelled pile first.
        # ----------------------------------------------------
        review_priority = (
            0.45 * (1.0 - pile_recall_threshold)
            + 0.35 * (1.0 - mean_waste_prob)
            + 0.20 * mean_top_competing_prob
        )

        review_priority = float(
            np.clip(review_priority, 0.0, 1.0)
        )

        review_flags = []

        if mean_waste_prob < LOW_MEAN_WASTE_PROB:
            review_flags.append("VERY_LOW_MODEL_SUPPORT")

        if pile_recall_threshold < LOW_PILE_RECALL:
            review_flags.append("MOSTLY_NOT_DETECTED")

        if mean_margin < STRONG_NEGATIVE_MARGIN:
            review_flags.append("STRONG_COMPETING_CLASS")

        if mean_entropy > HIGH_NORMALIZED_ENTROPY:
            review_flags.append("HIGH_UNCERTAINTY")

        if n_pixels < TINY_PILE_PIXELS:
            review_flags.append("TINY_COMPONENT")

        if touches_edge(pile_mask):
            review_flags.append("TOUCHES_TILE_EDGE")

        pile_rows.append(
            {
                "Filename": filename,
                "Historical_Split": infer_split_from_optional_csv(filename),
                "Pile_ID": pile_id,
                "Pile_Pixels_256": n_pixels,

                "Centroid_Row_256": round(centroid_row, 2),
                "Centroid_Col_256": round(centroid_col, 2),

                "BBox_Row_Min": r0,
                "BBox_Col_Min": c0,
                "BBox_Row_Max_Exclusive": r1,
                "BBox_Col_Max_Exclusive": c1,

                "Touches_Tile_Edge": touches_edge(pile_mask),

                "Mean_Waste_Probability": mean_waste_prob,
                "Median_Waste_Probability": median_waste_prob,
                "P10_Waste_Probability": p10_waste_prob,
                "P90_Waste_Probability": p90_waste_prob,
                "Min_Waste_Probability": min_waste_prob,
                "Max_Waste_Probability": max_waste_prob,

                "Pile_Recall_At_Threshold": pile_recall_threshold,
                "Pile_Recall_Argmax": pile_recall_argmax,

                "Mean_Top_Competing_Probability": mean_top_competing_prob,
                "Mean_Waste_Minus_Competitor_Margin": mean_margin,
                "Mean_Normalized_Entropy": mean_entropy,

                "Dominant_Competing_Raw_Label": dominant_competing_raw,
                "Dominant_Competing_Class": (
                    class_name(dominant_competing_raw)
                    if dominant_competing_raw is not None
                    else ""
                ),
                "Dominant_Competing_Fraction": dominant_competing_fraction,

                "Dominant_Argmax_Wrong_Raw_Label": dominant_wrong_raw,
                "Dominant_Argmax_Wrong_Class": (
                    class_name(dominant_wrong_raw)
                    if dominant_wrong_raw is not None
                    else ""
                ),
                "Dominant_Wrong_Fraction_Of_Pile": (
                    dominant_wrong_fraction_of_pile
                ),
                "Dominant_Wrong_Fraction_Of_Error_Pixels": (
                    dominant_wrong_fraction_of_errors
                ),

                "Review_Priority_Score": review_priority,
                "Review_Flags": ";".join(review_flags),
            }
        )

    # --------------------------------------------------------
    # Search for potential missing waste labels
    #
    # A high-confidence predicted-waste connected component that
    # lies outside the labelled waste class can reveal:
    #   - a true false positive, OR
    #   - waste that was omitted from the ground truth.
    #
    # It must be manually reviewed.
    # --------------------------------------------------------
    possible_missing_binary = (
        (waste_prob >= MISSING_LABEL_PROB_THRESHOLD)
        & (~gt_waste)
    )

    missing_cc, n_missing = connected_components(
        possible_missing_binary.astype(np.uint8),
        structure=CONNECTIVITY_STRUCTURE,
    )

    for candidate_id in range(1, n_missing + 1):

        candidate_mask = missing_cc == candidate_id
        candidate_pixels = int(np.sum(candidate_mask))

        if candidate_pixels < MIN_MISSING_COMPONENT_PIXELS:
            continue

        bbox = bbox_from_binary(candidate_mask)
        r0, c0, r1, c1 = bbox

        rows, cols = np.where(candidate_mask)

        candidate_waste_prob = waste_prob[candidate_mask]
        candidate_gt_labels = mask_model[candidate_mask]
        candidate_pred_idx = pred_model_idx[candidate_mask]

        dominant_gt_raw, dominant_gt_count = get_dominant_value(
            candidate_gt_labels
        )

        mean_prob = float(np.mean(candidate_waste_prob))
        max_prob = float(np.max(candidate_waste_prob))
        min_prob = float(np.min(candidate_waste_prob))

        # Rank larger, highly confident candidates higher.
        missing_review_score = float(
            mean_prob * np.log1p(candidate_pixels)
        )

        missing_label_rows.append(
            {
                "Filename": filename,
                "Historical_Split": infer_split_from_optional_csv(filename),
                "Candidate_ID": candidate_id,
                "Pixels_256": candidate_pixels,

                "Centroid_Row_256": round(float(np.mean(rows)), 2),
                "Centroid_Col_256": round(float(np.mean(cols)), 2),

                "BBox_Row_Min": r0,
                "BBox_Col_Min": c0,
                "BBox_Row_Max_Exclusive": r1,
                "BBox_Col_Max_Exclusive": c1,

                "Mean_Waste_Probability": mean_prob,
                "Min_Waste_Probability": min_prob,
                "Max_Waste_Probability": max_prob,

                "Dominant_Ground_Truth_Raw_Label": dominant_gt_raw,
                "Dominant_Ground_Truth_Class": class_name(dominant_gt_raw),
                "Dominant_GT_Fraction": float(
                    dominant_gt_count / candidate_pixels
                ),

                "Argmax_Waste_Fraction": float(
                    np.mean(candidate_pred_idx == WASTE_CLASS_IDX)
                ),

                "Touches_Tile_Edge": touches_edge(candidate_mask),

                "Missing_Label_Review_Score": missing_review_score,
            }
        )

    # --------------------------------------------------------
    # Image-level audit row
    # --------------------------------------------------------
    image_audit_rows.append(
        {
            "Filename": filename,
            "Historical_Split": infer_split_from_optional_csv(filename),

            "Image_Original_Height": original_img_shape[0],
            "Image_Original_Width": original_img_shape[1],
            "Image_Bands_Read": 3,
            "Image_Total_Bands": image_band_count,
            "Image_Dtype": image_dtype,

            "Mask_Original_Height": original_mask_shape[0],
            "Mask_Original_Width": original_mask_shape[1],
            "Mask_Dtype": mask_dtype,

            "Image_Min": image_min,
            "Image_Max": image_max,
            "Image_Mean": image_mean,
            "Nonfinite_Image_Value_Count": image_nan_count,

            "Mask_Labels_Original": ",".join(map(str, file_labels)),
            "Mask_Labels_After_Resize": ",".join(map(str, resized_labels)),
            "Expected_Labels_Missing_From_This_Tile": ",".join(
                map(str, missing_expected_labels)
            ),

            "Waste_Pixels_Original": int(
                np.sum(mask == WASTE_RAW_LABEL)
            ),
            "Waste_Pixels_256": int(np.sum(gt_waste)),
            "Ground_Truth_Waste_Components_256": int(n_piles),

            "Mean_Waste_Probability_On_GT_Waste": (
                float(np.mean(waste_prob[gt_waste]))
                if np.any(gt_waste)
                else np.nan
            ),

            "Mean_Waste_Probability_On_NonWaste": (
                float(np.mean(waste_prob[~gt_waste]))
                if np.any(~gt_waste)
                else np.nan
            ),

            "High_Confidence_Unlabelled_Waste_Pixels": int(
                np.sum(possible_missing_binary)
            ),

            "Audit_Flags": ";".join(audit_flags),
        }
    )


# ============================================================
# BUILD TABLES
# ============================================================

pile_df = pd.DataFrame(pile_rows)
missing_df = pd.DataFrame(missing_label_rows)
image_df = pd.DataFrame(image_audit_rows)

if not pile_df.empty:
    pile_df = pile_df.sort_values(
        ["Review_Priority_Score", "Mean_Waste_Probability"],
        ascending=[False, True],
    ).reset_index(drop=True)

    pile_df.insert(
        0,
        "Review_Rank",
        np.arange(1, len(pile_df) + 1),
    )

if not missing_df.empty:
    missing_df = missing_df.sort_values(
        [
            "Missing_Label_Review_Score",
            "Mean_Waste_Probability",
            "Pixels_256",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    missing_df.insert(
        0,
        "Review_Rank",
        np.arange(1, len(missing_df) + 1),
    )

image_df = image_df.sort_values("Filename").reset_index(drop=True)


# ============================================================
# SAVE CSV TABLES
# ============================================================

pile_csv = os.path.join(
    AUDIT_DIR,
    "waste_pile_diagnostics.csv",
)

missing_csv = os.path.join(
    AUDIT_DIR,
    "possible_missing_waste_annotations.csv",
)

image_csv = os.path.join(
    AUDIT_DIR,
    "image_mask_data_audit.csv",
)

pile_df.to_csv(pile_csv, index=False)
missing_df.to_csv(missing_csv, index=False)
image_df.to_csv(image_csv, index=False)


# ============================================================
# CONFUSION SUMMARY
# ============================================================

confusion_csv = os.path.join(
    AUDIT_DIR,
    "waste_pile_confusion_summary.csv",
)

if not pile_df.empty:

    confusion_df = (
        pile_df.groupby(
            [
                "Dominant_Competing_Raw_Label",
                "Dominant_Competing_Class",
            ],
            dropna=False,
        )
        .agg(
            Number_Of_Piles=("Pile_ID", "count"),
            Mean_Pile_Recall=(
                "Pile_Recall_At_Threshold",
                "mean",
            ),
            Mean_Waste_Probability=(
                "Mean_Waste_Probability",
                "mean",
            ),
            Mean_Review_Priority=(
                "Review_Priority_Score",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "Number_Of_Piles",
            ascending=False,
        )
    )

    confusion_df.to_csv(confusion_csv, index=False)

else:
    confusion_df = pd.DataFrame()
    confusion_df.to_csv(confusion_csv, index=False)


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def load_processed_pair(filename):
    """Load, preprocess, and predict one pair for visualization."""
    with rasterio.open(os.path.join(IMG_DIR, filename)) as src:
        img = (
            src.read([1, 2, 3])
            .transpose(1, 2, 0)
            .astype("float32")
        )

    with rasterio.open(os.path.join(MASK_DIR, filename)) as src:
        mask = src.read(1)

    img_model, mask_model = resize_image_and_mask(img, mask)

    probs = model.predict(
        np.expand_dims(img_model, axis=0),
        verbose=0,
    )[0]

    pred_idx = np.argmax(probs, axis=-1)

    pred_raw = np.zeros_like(pred_idx, dtype=np.int32)

    for idx, raw_label in INDEX_TO_LABEL.items():
        pred_raw[pred_idx == idx] = raw_label

    return img_model, mask_model, probs, pred_raw


def add_bbox_to_axis(ax, bbox, crop_bbox):
    """Draw object bbox in crop coordinates."""
    r0, c0, r1, c1 = bbox
    cr0, cc0, _, _ = crop_bbox

    x = c0 - cc0
    y = r0 - cr0
    width = c1 - c0
    height = r1 - r0

    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            fill=False,
            linewidth=2,
        )
    )


def save_pile_visual(row):
    """Save 4-panel diagnostic for one labelled pile."""

    filename = row["Filename"]
    pile_id = int(row["Pile_ID"])

    img_model, mask_model, probs, pred_raw = load_processed_pair(
        filename
    )

    gt_waste = mask_model == WASTE_RAW_LABEL

    pile_labels, _ = connected_components(
        gt_waste.astype(np.uint8),
        structure=CONNECTIVITY_STRUCTURE,
    )

    pile_mask = pile_labels == pile_id

    if not np.any(pile_mask):
        return

    bbox = bbox_from_binary(pile_mask)

    crop_bbox = padded_bbox(
        bbox,
        IMG_HEIGHT,
        IMG_WIDTH,
    )

    r0, c0, r1, c1 = crop_bbox

    img_crop = img_model[r0:r1, c0:c1]
    pile_crop = pile_mask[r0:r1, c0:c1]
    waste_prob_crop = probs[
        r0:r1,
        c0:c1,
        WASTE_CLASS_IDX,
    ]
    pred_raw_crop = pred_raw[r0:r1, c0:c1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(np.clip(img_crop, 0, 1))
    add_bbox_to_axis(axes[0], bbox, crop_bbox)
    axes[0].set_title("RGB image")

    axes[1].imshow(pile_crop)
    axes[1].set_title(
        f"Ground-truth waste pile {pile_id}"
    )

    im2 = axes[2].imshow(
        waste_prob_crop,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        "Waste probability\n"
        f"mean={row['Mean_Waste_Probability']:.3f}"
    )
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(
        pred_raw_crop,
        vmin=min(LABEL_LIST),
        vmax=max(LABEL_LIST),
    )
    axes[3].set_title(
        "Argmax predicted raw class\n"
        f"main competitor={row['Dominant_Competing_Class']}"
    )
    fig.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Review rank {int(row['Review_Rank'])} | "
        f"{filename} | pile {pile_id} | "
        f"recall={row['Pile_Recall_At_Threshold']:.3f} | "
        f"review score={row['Review_Priority_Score']:.3f}"
    )

    plt.tight_layout()

    out_name = (
        f"rank_{int(row['Review_Rank']):03d}_"
        f"{safe_name(Path(filename).stem)}_"
        f"pile_{pile_id:03d}.png"
    )

    plt.savefig(
        os.path.join(DIFFICULT_PILE_DIR, out_name),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_missing_candidate_visual(row):
    """Save 4-panel diagnostic for a possible missing waste label."""

    filename = row["Filename"]
    candidate_id = int(row["Candidate_ID"])

    img_model, mask_model, probs, pred_raw = load_processed_pair(
        filename
    )

    waste_prob = probs[..., WASTE_CLASS_IDX]
    gt_waste = mask_model == WASTE_RAW_LABEL

    binary = (
        (waste_prob >= MISSING_LABEL_PROB_THRESHOLD)
        & (~gt_waste)
    )

    cc, _ = connected_components(
        binary.astype(np.uint8),
        structure=CONNECTIVITY_STRUCTURE,
    )

    candidate_mask = cc == candidate_id

    if not np.any(candidate_mask):
        return

    bbox = bbox_from_binary(candidate_mask)

    crop_bbox = padded_bbox(
        bbox,
        IMG_HEIGHT,
        IMG_WIDTH,
    )

    r0, c0, r1, c1 = crop_bbox

    img_crop = img_model[r0:r1, c0:c1]
    mask_crop = mask_model[r0:r1, c0:c1]
    candidate_crop = candidate_mask[r0:r1, c0:c1]
    waste_prob_crop = waste_prob[r0:r1, c0:c1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(np.clip(img_crop, 0, 1))
    add_bbox_to_axis(axes[0], bbox, crop_bbox)
    axes[0].set_title("RGB image")

    im1 = axes[1].imshow(
        mask_crop,
        vmin=min(LABEL_LIST),
        vmax=max(LABEL_LIST),
    )
    axes[1].set_title(
        "Current ground-truth raw class"
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].imshow(candidate_crop)
    axes[2].set_title(
        "High-confidence predicted waste\n"
        "outside waste annotation"
    )

    im3 = axes[3].imshow(
        waste_prob_crop,
        vmin=0,
        vmax=1,
    )
    axes[3].set_title(
        f"Waste probability\n"
        f"mean={row['Mean_Waste_Probability']:.3f}"
    )
    fig.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Possible missing annotation rank "
        f"{int(row['Review_Rank'])} | "
        f"{filename} | candidate {candidate_id} | "
        f"current label={row['Dominant_Ground_Truth_Class']}"
    )

    plt.tight_layout()

    out_name = (
        f"rank_{int(row['Review_Rank']):03d}_"
        f"{safe_name(Path(filename).stem)}_"
        f"candidate_{candidate_id:03d}.png"
    )

    plt.savefig(
        os.path.join(MISSING_LABEL_DIR, out_name),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# SAVE TOP DIAGNOSTIC IMAGES
# ============================================================

if not pile_df.empty:

    n = min(
        TOP_N_DIFFICULT_PILES,
        len(pile_df),
    )

    print(
        f"\nSaving visual diagnostics for "
        f"{n} highest-priority labelled waste piles..."
    )

    for _, row in pile_df.head(n).iterrows():
        save_pile_visual(row)


if not missing_df.empty:

    n = min(
        TOP_N_MISSING_LABEL_CANDIDATES,
        len(missing_df),
    )

    print(
        f"Saving visual diagnostics for "
        f"{n} possible missing waste annotations..."
    )

    for _, row in missing_df.head(n).iterrows():
        save_missing_candidate_visual(row)


# ============================================================
# SUMMARY REPORT
# ============================================================

summary_path = os.path.join(
    AUDIT_DIR,
    "audit_summary.txt",
)

n_flagged_images = int(
    np.sum(image_df["Audit_Flags"].fillna("").astype(str) != "")
)

n_piles = len(pile_df)

if not pile_df.empty:
    n_very_low_support = int(
        np.sum(
            pile_df["Mean_Waste_Probability"]
            < LOW_MEAN_WASTE_PROB
        )
    )

    n_low_recall = int(
        np.sum(
            pile_df["Pile_Recall_At_Threshold"]
            < LOW_PILE_RECALL
        )
    )

    n_negative_margin = int(
        np.sum(
            pile_df["Mean_Waste_Minus_Competitor_Margin"]
            < STRONG_NEGATIVE_MARGIN
        )
    )
else:
    n_very_low_support = 0
    n_low_recall = 0
    n_negative_margin = 0

with open(summary_path, "w", encoding="utf-8") as f:

    f.write("WASTE PILE DATA AUDIT SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Image directory: {IMG_DIR}\n")
    f.write(f"Mask directory: {MASK_DIR}\n\n")

    f.write(
        "Historical train/validation split:\n"
        "The original random split cannot be reconstructed because the "
        "original script did not save the split and did not fix a random "
        "seed. This audit therefore evaluates all matching pairs, which "
        "covers both the original training and validation data.\n\n"
    )

    f.write(f"Matching image-mask pairs: {len(common_files)}\n")
    f.write(f"Images without masks: {len(missing_masks)}\n")
    f.write(f"Masks without images: {len(missing_images)}\n")
    f.write(f"Files with basic audit flags: {n_flagged_images}\n\n")

    f.write(f"Detected raw labels: {LABEL_LIST}\n")
    f.write(f"Expected raw labels: {EXPECTED_RAW_LABELS}\n")
    f.write(f"Waste raw label: {WASTE_RAW_LABEL}\n")
    f.write(f"Waste model channel: {WASTE_CLASS_IDX}\n\n")

    f.write(f"Waste threshold: {waste_threshold:.4f}\n")
    f.write(f"Threshold source: {threshold_source}\n\n")

    f.write(f"Ground-truth waste piles/components: {n_piles}\n")
    f.write(
        f"Piles with mean P(waste) < {LOW_MEAN_WASTE_PROB:.2f}: "
        f"{n_very_low_support}\n"
    )
    f.write(
        f"Piles with recall < {LOW_PILE_RECALL:.2f}: "
        f"{n_low_recall}\n"
    )
    f.write(
        f"Piles with waste-minus-competitor margin "
        f"< {STRONG_NEGATIVE_MARGIN:.2f}: "
        f"{n_negative_margin}\n"
    )

    f.write(
        f"\nPossible missing waste annotation candidates "
        f"(P(waste) >= {MISSING_LABEL_PROB_THRESHOLD:.2f}, "
        f"component >= {MIN_MISSING_COMPONENT_PIXELS} pixels): "
        f"{len(missing_df)}\n"
    )

    f.write("\nOUTPUTS\n")
    f.write("-" * 60 + "\n")
    f.write(f"{pile_csv}\n")
    f.write(f"{missing_csv}\n")
    f.write(f"{image_csv}\n")
    f.write(f"{confusion_csv}\n")
    f.write(f"{DIFFICULT_PILE_DIR}\n")
    f.write(f"{MISSING_LABEL_DIR}\n")


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n=========================================")
print("          DATA AUDIT COMPLETE")
print("=========================================")
print(f"Image-mask pairs audited      : {len(common_files)}")
print(f"Ground-truth waste piles      : {len(pile_df)}")
print(f"Possible missing annotations  : {len(missing_df)}")
print(f"Images with basic audit flags : {n_flagged_images}")

print("\nSaved:")
print(f" - Pile diagnostics:       {pile_csv}")
print(f" - Possible missing waste: {missing_csv}")
print(f" - Image/mask audit:       {image_csv}")
print(f" - Confusion summary:      {confusion_csv}")
print(f" - Summary:                {summary_path}")
print(f" - Difficult pile images:  {DIFFICULT_PILE_DIR}")
print(f" - Missing-label images:   {MISSING_LABEL_DIR}")
print("=========================================\n")

if not pile_df.empty:
    print("Top 10 labelled waste piles to review:")
    cols = [
        "Review_Rank",
        "Filename",
        "Pile_ID",
        "Pile_Pixels_256",
        "Mean_Waste_Probability",
        "Pile_Recall_At_Threshold",
        "Mean_Waste_Minus_Competitor_Margin",
        "Dominant_Competing_Class",
        "Review_Flags",
    ]
    print(
        pile_df[cols]
        .head(10)
        .to_string(index=False)
    )

if not missing_df.empty:
    print("\nTop 10 possible missing waste annotations:")
    cols = [
        "Review_Rank",
        "Filename",
        "Candidate_ID",
        "Pixels_256",
        "Mean_Waste_Probability",
        "Dominant_Ground_Truth_Class",
    ]
    print(
        missing_df[cols]
        .head(10)
        .to_string(index=False)
    )
