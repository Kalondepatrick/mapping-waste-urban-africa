# =============================================================================
# HARD-NEGATIVE FINE-TUNED U-NET:
# TEST-AREA PREDICTION + INDEPENDENT PIXEL-LEVEL EVALUATION
# =============================================================================
#
# OUTPUT 1
# --------
# Continuous waste-probability surface for the spatial test area.
#
# OUTPUT 2
# --------
# Multiclass predicted land-cover surface:
#
#   1  Waste
#   2  Soil
#   3  Vegetation
#   4  Built-up surface
#   5  Built-up surface (paved area)
#   6  Water Body
#   7  Natural Surface - rocks
#   8  Non-persistent object
#   9  Ancillary class
#   10 Unidentified
#
# OUTPUT 3
# --------
# Pixel-level accuracy on the withheld Validation image-mask dataset:
#
#   TP
#   FP
#   FN
#   TN
#   Accuracy
#   Precision
#   Recall
#   F1
#   IoU
#
#
# IMPORTANT
# ---------
# The test data are NOT used for:
#
#   - fine-tuning
#   - threshold optimisation
#   - model selection
#
# They are used only for final evaluation.
#
#
# PRIMARY EVALUATION
# ------------------
# Multiclass ARGMAX:
#
#     predicted channel 0 = waste
#
# This evaluates the actual multiclass semantic-segmentation decision.
#
#
# SECONDARY EVALUATION
# --------------------
# Waste probability >= 0.50
#
# This is retained only to reproduce the previous binary evaluation convention.
#
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
import math

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio

from rasterio.windows import Window
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask

from shapely.geometry import mapping
from shapely.ops import unary_union

import tensorflow as tf


# =============================================================================
# 2. PATHS
# =============================================================================

BASE_PATH = (
    Path.home()
    / "Documents"
    / "GitHub"
    / "mapping-waste-urban-africa"
)


# -----------------------------------------------------------------------------
# Full orthomosaic
# -----------------------------------------------------------------------------

ORTHOMOSAIC_PATH = (
    BASE_PATH
    / "data"
    / "optical.tiff"
)


# -----------------------------------------------------------------------------
# Spatial test-area polygon
# -----------------------------------------------------------------------------

TEST_AREA_PATH = (
    BASE_PATH
    / "data"
    / "aoi"
    / "testarea.shp"
)


# -----------------------------------------------------------------------------
# Fine-tuned model
# -----------------------------------------------------------------------------

MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_hardnegative_finetuned.keras"
)


# -----------------------------------------------------------------------------
# Original withheld validation / test tiles
#
# These are used for accuracy assessment.
# -----------------------------------------------------------------------------

VALIDATION_DIR = (
    BASE_PATH
    / "data"
    / "tiled_data"
    / "partitioned"
    / "Validation"
)


VALIDATION_IMAGE_DIR = (
    VALIDATION_DIR
    / "images"
)


VALIDATION_MASK_DIR = (
    VALIDATION_DIR
    / "masks"
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

OUTPUT_DIR = (
    BASE_PATH
    / "results"
    / "test_area_finetuned"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CLASS_RASTER_PATH = (
    OUTPUT_DIR
    / "testarea_predicted_landcover1.tif"
)


WASTE_PROBABILITY_PATH = (
    OUTPUT_DIR
    / "testarea_waste_probability11.tif"
)


CLASS_LOOKUP_PATH = (
    OUTPUT_DIR
    / "class_lookup.csv"
)


ACCURACY_CSV_PATH = (
    OUTPUT_DIR
    / "testarea_accuracy_metrics.csv"
)


ACCURACY_TXT_PATH = (
    OUTPUT_DIR
    / "testarea_accuracy_metrics.txt"
)


PER_TILE_CSV_PATH = (
    OUTPUT_DIR
    / "testarea_accuracy_by_tile.csv"
)


MASK_AUDIT_PATH = (
    OUTPUT_DIR
    / "test_reference_mask_audit.csv"
)


# =============================================================================
# 3. MODEL CONFIGURATION
# =============================================================================

TILE_SIZE = 256


# Spatial surface inference:
#
# predict 256 x 256 but retain only the central 128 x 128.
CORE_SIZE = 128


HALO = (
    TILE_SIZE
    - CORE_SIZE
) // 2


RGB_BANDS = [
    1,
    2,
    3
]


N_CLASSES = 10


WASTE_RAW_LABEL = 1


# Raw class 1 corresponds to softmax channel 0.
WASTE_CLASS_INDEX = 0


BATCH_SIZE = 8


# =============================================================================
# 4. SECONDARY FIXED THRESHOLD
# =============================================================================

# This is NOT selected using the test data.
#
# It is provided only to reproduce the old binary evaluation convention.

FIXED_WASTE_THRESHOLD = 0.50


# =============================================================================
# 5. OUTPUT NODATA
# =============================================================================

CLASS_NODATA = 0


PROBABILITY_NODATA = -9999.0


# =============================================================================
# 6. RAW CLASS DEFINITIONS
# =============================================================================

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


VALID_RAW_LABELS = set(
    range(
        1,
        11
    )
)


# =============================================================================
# 7. CHECK REQUIRED INPUTS
# =============================================================================

print("\n")
print("=" * 80)
print("CHECKING INPUTS")
print("=" * 80)


required_paths = [

    ORTHOMOSAIC_PATH,
    TEST_AREA_PATH,
    MODEL_PATH,
    VALIDATION_IMAGE_DIR,
    VALIDATION_MASK_DIR

]


for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired input not found:\n"
            f"{path}"
        )


print(
    f"\nFine-tuned model:\n"
    f"{MODEL_PATH}"
)


print(
    f"\nOrthomosaic:\n"
    f"{ORTHOMOSAIC_PATH}"
)


print(
    f"\nSpatial test area:\n"
    f"{TEST_AREA_PATH}"
)


print(
    f"\nValidation images:\n"
    f"{VALIDATION_IMAGE_DIR}"
)


print(
    f"\nValidation masks:\n"
    f"{VALIDATION_MASK_DIR}"
)


# =============================================================================
# 8. REMOVE PREVIOUS OUTPUT RASTERS
# =============================================================================

for output_path in [

    CLASS_RASTER_PATH,
    WASTE_PROBABILITY_PATH

]:

    if output_path.exists():

        try:

            output_path.unlink()

        except PermissionError as exc:

            raise PermissionError(
                f"\nCannot overwrite:\n"
                f"{output_path}\n\n"
                "Close the file in QGIS/ArcGIS and rerun."
            ) from exc


# =============================================================================
# 9. SAVE CLASS LOOKUP
# =============================================================================

pd.DataFrame(

    [

        {

            "raw_label":
                raw_label,

            "model_channel":
                raw_label - 1,

            "class_name":
                class_name

        }

        for raw_label, class_name
        in CLASS_NAMES.items()

    ]

).to_csv(
    CLASS_LOOKUP_PATH,
    index=False
)


# =============================================================================
# 10. LOAD FINE-TUNED MODEL
# =============================================================================

print("\n")
print("=" * 80)
print("LOADING FINE-TUNED MODEL")
print("=" * 80)


tf.keras.backend.clear_session()


model = tf.keras.models.load_model(

    MODEL_PATH,

    compile=False

)


print(
    f"Input shape : "
    f"{model.input_shape}"
)


print(
    f"Output shape: "
    f"{model.output_shape}"
)


if tuple(
    model.input_shape[
        1:
    ]
) != (
    256,
    256,
    3
):

    raise ValueError(
        "\nUnexpected model input shape:\n"
        f"{model.input_shape}"
    )


if int(
    model.output_shape[
        -1
    ]
) != N_CLASSES:

    raise ValueError(
        "\nExpected 10 model output classes, "
        f"but found {model.output_shape[-1]}."
    )


# =============================================================================
# PART A
# CREATE THE SPATIAL TEST-AREA SURFACES
# =============================================================================


# =============================================================================
# 11. READ TEST-AREA POLYGON
# =============================================================================

print("\n")
print("=" * 80)
print("PART A: SPATIAL TEST-AREA PREDICTION")
print("=" * 80)


test_area = gpd.read_file(
    TEST_AREA_PATH
)


test_area = test_area[
    test_area.geometry.notna()
    & (~test_area.geometry.is_empty)
].copy()


if test_area.empty:

    raise ValueError(
        "No usable test-area geometry found."
    )


# =============================================================================
# 12. CALCULATE RASTER-ALIGNED TEST WINDOW
# =============================================================================

with rasterio.open(
    ORTHOMOSAIC_PATH
) as src:


    if src.count < 3:

        raise ValueError(
            "Orthomosaic contains fewer than three bands."
        )


    if src.crs is None:

        raise ValueError(
            "Orthomosaic has no CRS."
        )


    # -------------------------------------------------------------------------
    # Project test polygon to image CRS
    # -------------------------------------------------------------------------

    test_area = test_area.to_crs(
        src.crs
    )


    # Geometry repair
    test_area[
        "geometry"
    ] = test_area.geometry.buffer(
        0
    )


    test_area = test_area[
        test_area.geometry.notna()
        & (~test_area.geometry.is_empty)
    ].copy()


    if test_area.empty:

        raise ValueError(
            "Test-area geometry became empty after geometry repair."
        )


    test_geometry = unary_union(
        test_area.geometry.tolist()
    )


    minx, miny, maxx, maxy = (
        test_geometry.bounds
    )


    raw_window = from_bounds(

        minx,
        miny,
        maxx,
        maxy,

        transform=src.transform

    )


    raw_row_start = int(
        math.floor(
            raw_window.row_off
        )
    )


    raw_col_start = int(
        math.floor(
            raw_window.col_off
        )
    )


    raw_row_stop = int(
        math.ceil(
            raw_window.row_off
            + raw_window.height
        )
    )


    raw_col_stop = int(
        math.ceil(
            raw_window.col_off
            + raw_window.width
        )
    )


    # -------------------------------------------------------------------------
    # Align to the GLOBAL 128-pixel prediction grid.
    # -------------------------------------------------------------------------

    aligned_row_start = (

        raw_row_start
        // CORE_SIZE

    ) * CORE_SIZE


    aligned_col_start = (

        raw_col_start
        // CORE_SIZE

    ) * CORE_SIZE


    aligned_row_stop = (

        math.ceil(
            raw_row_stop
            / CORE_SIZE
        )

        * CORE_SIZE

    )


    aligned_col_stop = (

        math.ceil(
            raw_col_stop
            / CORE_SIZE
        )

        * CORE_SIZE

    )


    aligned_row_start = max(
        0,
        aligned_row_start
    )


    aligned_col_start = max(
        0,
        aligned_col_start
    )


    aligned_row_stop = min(
        src.height,
        aligned_row_stop
    )


    aligned_col_stop = min(
        src.width,
        aligned_col_stop
    )


    output_height = (

        aligned_row_stop
        - aligned_row_start

    )


    output_width = (

        aligned_col_stop
        - aligned_col_start

    )


    if (
        output_height <= 0
        or output_width <= 0
    ):

        raise ValueError(
            "Test-area output window is empty."
        )


    output_window = Window(

        aligned_col_start,
        aligned_row_start,

        output_width,
        output_height

    )


    output_transform = (
        src.window_transform(
            output_window
        )
    )


    raster_crs = src.crs


    print(
        f"\nOrthomosaic size: "
        f"{src.width:,} x "
        f"{src.height:,}"
    )


    print(
        f"Resolution: "
        f"{src.res}"
    )


    print(
        f"Output test raster size: "
        f"{output_width:,} x "
        f"{output_height:,}"
    )


# =============================================================================
# 13. CREATE INFERENCE WINDOWS
# =============================================================================

prediction_windows = []


for global_row in range(

    aligned_row_start,
    aligned_row_stop,
    CORE_SIZE

):


    core_height = min(

        CORE_SIZE,

        aligned_row_stop
        - global_row

    )


    for global_col in range(

        aligned_col_start,
        aligned_col_stop,
        CORE_SIZE

    ):


        core_width = min(

            CORE_SIZE,

            aligned_col_stop
            - global_col

        )


        prediction_windows.append(
            (
                global_row,
                global_col,
                core_height,
                core_width
            )
        )


print(
    f"Spatial prediction windows: "
    f"{len(prediction_windows):,}"
)


# =============================================================================
# 14. OUTPUT RASTER PROFILES
# =============================================================================

class_profile = {

    "driver":
        "GTiff",

    "height":
        output_height,

    "width":
        output_width,

    "count":
        1,

    "dtype":
        "uint8",

    "crs":
        raster_crs,

    "transform":
        output_transform,

    "nodata":
        CLASS_NODATA,

    "compress":
        "deflate",

    "tiled":
        True,

    "blockxsize":
        256,

    "blockysize":
        256,

    "BIGTIFF":
        "IF_SAFER"

}


probability_profile = {

    "driver":
        "GTiff",

    "height":
        output_height,

    "width":
        output_width,

    "count":
        1,

    "dtype":
        "float32",

    "crs":
        raster_crs,

    "transform":
        output_transform,

    "nodata":
        PROBABILITY_NODATA,

    "compress":
        "deflate",

    "predictor":
        2,

    "tiled":
        True,

    "blockxsize":
        256,

    "blockysize":
        256,

    "BIGTIFF":
        "IF_SAFER"

}


# =============================================================================
# 15. READ 256 x 256 CONTEXT TILE
# =============================================================================

def read_context_tile(
    src,
    global_row,
    global_col
):


    input_window = Window(

        global_col
        - HALO,

        global_row
        - HALO,

        TILE_SIZE,
        TILE_SIZE

    )


    tile = src.read(

        RGB_BANDS,

        window=input_window,

        boundless=True,

        fill_value=0

    )


    tile = (

        tile
        .transpose(
            1,
            2,
            0
        )
        .astype(
            np.float32
        )

    )


    if tile.shape != (
        TILE_SIZE,
        TILE_SIZE,
        3
    ):

        raise RuntimeError(
            f"Unexpected image tile shape: "
            f"{tile.shape}"
        )


    # Same normalization used during training.
    tile /= 255.0


    return tile


# =============================================================================
# 16. RUN SPATIAL INFERENCE
# =============================================================================

with rasterio.open(
    ORTHOMOSAIC_PATH
) as src, \
     rasterio.open(
         CLASS_RASTER_PATH,
         "w",
         **class_profile
     ) as class_dst, \
     rasterio.open(
         WASTE_PROBABILITY_PATH,
         "w",
         **probability_profile
     ) as probability_dst:


    class_dst.set_band_description(
        1,
        "Predicted raw land-cover class"
    )


    probability_dst.set_band_description(
        1,
        "Waste probability"
    )


    class_dst.update_tags(

        prediction_type="multiclass_land_cover",

        valid_classes="1-10",

        outside_test_area="0"

    )


    probability_dst.update_tags(

        target_class="Waste",

        waste_raw_label="1",

        waste_model_channel="0",

        probability_range="0-1"

    )


    n_windows = len(
        prediction_windows
    )


    for batch_start in range(

        0,
        n_windows,
        BATCH_SIZE

    ):


        batch_meta = prediction_windows[

            batch_start:
            batch_start
            + BATCH_SIZE

        ]


        batch_tiles = []


        for (
            global_row,
            global_col,
            core_height,
            core_width
        ) in batch_meta:


            batch_tiles.append(

                read_context_tile(

                    src,
                    global_row,
                    global_col

                )

            )


        batch_tiles = np.stack(
            batch_tiles,
            axis=0
        )


        predictions = model.predict(

            batch_tiles,

            verbose=0

        )


        for prediction, (
            global_row,
            global_col,
            core_height,
            core_width
        ) in zip(
            predictions,
            batch_meta
        ):


            # -----------------------------------------------------------------
            # Central output only
            # -----------------------------------------------------------------

            core_probabilities = prediction[

                HALO:
                HALO + core_height,

                HALO:
                HALO + core_width,

                :

            ]


            # -----------------------------------------------------------------
            # Multiclass prediction
            #
            # model index 0-9 -> raw class 1-10
            # -----------------------------------------------------------------

            predicted_class = (

                np.argmax(
                    core_probabilities,
                    axis=-1
                )
                .astype(
                    np.uint8
                )

                + 1

            )


            # -----------------------------------------------------------------
            # Waste probability
            # -----------------------------------------------------------------

            waste_probability = (

                core_probabilities[
                    ...,
                    WASTE_CLASS_INDEX
                ]

                .astype(
                    np.float32
                )

            )


            global_core_window = Window(

                global_col,
                global_row,

                core_width,
                core_height

            )


            core_transform = (
                src.window_transform(
                    global_core_window
                )
            )


            # -----------------------------------------------------------------
            # Test-area polygon mask
            # -----------------------------------------------------------------

            inside_test_area = geometry_mask(

                [
                    mapping(
                        test_geometry
                    )
                ],

                out_shape=(
                    core_height,
                    core_width
                ),

                transform=core_transform,

                invert=True,

                all_touched=False

            )


            predicted_class[
                ~inside_test_area
            ] = CLASS_NODATA


            waste_probability[
                ~inside_test_area
            ] = PROBABILITY_NODATA


            local_window = Window(

                global_col
                - aligned_col_start,

                global_row
                - aligned_row_start,

                core_width,
                core_height

            )


            class_dst.write(

                predicted_class,

                1,

                window=local_window

            )


            probability_dst.write(

                waste_probability,

                1,

                window=local_window

            )


        completed = min(

            batch_start
            + len(
                batch_meta
            ),

            n_windows

        )


        percentage = (

            100
            * completed
            / n_windows

        )


        print(

            f"\rSpatial prediction: "
            f"{completed:,}/"
            f"{n_windows:,} "
            f"({percentage:6.2f}%)",

            end="",

            flush=True

        )


print("\n")


# =============================================================================
# PART B
# INDEPENDENT PIXEL-LEVEL ACCURACY EVALUATION
# =============================================================================


# =============================================================================
# 17. DISCOVER TEST IMAGE-MASK PAIRS
# =============================================================================

print("\n")
print("=" * 80)
print("PART B: WITHHELD TEST-DATA EVALUATION")
print("=" * 80)


validation_images = {

    path.name:
        path

    for path in (
        VALIDATION_IMAGE_DIR.glob(
            "*.tif"
        )
    )

    if not path.name.endswith(
        ".aux.xml"
    )

}


validation_masks = {

    path.name:
        path

    for path in (
        VALIDATION_MASK_DIR.glob(
            "*.tif"
        )
    )

    if not path.name.endswith(
        ".aux.xml"
    )

}


common_test_files = sorted(

    set(
        validation_images.keys()
    )
    .intersection(
        validation_masks.keys()
    )

)


if not common_test_files:

    raise FileNotFoundError(
        "\nNo matching Validation image-mask pairs found."
    )


print(
    f"\nMatched withheld test pairs: "
    f"{len(common_test_files):,}"
)


# =============================================================================
# 18. METRIC FUNCTION
# =============================================================================

def compute_metrics(
    tp,
    fp,
    fn,
    tn
):


    total = (
        tp
        + fp
        + fn
        + tn
    )


    accuracy = (

        (
            tp
            + tn
        )
        / total

        if total > 0

        else np.nan

    )


    precision = (

        tp
        / (
            tp
            + fp
        )

        if (
            tp
            + fp
        ) > 0

        else 0.0

    )


    recall = (

        tp
        / (
            tp
            + fn
        )

        if (
            tp
            + fn
        ) > 0

        else 0.0

    )


    f1 = (

        2
        * precision
        * recall
        / (
            precision
            + recall
        )

        if (
            precision
            + recall
        ) > 0

        else 0.0

    )


    iou = (

        tp
        / (
            tp
            + fp
            + fn
        )

        if (
            tp
            + fp
            + fn
        ) > 0

        else 0.0

    )


    return {

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "iou":
            iou

    }


# =============================================================================
# 19. GLOBAL CONFUSION COUNTS
# =============================================================================

# -----------------------------------------------------------------------------
# Primary:
# multiclass ARGMAX -> waste / non-waste
# -----------------------------------------------------------------------------

argmax_counts = {

    "TP": 0,
    "FP": 0,
    "FN": 0,
    "TN": 0

}


# -----------------------------------------------------------------------------
# Secondary:
# waste probability >= 0.50
# -----------------------------------------------------------------------------

threshold_counts = {

    "TP": 0,
    "FP": 0,
    "FN": 0,
    "TN": 0

}


# -----------------------------------------------------------------------------
# Valid-label-only ARGMAX diagnostic
#
# This excludes pixels whose reference value is not 1-10.
# -----------------------------------------------------------------------------

clean_argmax_counts = {

    "TP": 0,
    "FP": 0,
    "FN": 0,
    "TN": 0

}


# =============================================================================
# 20. EVALUATE EACH HELD-OUT TILE
# =============================================================================

per_tile_records = []

mask_audit_records = []


for tile_number, filename in enumerate(
    common_test_files,
    start=1
):


    image_path = (
        validation_images[
            filename
        ]
    )


    mask_path = (
        validation_masks[
            filename
        ]
    )


    # =========================================================================
    # READ RGB
    # =========================================================================

    with rasterio.open(
        image_path
    ) as src:


        image = src.read(
            RGB_BANDS
        )


    image = (

        image
        .transpose(
            1,
            2,
            0
        )
        .astype(
            np.float32
        )

    )


    # =========================================================================
    # READ REFERENCE MASK
    # =========================================================================

    with rasterio.open(
        mask_path
    ) as src:


        reference_mask = src.read(
            1
        )


    reference_mask = np.rint(
        reference_mask
    ).astype(
        np.int32
    )


    # =========================================================================
    # SHAPE CHECK
    # =========================================================================

    if image.shape[:2] != reference_mask.shape:

        raise ValueError(
            "\nImage/mask size mismatch:\n"
            f"{filename}\n"
            f"Image: {image.shape}\n"
            f"Mask : {reference_mask.shape}"
        )


    if image.shape[:2] != (
        TILE_SIZE,
        TILE_SIZE
    ):

        raise ValueError(
            "\nUnexpected test-tile dimensions:\n"
            f"{filename}\n"
            f"{image.shape}"
        )


    # =========================================================================
    # REFERENCE LABEL AUDIT
    # =========================================================================

    observed_labels = sorted(

        int(
            value
        )

        for value in np.unique(
            reference_mask
        )

    )


    invalid_labels = sorted(

        set(
            observed_labels
        )

        - VALID_RAW_LABELS

    )


    invalid_pixel_mask = ~np.isin(

        reference_mask,

        list(
            VALID_RAW_LABELS
        )

    )


    n_invalid_pixels = int(

        np.sum(
            invalid_pixel_mask
        )

    )


    mask_audit_records.append(
        {

            "filename":
                filename,

            "observed_labels":
                ",".join(
                    str(v)
                    for v in observed_labels
                ),

            "invalid_labels":
                ",".join(
                    str(v)
                    for v in invalid_labels
                ),

            "invalid_pixels":
                n_invalid_pixels

        }
    )


    # =========================================================================
    # PREPROCESS IMAGE
    # =========================================================================

    image /= 255.0


    batch = np.expand_dims(

        image,

        axis=0

    )


    # =========================================================================
    # PREDICT
    # =========================================================================

    prediction = model.predict(

        batch,

        verbose=0

    )[0]


    # Shape:
    #
    # 256 x 256 x 10

    if prediction.shape != (
        TILE_SIZE,
        TILE_SIZE,
        N_CLASSES
    ):

        raise RuntimeError(
            "\nUnexpected model prediction shape:\n"
            f"{prediction.shape}"
        )


    # =========================================================================
    # PRIMARY PREDICTION:
    # MULTICLASS ARGMAX
    # =========================================================================

    predicted_model_channel = np.argmax(

        prediction,

        axis=-1

    )


    predicted_raw_class = (

        predicted_model_channel
        + 1

    )


    predicted_waste_argmax = (

        predicted_raw_class
        == WASTE_RAW_LABEL

    )


    # =========================================================================
    # SECONDARY PREDICTION:
    # FIXED P(WASTE) >= 0.50
    # =========================================================================

    waste_probability = prediction[
        ...,
        WASTE_CLASS_INDEX
    ]


    predicted_waste_threshold = (

        waste_probability
        >= FIXED_WASTE_THRESHOLD

    )


    # =========================================================================
    # GROUND TRUTH
    #
    # Legacy-compatible definition:
    #
    # raw class 1 = waste
    # every other value = non-waste
    #
    # This reproduces the earlier binary evaluation convention.
    # =========================================================================

    true_waste = (

        reference_mask
        == WASTE_RAW_LABEL

    )


    # =========================================================================
    # PRIMARY ARGMAX CONFUSION MATRIX
    # =========================================================================

    tp = int(
        np.sum(
            predicted_waste_argmax
            & true_waste
        )
    )


    fp = int(
        np.sum(
            predicted_waste_argmax
            & (~true_waste)
        )
    )


    fn = int(
        np.sum(
            (~predicted_waste_argmax)
            & true_waste
        )
    )


    tn = int(
        np.sum(
            (~predicted_waste_argmax)
            & (~true_waste)
        )
    )


    argmax_counts[
        "TP"
    ] += tp


    argmax_counts[
        "FP"
    ] += fp


    argmax_counts[
        "FN"
    ] += fn


    argmax_counts[
        "TN"
    ] += tn


    # =========================================================================
    # FIXED-THRESHOLD CONFUSION MATRIX
    # =========================================================================

    tp_t = int(
        np.sum(
            predicted_waste_threshold
            & true_waste
        )
    )


    fp_t = int(
        np.sum(
            predicted_waste_threshold
            & (~true_waste)
        )
    )


    fn_t = int(
        np.sum(
            (~predicted_waste_threshold)
            & true_waste
        )
    )


    tn_t = int(
        np.sum(
            (~predicted_waste_threshold)
            & (~true_waste)
        )
    )


    threshold_counts[
        "TP"
    ] += tp_t


    threshold_counts[
        "FP"
    ] += fp_t


    threshold_counts[
        "FN"
    ] += fn_t


    threshold_counts[
        "TN"
    ] += tn_t


    # =========================================================================
    # VALID-LABEL-ONLY ARGMAX DIAGNOSTIC
    #
    # Exclude any reference pixels outside raw labels 1-10.
    # =========================================================================

    valid_reference = np.isin(

        reference_mask,

        list(
            VALID_RAW_LABELS
        )

    )


    pred_clean = predicted_waste_argmax[
        valid_reference
    ]


    truth_clean = true_waste[
        valid_reference
    ]


    clean_argmax_counts[
        "TP"
    ] += int(
        np.sum(
            pred_clean
            & truth_clean
        )
    )


    clean_argmax_counts[
        "FP"
    ] += int(
        np.sum(
            pred_clean
            & (~truth_clean)
        )
    )


    clean_argmax_counts[
        "FN"
    ] += int(
        np.sum(
            (~pred_clean)
            & truth_clean
        )
    )


    clean_argmax_counts[
        "TN"
    ] += int(
        np.sum(
            (~pred_clean)
            & (~truth_clean)
        )
    )


    # =========================================================================
    # PER-TILE METRICS
    # =========================================================================

    tile_metrics = compute_metrics(
        tp,
        fp,
        fn,
        tn
    )


    per_tile_records.append(
        {

            "tile":
                filename,

            "TP":
                tp,

            "FP":
                fp,

            "FN":
                fn,

            "TN":
                tn,

            "accuracy":
                tile_metrics[
                    "accuracy"
                ],

            "precision":
                tile_metrics[
                    "precision"
                ],

            "recall":
                tile_metrics[
                    "recall"
                ],

            "f1":
                tile_metrics[
                    "f1"
                ],

            "iou":
                tile_metrics[
                    "iou"
                ],

            "reference_waste_pixels":
                int(
                    np.sum(
                        true_waste
                    )
                ),

            "predicted_waste_pixels":
                int(
                    np.sum(
                        predicted_waste_argmax
                    )
                ),

            "invalid_reference_pixels":
                n_invalid_pixels

        }
    )


    if (
        tile_number % 20 == 0
        or tile_number == len(
            common_test_files
        )
    ):


        print(

            f"\rEvaluated "
            f"{tile_number:,}/"
            f"{len(common_test_files):,} "
            f"test tiles",

            end="",

            flush=True

        )


print("\n")


# =============================================================================
# 21. SAVE MASK AUDIT
# =============================================================================

mask_audit_df = pd.DataFrame(
    mask_audit_records
)


mask_audit_df.to_csv(
    MASK_AUDIT_PATH,
    index=False
)


n_masks_with_invalid = int(

    (
        mask_audit_df[
            "invalid_pixels"
        ]
        > 0
    ).sum()

)


total_invalid_test_pixels = int(

    mask_audit_df[
        "invalid_pixels"
    ].sum()

)


# =============================================================================
# 22. CALCULATE GLOBAL METRICS
# =============================================================================

argmax_metrics = compute_metrics(

    argmax_counts[
        "TP"
    ],

    argmax_counts[
        "FP"
    ],

    argmax_counts[
        "FN"
    ],

    argmax_counts[
        "TN"
    ]

)


threshold_metrics = compute_metrics(

    threshold_counts[
        "TP"
    ],

    threshold_counts[
        "FP"
    ],

    threshold_counts[
        "FN"
    ],

    threshold_counts[
        "TN"
    ]

)


clean_argmax_metrics = compute_metrics(

    clean_argmax_counts[
        "TP"
    ],

    clean_argmax_counts[
        "FP"
    ],

    clean_argmax_counts[
        "FN"
    ],

    clean_argmax_counts[
        "TN"
    ]

)


# =============================================================================
# 23. CREATE ACCURACY TABLE
# =============================================================================

accuracy_results = pd.DataFrame(

    [

        {

            "evaluation":
                "Multiclass_argmax_primary",

            "model":
                MODEL_PATH.name,

            "test_tiles":
                len(
                    common_test_files
                ),

            "decision_rule":
                "argmax_class_equals_raw_class_1",

            "TP":
                argmax_counts[
                    "TP"
                ],

            "FP":
                argmax_counts[
                    "FP"
                ],

            "FN":
                argmax_counts[
                    "FN"
                ],

            "TN":
                argmax_counts[
                    "TN"
                ],

            "Accuracy":
                argmax_metrics[
                    "accuracy"
                ],

            "Precision":
                argmax_metrics[
                    "precision"
                ],

            "Recall":
                argmax_metrics[
                    "recall"
                ],

            "F1":
                argmax_metrics[
                    "f1"
                ],

            "IoU":
                argmax_metrics[
                    "iou"
                ]

        },


        {

            "evaluation":
                "Waste_probability_fixed_0.50_secondary",

            "model":
                MODEL_PATH.name,

            "test_tiles":
                len(
                    common_test_files
                ),

            "decision_rule":
                "waste_probability_ge_0.50",

            "TP":
                threshold_counts[
                    "TP"
                ],

            "FP":
                threshold_counts[
                    "FP"
                ],

            "FN":
                threshold_counts[
                    "FN"
                ],

            "TN":
                threshold_counts[
                    "TN"
                ],

            "Accuracy":
                threshold_metrics[
                    "accuracy"
                ],

            "Precision":
                threshold_metrics[
                    "precision"
                ],

            "Recall":
                threshold_metrics[
                    "recall"
                ],

            "F1":
                threshold_metrics[
                    "f1"
                ],

            "IoU":
                threshold_metrics[
                    "iou"
                ]

        },


        {

            "evaluation":
                "Multiclass_argmax_valid_labels_1_to_10_only",

            "model":
                MODEL_PATH.name,

            "test_tiles":
                len(
                    common_test_files
                ),

            "decision_rule":
                "argmax_class_1_valid_reference_labels_only",

            "TP":
                clean_argmax_counts[
                    "TP"
                ],

            "FP":
                clean_argmax_counts[
                    "FP"
                ],

            "FN":
                clean_argmax_counts[
                    "FN"
                ],

            "TN":
                clean_argmax_counts[
                    "TN"
                ],

            "Accuracy":
                clean_argmax_metrics[
                    "accuracy"
                ],

            "Precision":
                clean_argmax_metrics[
                    "precision"
                ],

            "Recall":
                clean_argmax_metrics[
                    "recall"
                ],

            "F1":
                clean_argmax_metrics[
                    "f1"
                ],

            "IoU":
                clean_argmax_metrics[
                    "iou"
                ]

        }

    ]

)


accuracy_results.to_csv(
    ACCURACY_CSV_PATH,
    index=False
)


# =============================================================================
# 24. SAVE PER-TILE METRICS
# =============================================================================

pd.DataFrame(
    per_tile_records
).to_csv(
    PER_TILE_CSV_PATH,
    index=False
)


# =============================================================================
# 25. PRINT PRIMARY RESULTS
# =============================================================================

print("\n")
print("=" * 80)
print("FINAL TEST-AREA ACCURACY")
print("=" * 80)


print(
    "\nPRIMARY RESULT:"
)


print(
    "Multiclass U-Net ARGMAX, "
    "raw class 1 = waste"
)


print(
    "\nConfusion matrix:"
)


print(
    f"TP = "
    f"{argmax_counts['TP']:,}"
)


print(
    f"FP = "
    f"{argmax_counts['FP']:,}"
)


print(
    f"FN = "
    f"{argmax_counts['FN']:,}"
)


print(
    f"TN = "
    f"{argmax_counts['TN']:,}"
)


print("\nMetrics:")


print(
    f"Accuracy  = "
    f"{argmax_metrics['accuracy']:.4f}"
)


print(
    f"Precision = "
    f"{argmax_metrics['precision']:.4f}"
)


print(
    f"Recall    = "
    f"{argmax_metrics['recall']:.4f}"
)


print(
    f"F1        = "
    f"{argmax_metrics['f1']:.4f}"
)


print(
    f"IoU       = "
    f"{argmax_metrics['iou']:.4f}"
)


# =============================================================================
# 26. PRINT SECONDARY FIXED-THRESHOLD RESULTS
# =============================================================================

print("\n")
print("-" * 80)


print(
    f"SECONDARY RESULT: "
    f"P(waste) >= "
    f"{FIXED_WASTE_THRESHOLD:.2f}"
)


print(
    f"Precision = "
    f"{threshold_metrics['precision']:.4f}"
)


print(
    f"Recall    = "
    f"{threshold_metrics['recall']:.4f}"
)


print(
    f"F1        = "
    f"{threshold_metrics['f1']:.4f}"
)


print(
    f"IoU       = "
    f"{threshold_metrics['iou']:.4f}"
)


# =============================================================================
# 27. PRINT REFERENCE MASK AUDIT
# =============================================================================

print("\n")
print("-" * 80)


print(
    "REFERENCE MASK AUDIT"
)


print(
    f"Test masks with values outside 1-10: "
    f"{n_masks_with_invalid:,}"
)


print(
    f"Pixels outside 1-10: "
    f"{total_invalid_test_pixels:,}"
)


if total_invalid_test_pixels > 0:


    print(
        "\nBecause some historical reference masks contain "
        "values outside the current 1-10 class system, "
        "a valid-label-only ARGMAX result was also calculated."
    )


    print(
        f"\nValid-label-only Precision = "
        f"{clean_argmax_metrics['precision']:.4f}"
    )


    print(
        f"Valid-label-only Recall    = "
        f"{clean_argmax_metrics['recall']:.4f}"
    )


    print(
        f"Valid-label-only F1        = "
        f"{clean_argmax_metrics['f1']:.4f}"
    )


    print(
        f"Valid-label-only IoU       = "
        f"{clean_argmax_metrics['iou']:.4f}"
    )


# =============================================================================
# 28. SAVE TEXT REPORT
# =============================================================================

with open(
    ACCURACY_TXT_PATH,
    "w"
) as f:


    f.write(
        "HARD-NEGATIVE FINE-TUNED U-NET\n"
    )


    f.write(
        "INDEPENDENT TEST ACCURACY\n"
    )


    f.write(
        "=" * 70
        + "\n\n"
    )


    f.write(
        f"Model: "
        f"{MODEL_PATH}\n"
    )


    f.write(
        f"Test image-mask pairs: "
        f"{len(common_test_files)}\n\n"
    )


    f.write(
        "PRIMARY EVALUATION\n"
    )


    f.write(
        "Multiclass ARGMAX; raw class 1 = waste\n"
    )


    f.write(
        "-" * 70
        + "\n"
    )


    f.write(
        f"TP: "
        f"{argmax_counts['TP']}\n"
    )


    f.write(
        f"FP: "
        f"{argmax_counts['FP']}\n"
    )


    f.write(
        f"FN: "
        f"{argmax_counts['FN']}\n"
    )


    f.write(
        f"TN: "
        f"{argmax_counts['TN']}\n\n"
    )


    f.write(
        f"Accuracy : "
        f"{argmax_metrics['accuracy']:.6f}\n"
    )


    f.write(
        f"Precision: "
        f"{argmax_metrics['precision']:.6f}\n"
    )


    f.write(
        f"Recall   : "
        f"{argmax_metrics['recall']:.6f}\n"
    )


    f.write(
        f"F1       : "
        f"{argmax_metrics['f1']:.6f}\n"
    )


    f.write(
        f"IoU      : "
        f"{argmax_metrics['iou']:.6f}\n\n"
    )


    f.write(
        "SECONDARY EVALUATION\n"
    )


    f.write(
        f"Fixed waste probability threshold = "
        f"{FIXED_WASTE_THRESHOLD:.2f}\n"
    )


    f.write(
        "-" * 70
        + "\n"
    )


    f.write(
        f"Precision: "
        f"{threshold_metrics['precision']:.6f}\n"
    )


    f.write(
        f"Recall   : "
        f"{threshold_metrics['recall']:.6f}\n"
    )


    f.write(
        f"F1       : "
        f"{threshold_metrics['f1']:.6f}\n"
    )


    f.write(
        f"IoU      : "
        f"{threshold_metrics['iou']:.6f}\n\n"
    )


    f.write(
        "REFERENCE MASK AUDIT\n"
    )


    f.write(
        "-" * 70
        + "\n"
    )


    f.write(
        f"Masks containing values outside 1-10: "
        f"{n_masks_with_invalid}\n"
    )


    f.write(
        f"Total pixels outside 1-10: "
        f"{total_invalid_test_pixels}\n"
    )


# =============================================================================
# 29. FINAL OUTPUT REPORT
# =============================================================================

print("\n")
print("=" * 80)
print("OUTPUTS")
print("=" * 80)


print(
    "\nMulticlass prediction surface:"
)

print(
    CLASS_RASTER_PATH
)


print(
    "\nWaste-probability surface:"
)

print(
    WASTE_PROBABILITY_PATH
)


print(
    "\nAccuracy metrics:"
)

print(
    ACCURACY_CSV_PATH
)


print(
    "\nAccuracy text report:"
)

print(
    ACCURACY_TXT_PATH
)


print(
    "\nPer-tile metrics:"
)

print(
    PER_TILE_CSV_PATH
)


print(
    "\nReference-mask audit:"
)

print(
    MASK_AUDIT_PATH
)


print("\nDone.")