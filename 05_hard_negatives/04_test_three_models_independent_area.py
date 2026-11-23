# =============================================================================
# INDEPENDENT TEST:
# MARCH BASELINE vs TWO HARD-NEGATIVE FINE-TUNED U-NETS
# =============================================================================
#
# Primary comparison:
#   threshold = 0.50
#
# Additional thresholds (0.60-0.90) are reported only as descriptive
# sensitivity analyses. They are NOT used to choose/tune a threshold on the
# independent test area.
#
# Evaluation:
#   1. Pixel-level precision, recall, F1 and IoU.
#   2. Object-level prediction count, matched reference piles, false-positive
#      objects, missed reference piles, precision, recall and F1.
#   3. Predicted waste area versus reference waste area.
#
# Object definition:
#   - 8-connected thresholded waste pixels
#   - minimum mapping unit = 0.25 m2
#
# Reference matching:
#   - touching/overlapping reference polygons are merged into contiguous units
#   - a candidate match must cover >=1% of a reference object's area
#   - one-to-one greedy assignment, prioritising IoU, then reference coverage
#
# Inference:
#   - model input = 256 x 256 RGB
#   - /255 preprocessing
#   - 64-pixel context halo
#   - retain central 128 x 128 predictions
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
import hashlib
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import tensorflow as tf

from rasterio.features import rasterize, shapes
from rasterio.windows import Window
from rasterio.windows import from_bounds
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid
from tensorflow.keras.models import load_model


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
# Imagery used by the March model
# -----------------------------------------------------------------------------

ORTHO_PATH = (
    BASE_PATH
    / "data"
    / "optical.tiff"
)


# -----------------------------------------------------------------------------
# LOCKED independent test area
# -----------------------------------------------------------------------------

TEST_AOI_PATH = Path(
    r"C:\Users\Peter\Documents\GitHub\waste_spatio_temporal"
    r"\inputs\finetuning\AOI\AOI.shp"
)


# -----------------------------------------------------------------------------
# Manually digitised waste inside the test area
# -----------------------------------------------------------------------------

REFERENCE_WASTE_PATH = (
    BASE_PATH
    / "data"
    / "vectors"
    / "waste_probability_step0"
    / "waste.shp"
)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

BASELINE_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_march_finetuned_best.keras"
)


COMPLEMENTARY_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_complementary_rotating_finetuned.keras"
)


ERROR_DIRECTED_MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_error_directed_soft_reassignment_finetuned.keras"
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

RESULTS_DIR = (
    BASE_PATH
    / "results"
    / "independent_test_three_models"
)


PROBABILITY_DIR = (
    RESULTS_DIR
    / "probability"
)


VECTOR_DIR = (
    RESULTS_DIR
    / "vectors"
)


TEMP_DIR = (
    RESULTS_DIR
    / "_temporary"
)


for directory in [
    RESULTS_DIR,
    PROBABILITY_DIR,
    VECTOR_DIR,
    TEMP_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# 3. SETTINGS
# =============================================================================

SEED = 20260919

np.random.seed(
    SEED
)

tf.random.set_seed(
    SEED
)


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

MODEL_TILE = 256

CORE_SIZE = 128

HALO = (
    (MODEL_TILE - CORE_SIZE)
    // 2
)

INFERENCE_BATCH_SIZE = 16


# -----------------------------------------------------------------------------
# Waste class
# -----------------------------------------------------------------------------

WASTE_CLASS_IDX = 0


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

PRIMARY_THRESHOLD = 0.50

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


MIN_OBJECT_AREA_M2 = 0.25

CONNECTIVITY = 8

MIN_REFERENCE_COVERAGE = 0.01


# -----------------------------------------------------------------------------
# Probability raster
# -----------------------------------------------------------------------------

PROBABILITY_NODATA = -9999.0

PROBABILITY_DTYPE = "float32"


# =============================================================================
# 4. HELPERS
# =============================================================================

def sha256_file(
    path,
    chunk_size=1024 * 1024,
):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def safe_make_valid_gdf(
    gdf
):

    gdf = gdf[
        ~gdf.geometry.isna()
    ].copy()


    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()


    gdf[
        "geometry"
    ] = gdf.geometry.apply(
        make_valid
    )


    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()


    return gdf


def pixel_area_m2(
    transform
):

    return abs(
        transform.a
        * transform.e
    )


def f1_from_pr(
    precision,
    recall,
):

    denominator = (
        precision
        + recall
    )

    if denominator == 0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / denominator
    )


def threshold_code(
    threshold
):

    return (
        f"{int(round(threshold * 100)):03d}"
    )


# =============================================================================
# 5. CHECK INPUTS
# =============================================================================

required_paths = [
    ORTHO_PATH,
    TEST_AOI_PATH,
    REFERENCE_WASTE_PATH,
    BASELINE_MODEL_PATH,
    COMPLEMENTARY_MODEL_PATH,
    ERROR_DIRECTED_MODEL_PATH,
]


missing_paths = [
    path

    for path in required_paths

    if not path.exists()
]


if missing_paths:

    raise FileNotFoundError(
        "\nMissing required input(s):\n"
        + "\n".join(
            str(
                path
            )
            for path in missing_paths
        )
    )


print("\n")
print("=" * 80)
print("INDEPENDENT TEST: BASELINE vs COMPLEMENTARY vs ERROR-DIRECTED MODEL")
print("=" * 80)


print(
    f"\nOrthomosaic:\n"
    f"{ORTHO_PATH}"
)


print(
    f"\nLocked test AOI:\n"
    f"{TEST_AOI_PATH}"
)


print(
    f"\nReference waste:\n"
    f"{REFERENCE_WASTE_PATH}"
)


print(
    f"\nBaseline model:\n"
    f"{BASELINE_MODEL_PATH}"
)


print(
    f"\nComplementary-only model:\n"
    f"{COMPLEMENTARY_MODEL_PATH}"
)


print(
    f"\nError-directed soft-reassignment model:\n"
    f"{ERROR_DIRECTED_MODEL_PATH}"
)


# =============================================================================
# 6. READ RASTER / TEST AOI
# =============================================================================

with rasterio.open(
    ORTHO_PATH
) as src:

    raster_crs = (
        src.crs
    )

    raster_transform = (
        src.transform
    )

    raster_width = (
        src.width
    )

    raster_height = (
        src.height
    )

    raster_profile = (
        src.profile.copy()
    )


test_aoi = gpd.read_file(
    TEST_AOI_PATH
)


test_aoi = safe_make_valid_gdf(
    test_aoi
)


if (
    test_aoi.crs
    != raster_crs
):

    test_aoi = test_aoi.to_crs(
        raster_crs
    )


test_union = unary_union(
    test_aoi.geometry.tolist()
)


# -----------------------------------------------------------------------------
# Raster-aligned bounding window around test AOI
# -----------------------------------------------------------------------------

aoi_bounds = (
    test_union.bounds
)


with rasterio.open(
    ORTHO_PATH
) as src:

    test_window = from_bounds(
        *aoi_bounds,
        transform=src.transform,
    )


    # Expand to whole source pixels.
    row_off = max(
        0,
        int(
            math.floor(
                test_window.row_off
            )
        ),
    )

    col_off = max(
        0,
        int(
            math.floor(
                test_window.col_off
            )
        ),
    )

    row_end = min(
        src.height,
        int(
            math.ceil(
                test_window.row_off
                + test_window.height
            )
        ),
    )

    col_end = min(
        src.width,
        int(
            math.ceil(
                test_window.col_off
                + test_window.width
            )
        ),
    )


    test_window = Window(
        col_off=col_off,
        row_off=row_off,
        width=(
            col_end
            - col_off
        ),
        height=(
            row_end
            - row_off
        ),
    )


    test_transform = (
        rasterio.windows.transform(
            test_window,
            src.transform,
        )
    )


TEST_HEIGHT = int(
    test_window.height
)

TEST_WIDTH = int(
    test_window.width
)


PIXEL_AREA_M2 = pixel_area_m2(
    test_transform
)


print("\n")
print(
    f"Test raster window: "
    f"{TEST_WIDTH:,} x "
    f"{TEST_HEIGHT:,} pixels"
)


print(
    f"Pixel area: "
    f"{PIXEL_AREA_M2:.8f} m2"
)


# =============================================================================
# 7. PREPARE AOI MASK
# =============================================================================

print("\nRasterising test AOI mask...")


AOI_MASK_PATH = (
    TEMP_DIR
    / "test_aoi_mask.dat"
)


aoi_mask = np.memmap(

    AOI_MASK_PATH,

    dtype="uint8",

    mode="w+",

    shape=(
        TEST_HEIGHT,
        TEST_WIDTH,
    ),

)


aoi_mask[:] = rasterize(

    [
        (
            test_union,
            1,
        )
    ],

    out_shape=(
        TEST_HEIGHT,
        TEST_WIDTH,
    ),

    transform=test_transform,

    fill=0,

    default_value=1,

    dtype="uint8",

    all_touched=False,

)


aoi_mask.flush()


N_AOI_PIXELS = int(
    np.sum(
        aoi_mask
    )
)


print(
    f"Pixels inside test AOI: "
    f"{N_AOI_PIXELS:,}"
)


print(
    f"Rasterised AOI area: "
    f"{N_AOI_PIXELS * PIXEL_AREA_M2:,.3f} m2 "
    f"({N_AOI_PIXELS * PIXEL_AREA_M2 / 1e6:.6f} km2)"
)


# =============================================================================
# 8. PREPARE REFERENCE WASTE
# =============================================================================

print("\n")
print("=" * 80)
print("PREPARING REFERENCE WASTE")
print("=" * 80)


reference = gpd.read_file(
    REFERENCE_WASTE_PATH
)


reference = safe_make_valid_gdf(
    reference
)


if (
    reference.crs
    != raster_crs
):

    reference = reference.to_crs(
        raster_crs
    )


# -----------------------------------------------------------------------------
# Restrict reference polygons to test AOI.
# -----------------------------------------------------------------------------

reference = reference[
    reference.geometry.intersects(
        test_union
    )
].copy()


reference[
    "geometry"
] = reference.geometry.intersection(
    test_union
)


reference = safe_make_valid_gdf(
    reference
)


REFERENCE_SOURCE_FEATURES = (
    len(
        reference
    )
)


# -----------------------------------------------------------------------------
# Merge touching / overlapping reference polygons into contiguous reference
# objects.
# -----------------------------------------------------------------------------

reference_union = unary_union(
    reference.geometry.tolist()
)


if (
    reference_union.geom_type
    == "Polygon"
):

    reference_parts = [
        reference_union
    ]

elif (
    reference_union.geom_type
    == "MultiPolygon"
):

    reference_parts = list(
        reference_union.geoms
    )

else:

    # GeometryCollection defensive fallback.
    reference_parts = [

        geometry

        for geometry
        in getattr(
            reference_union,
            "geoms",
            []
        )

        if geometry.geom_type
        == "Polygon"

    ]


reference_units = gpd.GeoDataFrame(

    {
        "ref_id":
            np.arange(
                1,
                len(
                    reference_parts
                )
                + 1,
            )
    },

    geometry=reference_parts,

    crs=raster_crs,

)


reference_units[
    "area_m2"
] = reference_units.geometry.area


N_REFERENCE_UNITS = (
    len(
        reference_units
    )
)


REFERENCE_VECTOR_AREA_M2 = float(
    reference_units[
        "area_m2"
    ].sum()
)


print(
    f"Reference source features in AOI: "
    f"{REFERENCE_SOURCE_FEATURES:,}"
)


print(
    f"Contiguous reference waste units: "
    f"{N_REFERENCE_UNITS:,}"
)


print(
    f"Reference vector area: "
    f"{REFERENCE_VECTOR_AREA_M2:,.3f} m2"
)


reference_units.to_file(
    VECTOR_DIR
    / "reference_waste_units.gpkg",
    layer="reference_waste_units",
    driver="GPKG",
)


# -----------------------------------------------------------------------------
# Rasterised reference mask
# -----------------------------------------------------------------------------

REFERENCE_MASK_PATH = (
    TEMP_DIR
    / "reference_mask.dat"
)


reference_mask = np.memmap(

    REFERENCE_MASK_PATH,

    dtype="uint8",

    mode="w+",

    shape=(
        TEST_HEIGHT,
        TEST_WIDTH,
    ),

)


reference_mask[:] = rasterize(

    [
        (
            geometry,
            1,
        )

        for geometry
        in reference_units.geometry
    ],

    out_shape=(
        TEST_HEIGHT,
        TEST_WIDTH,
    ),

    transform=test_transform,

    fill=0,

    default_value=1,

    dtype="uint8",

    all_touched=False,

)


# Mask reference to test AOI.
reference_mask[
    aoi_mask == 0
] = 0


reference_mask.flush()


REFERENCE_RASTER_PIXELS = int(
    np.sum(
        reference_mask
    )
)


REFERENCE_RASTER_AREA_M2 = (
    REFERENCE_RASTER_PIXELS
    * PIXEL_AREA_M2
)


print(
    f"Reference rasterised area: "
    f"{REFERENCE_RASTER_AREA_M2:,.3f} m2"
)


# =============================================================================
# 9. GPU INFORMATION
# =============================================================================

print("\n")
print("=" * 80)
print("TENSORFLOW DEVICE CHECK")
print("=" * 80)


gpus = tf.config.list_physical_devices(
    "GPU"
)


print(
    f"GPUs visible to TensorFlow: "
    f"{len(gpus)}"
)


for gpu in gpus:

    print(
        f"  {gpu}"
    )


# =============================================================================
# 10. INFERENCE FUNCTION
# =============================================================================

def run_seam_reduced_inference(
    model_path,
    model_name,
    output_probability_path,
):

    """
    Predict the waste-class probability over the test AOI.

    Each 128x128 output core receives 64 pixels of context on all sides,
    using a 256x256 U-Net input tile.

    Only the central 128x128 prediction is retained.
    """

    print("\n")
    print("=" * 80)
    print(
        f"INFERENCE: {model_name}"
    )
    print("=" * 80)


    tf.keras.backend.clear_session()


    model = load_model(
        model_path,
        compile=False,
    )


    if tuple(
        model.input_shape[
            1:
        ]
    ) != (
        MODEL_TILE,
        MODEL_TILE,
        3,
    ):

        raise ValueError(
            "\nUnexpected model input shape.\n"
            f"Expected: "
            f"({MODEL_TILE}, {MODEL_TILE}, 3)\n"
            f"Found: {model.input_shape}"
        )


    if int(
        model.output_shape[
            -1
        ]
    ) <= WASTE_CLASS_IDX:

        raise ValueError(
            "Model output does not contain "
            "the requested waste channel."
        )


    probability_profile = (
        raster_profile.copy()
    )


    probability_profile.update(
        driver="GTiff",
        height=TEST_HEIGHT,
        width=TEST_WIDTH,
        count=1,
        transform=test_transform,
        crs=raster_crs,
        dtype=PROBABILITY_DTYPE,
        nodata=PROBABILITY_NODATA,
        compress="deflate",
        tiled=True,
        BIGTIFF="YES",
    )


    row_starts = list(
        range(
            0,
            TEST_HEIGHT,
            CORE_SIZE,
        )
    )


    col_starts = list(
        range(
            0,
            TEST_WIDTH,
            CORE_SIZE,
        )
    )


    total_cores = (
        len(
            row_starts
        )
        * len(
            col_starts
        )
    )


    print(
        f"Core windows: "
        f"{total_cores:,}"
    )


    start_time = time.time()


    with rasterio.open(
        ORTHO_PATH
    ) as src, rasterio.open(
        output_probability_path,
        "w",
        **probability_profile,
    ) as dst:


        # Initialise output to nodata.
        fill_block = np.full(
            (
                1,
                min(
                    1024,
                    TEST_HEIGHT,
                ),
                TEST_WIDTH,
            ),
            PROBABILITY_NODATA,
            dtype=np.float32,
        )


        for r0 in range(
            0,
            TEST_HEIGHT,
            fill_block.shape[1],
        ):

            h = min(
                fill_block.shape[1],
                TEST_HEIGHT
                - r0,
            )

            dst.write(
                fill_block[
                    :,
                    :h,
                    :
                ],
                window=Window(
                    0,
                    r0,
                    TEST_WIDTH,
                    h,
                ),
            )


        batch_images = []

        batch_metadata = []


        completed = 0


        def flush_batch():

            nonlocal completed

            if not batch_images:
                return


            x = np.stack(
                batch_images,
                axis=0,
            ).astype(
                np.float32
            )


            predictions = model.predict(
                x,
                verbose=0,
            )


            for prediction, metadata in zip(
                predictions,
                batch_metadata,
            ):

                (
                    local_row,
                    local_col,
                    core_h,
                    core_w,
                ) = metadata


                core_probability = (
                    prediction[
                        HALO:
                        HALO
                        + core_h,
                        HALO:
                        HALO
                        + core_w,
                        WASTE_CLASS_IDX
                    ]
                    .astype(
                        np.float32
                    )
                )


                # AOI mask corresponding to this output core.
                local_aoi = np.asarray(

                    aoi_mask[
                        local_row:
                        local_row
                        + core_h,
                        local_col:
                        local_col
                        + core_w
                    ],

                    dtype=np.uint8,

                )


                core_probability[
                    local_aoi == 0
                ] = PROBABILITY_NODATA


                dst.write(
                    core_probability[
                        np.newaxis,
                        ...,
                    ],
                    window=Window(
                        local_col,
                        local_row,
                        core_w,
                        core_h,
                    ),
                )


                completed += 1


            batch_images.clear()

            batch_metadata.clear()


            if (
                completed % 100
                == 0
                or completed
                == total_cores
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                print(
                    f"\rCompleted cores: "
                    f"{completed:,} / "
                    f"{total_cores:,} "
                    f"| elapsed "
                    f"{elapsed / 60:.1f} min",
                    end="",
                    flush=True,
                )


        for local_row in (
            row_starts
        ):

            core_h = min(
                CORE_SIZE,
                TEST_HEIGHT
                - local_row,
            )


            for local_col in (
                col_starts
            ):

                core_w = min(
                    CORE_SIZE,
                    TEST_WIDTH
                    - local_col,
                )


                # Skip cores outside AOI entirely.
                local_aoi = aoi_mask[
                    local_row:
                    local_row
                    + core_h,
                    local_col:
                    local_col
                    + core_w
                ]


                if not np.any(
                    local_aoi
                ):

                    completed += 1

                    continue


                # Global raster position of the core.
                global_core_row = (
                    int(
                        test_window.row_off
                    )
                    + local_row
                )


                global_core_col = (
                    int(
                        test_window.col_off
                    )
                    + local_col
                )


                # Read 64 pixels of context around the 128x128 retained core.
                input_window = Window(

                    global_core_col
                    - HALO,

                    global_core_row
                    - HALO,

                    MODEL_TILE,

                    MODEL_TILE,

                )


                image = src.read(

                    [
                        1,
                        2,
                        3,
                    ],

                    window=input_window,

                    boundless=True,

                    fill_value=0,

                )


                image = (
                    image
                    .transpose(
                        1,
                        2,
                        0,
                    )
                    .astype(
                        np.float32
                    )
                    / 255.0
                )


                if image.shape != (
                    MODEL_TILE,
                    MODEL_TILE,
                    3,
                ):

                    raise RuntimeError(
                        f"Unexpected inference tile shape: "
                        f"{image.shape}"
                    )


                batch_images.append(
                    image
                )


                batch_metadata.append(
                    (
                        local_row,
                        local_col,
                        core_h,
                        core_w,
                    )
                )


                if len(
                    batch_images
                ) >= INFERENCE_BATCH_SIZE:

                    flush_batch()


        flush_batch()


    print("\n")


    elapsed = (
        time.time()
        - start_time
    )


    print(
        f"Inference finished in "
        f"{elapsed / 60:.2f} minutes."
    )


    return output_probability_path


# =============================================================================
# 11. PIXEL-LEVEL EVALUATION
# =============================================================================

def evaluate_pixels(
    probability_path,
    model_name,
    thresholds,
):

    records = []


    with rasterio.open(
        probability_path
    ) as src:


        block_height = 1024


        for threshold in (
            thresholds
        ):

            tp = 0
            fp = 0
            fn = 0
            tn = 0

            predicted_positive_pixels = 0


            for row_off in range(
                0,
                TEST_HEIGHT,
                block_height,
            ):

                h = min(
                    block_height,
                    TEST_HEIGHT
                    - row_off,
                )


                window = Window(
                    0,
                    row_off,
                    TEST_WIDTH,
                    h,
                )


                probability = src.read(
                    1,
                    window=window,
                )


                local_aoi = np.asarray(

                    aoi_mask[
                        row_off:
                        row_off
                        + h,
                        :
                    ],

                    dtype=bool,

                )


                truth = np.asarray(

                    reference_mask[
                        row_off:
                        row_off
                        + h,
                        :
                    ],

                    dtype=bool,

                )


                valid = (
                    local_aoi
                    & (
                        probability
                        != PROBABILITY_NODATA
                    )
                )


                prediction = (
                    probability
                    >= threshold
                )


                prediction = (
                    prediction
                    & valid
                )


                truth = (
                    truth
                    & valid
                )


                tp += int(
                    np.sum(
                        prediction
                        & truth
                    )
                )


                fp += int(
                    np.sum(
                        prediction
                        & ~truth
                        & valid
                    )
                )


                fn += int(
                    np.sum(
                        ~prediction
                        & truth
                        & valid
                    )
                )


                tn += int(
                    np.sum(
                        ~prediction
                        & ~truth
                        & valid
                    )
                )


                predicted_positive_pixels += int(
                    np.sum(
                        prediction
                    )
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
                )
                > 0
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
                )
                > 0
                else 0.0
            )


            f1 = f1_from_pr(
                precision,
                recall,
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
                )
                > 0
                else 0.0
            )


            accuracy = (
                (
                    tp
                    + tn
                )
                / (
                    tp
                    + fp
                    + fn
                    + tn
                )
                if (
                    tp
                    + fp
                    + fn
                    + tn
                )
                > 0
                else 0.0
            )


            predicted_area_m2 = (
                predicted_positive_pixels
                * PIXEL_AREA_M2
            )


            records.append(
                {
                    "model":
                        model_name,

                    "threshold":
                        threshold,

                    "tp_pixels":
                        tp,

                    "fp_pixels":
                        fp,

                    "fn_pixels":
                        fn,

                    "tn_pixels":
                        tn,

                    "precision":
                        precision,

                    "recall":
                        recall,

                    "f1":
                        f1,

                    "iou":
                        iou,

                    "accuracy":
                        accuracy,

                    "predicted_positive_pixels":
                        predicted_positive_pixels,

                    "predicted_pixel_area_m2":
                        predicted_area_m2,

                    "reference_pixel_area_m2":
                        REFERENCE_RASTER_AREA_M2,

                    "pixel_area_ratio_pred_ref":
                        (
                            predicted_area_m2
                            / REFERENCE_RASTER_AREA_M2
                            if REFERENCE_RASTER_AREA_M2
                            > 0
                            else np.nan
                        ),
                }
            )


    return pd.DataFrame(
        records
    )


# =============================================================================
# 12. THRESHOLD RASTER -> 8-CONNECTED OBJECT POLYGONS
# =============================================================================

def extract_prediction_objects(
    probability_path,
    threshold,
    model_name,
):

    code = threshold_code(
        threshold
    )


    binary_memmap_path = (
        TEMP_DIR
        / (
            f"{model_name}_"
            f"threshold_{code}.dat"
        )
    )


    binary = np.memmap(

        binary_memmap_path,

        dtype="uint8",

        mode="w+",

        shape=(
            TEST_HEIGHT,
            TEST_WIDTH,
        ),

    )


    binary[:] = 0


    with rasterio.open(
        probability_path
    ) as src:


        block_height = 1024


        for row_off in range(
            0,
            TEST_HEIGHT,
            block_height,
        ):

            h = min(
                block_height,
                TEST_HEIGHT
                - row_off,
            )


            probability = src.read(
                1,
                window=Window(
                    0,
                    row_off,
                    TEST_WIDTH,
                    h,
                ),
            )


            local_aoi = np.asarray(

                aoi_mask[
                    row_off:
                    row_off
                    + h,
                    :
                ],

                dtype=np.uint8,

            )


            valid = (
                local_aoi
                == 1
            ) & (
                probability
                != PROBABILITY_NODATA
            )


            binary[
                row_off:
                row_off
                + h,
                :
            ] = (

                (
                    probability
                    >= threshold
                )
                & valid

            ).astype(
                np.uint8
            )


    binary.flush()


    object_records = []


    object_id = 0


    for geometry_mapping, value in shapes(

        binary,

        mask=binary,

        transform=test_transform,

        connectivity=CONNECTIVITY,

    ):

        if int(
            value
        ) != 1:

            continue


        geometry = make_valid(
            shape(
                geometry_mapping
            )
        )


        if geometry.is_empty:

            continue


        area_m2 = float(
            geometry.area
        )


        if area_m2 < (
            MIN_OBJECT_AREA_M2
        ):

            continue


        object_id += 1


        object_records.append(
            {
                "pred_id":
                    object_id,

                "area_m2":
                    area_m2,

                "geometry":
                    geometry,
            }
        )


    del binary


    try:

        binary_memmap_path.unlink()

    except Exception:

        pass


    prediction_gdf = gpd.GeoDataFrame(

        object_records,

        geometry="geometry",

        crs=raster_crs,

    )


    if len(
        prediction_gdf
    ) > 0:

        prediction_gdf[
            "model"
        ] = model_name


        prediction_gdf[
            "threshold"
        ] = threshold


    return prediction_gdf


# =============================================================================
# 13. ONE-TO-ONE OBJECT MATCHING
# =============================================================================

def match_objects(
    prediction_gdf,
    reference_gdf,
):

    if len(
        prediction_gdf
    ) == 0:

        return (
            pd.DataFrame(),
            0,
        )


    reference_sindex = (
        reference_gdf.sindex
    )


    candidate_matches = []


    for pred_idx, pred_row in (
        prediction_gdf.iterrows()
    ):

        pred_geom = (
            pred_row.geometry
        )


        possible_ref_positions = list(

            reference_sindex.intersection(
                pred_geom.bounds
            )

        )


        for ref_position in (
            possible_ref_positions
        ):

            ref_row = reference_gdf.iloc[
                ref_position
            ]


            ref_geom = (
                ref_row.geometry
            )


            if not pred_geom.intersects(
                ref_geom
            ):

                continue


            intersection = pred_geom.intersection(
                ref_geom
            )


            intersection_area = float(
                intersection.area
            )


            if intersection_area <= 0:

                continue


            ref_area = float(
                ref_geom.area
            )


            pred_area = float(
                pred_geom.area
            )


            ref_coverage = (
                intersection_area
                / ref_area
                if ref_area > 0
                else 0.0
            )


            if (
                ref_coverage
                < MIN_REFERENCE_COVERAGE
            ):

                continue


            union_area = (
                pred_area
                + ref_area
                - intersection_area
            )


            iou = (
                intersection_area
                / union_area
                if union_area > 0
                else 0.0
            )


            pred_coverage = (
                intersection_area
                / pred_area
                if pred_area > 0
                else 0.0
            )


            candidate_matches.append(
                {
                    "pred_index":
                        pred_idx,

                    "pred_id":
                        int(
                            pred_row[
                                "pred_id"
                            ]
                        ),

                    "ref_position":
                        ref_position,

                    "ref_id":
                        int(
                            ref_row[
                                "ref_id"
                            ]
                        ),

                    "intersection_m2":
                        intersection_area,

                    "iou":
                        iou,

                    "reference_coverage":
                        ref_coverage,

                    "prediction_coverage":
                        pred_coverage,
                }
            )


    if not candidate_matches:

        return (
            pd.DataFrame(),
            0,
        )


    candidates = pd.DataFrame(
        candidate_matches
    )


    candidates = candidates.sort_values(

        by=[
            "iou",
            "reference_coverage",
            "intersection_m2",
        ],

        ascending=[
            False,
            False,
            False,
        ],

    ).reset_index(
        drop=True
    )


    matched_pred = set()

    matched_ref = set()

    accepted = []


    for _, row in (
        candidates.iterrows()
    ):

        pred_id = int(
            row[
                "pred_id"
            ]
        )


        ref_id = int(
            row[
                "ref_id"
            ]
        )


        if (
            pred_id
            in matched_pred
        ):

            continue


        if (
            ref_id
            in matched_ref
        ):

            continue


        accepted.append(
            row.to_dict()
        )


        matched_pred.add(
            pred_id
        )


        matched_ref.add(
            ref_id
        )


    accepted_df = pd.DataFrame(
        accepted
    )


    return (
        accepted_df,
        len(
            accepted_df
        ),
    )


# =============================================================================
# 14. OBJECT-LEVEL EVALUATION
# =============================================================================

def evaluate_objects(
    probability_path,
    model_name,
    thresholds,
):

    metric_records = []


    primary_prediction_path = (
        VECTOR_DIR
        / (
            f"{model_name}_"
            f"predictions_t"
            f"{threshold_code(PRIMARY_THRESHOLD)}.gpkg"
        )
    )


    primary_match_path = (
        RESULTS_DIR
        / (
            f"{model_name}_"
            f"object_matches_t"
            f"{threshold_code(PRIMARY_THRESHOLD)}.csv"
        )
    )


    for threshold in (
        thresholds
    ):

        print(
            f"\n{model_name}: "
            f"extracting objects at "
            f"threshold {threshold:.2f}..."
        )


        predictions = extract_prediction_objects(
            probability_path,
            threshold,
            model_name,
        )


        n_predictions = (
            len(
                predictions
            )
        )


        predicted_area_m2 = (
            float(
                predictions[
                    "area_m2"
                ].sum()
            )
            if n_predictions > 0
            else 0.0
        )


        matches, n_matches = match_objects(
            predictions,
            reference_units,
        )


        unmatched_predictions = (
            n_predictions
            - n_matches
        )


        missed_references = (
            N_REFERENCE_UNITS
            - n_matches
        )


        precision = (
            n_matches
            / n_predictions
            if n_predictions > 0
            else 0.0
        )


        recall = (
            n_matches
            / N_REFERENCE_UNITS
            if N_REFERENCE_UNITS > 0
            else 0.0
        )


        f1 = f1_from_pr(
            precision,
            recall,
        )


        metric_records.append(
            {
                "model":
                    model_name,

                "threshold":
                    threshold,

                "min_object_area_m2":
                    MIN_OBJECT_AREA_M2,

                "connectivity":
                    CONNECTIVITY,

                "reference_units":
                    N_REFERENCE_UNITS,

                "predicted_objects":
                    n_predictions,

                "matched_reference_units":
                    n_matches,

                "unmatched_predicted_objects":
                    unmatched_predictions,

                "missed_reference_units":
                    missed_references,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "predicted_object_area_m2":
                    predicted_area_m2,

                "reference_vector_area_m2":
                    REFERENCE_VECTOR_AREA_M2,

                "object_area_ratio_pred_ref":
                    (
                        predicted_area_m2
                        / REFERENCE_VECTOR_AREA_M2
                        if REFERENCE_VECTOR_AREA_M2
                        > 0
                        else np.nan
                    ),
            }
        )


        print(
            f"  predicted objects: "
            f"{n_predictions:,}"
        )


        print(
            f"  matched references: "
            f"{n_matches:,} / "
            f"{N_REFERENCE_UNITS:,}"
        )


        print(
            f"  precision: "
            f"{precision:.4f}"
        )


        print(
            f"  recall: "
            f"{recall:.4f}"
        )


        # Save only primary-threshold geometries to avoid many large files.
        if np.isclose(
            threshold,
            PRIMARY_THRESHOLD
        ):

            if n_predictions > 0:

                predictions.to_file(
                    primary_prediction_path,
                    layer="predictions",
                    driver="GPKG",
                )


            matches.to_csv(
                primary_match_path,
                index=False,
            )


    return pd.DataFrame(
        metric_records
    )


# =============================================================================
# 15. RUN ALL THREE MODELS
# =============================================================================

models = [
    {
        "name":
            "baseline_march",

        "path":
            BASELINE_MODEL_PATH,

        "probability_path":
            PROBABILITY_DIR
            / "baseline_march_waste_probability.tif",
    },

    {
        "name":
            "complementary_refined",

        "path":
            COMPLEMENTARY_MODEL_PATH,

        "probability_path":
            PROBABILITY_DIR
            / "complementary_refined_waste_probability.tif",
    },

    {
        "name":
            "error_directed_refined",

        "path":
            ERROR_DIRECTED_MODEL_PATH,

        "probability_path":
            PROBABILITY_DIR
            / "error_directed_refined_waste_probability.tif",
    },
]


all_pixel_metrics = []

all_object_metrics = []


for model_info in (
    models
):

    probability_path = run_seam_reduced_inference(

        model_path=model_info[
            "path"
        ],

        model_name=model_info[
            "name"
        ],

        output_probability_path=model_info[
            "probability_path"
        ],

    )


    pixel_metrics = evaluate_pixels(
        probability_path,
        model_info[
            "name"
        ],
        THRESHOLDS,
    )


    object_metrics = evaluate_objects(
        probability_path,
        model_info[
            "name"
        ],
        THRESHOLDS,
    )


    all_pixel_metrics.append(
        pixel_metrics
    )


    all_object_metrics.append(
        object_metrics
    )


# =============================================================================
# 16. COMBINE / SAVE RESULTS
# =============================================================================

pixel_metrics_df = pd.concat(
    all_pixel_metrics,
    ignore_index=True,
)


object_metrics_df = pd.concat(
    all_object_metrics,
    ignore_index=True,
)


PIXEL_METRICS_PATH = (
    RESULTS_DIR
    / "pixel_metrics_all_thresholds.csv"
)


OBJECT_METRICS_PATH = (
    RESULTS_DIR
    / "object_metrics_all_thresholds.csv"
)


pixel_metrics_df.to_csv(
    PIXEL_METRICS_PATH,
    index=False,
)


object_metrics_df.to_csv(
    OBJECT_METRICS_PATH,
    index=False,
)


# =============================================================================
# 17. PRIMARY 0.50 COMPARISON TABLE
# =============================================================================

primary_pixel = pixel_metrics_df[
    np.isclose(
        pixel_metrics_df[
            "threshold"
        ],
        PRIMARY_THRESHOLD,
    )
].copy()


primary_object = object_metrics_df[
    np.isclose(
        object_metrics_df[
            "threshold"
        ],
        PRIMARY_THRESHOLD,
    )
].copy()


primary = primary_pixel.merge(

    primary_object,

    on=[
        "model",
        "threshold",
    ],

    suffixes=(
        "_pixel",
        "_object",
    ),

)


PRIMARY_COMPARISON_PATH = (
    RESULTS_DIR
    / "primary_comparison_threshold_0.50.csv"
)


primary.to_csv(
    PRIMARY_COMPARISON_PATH,
    index=False,
)


# -----------------------------------------------------------------------------
# Wide comparison: baseline and both refined models
# -----------------------------------------------------------------------------

wide_records = []


for model_name in [
    "baseline_march",
    "complementary_refined",
    "error_directed_refined",
]:

    p = primary[
        primary[
            "model"
        ]
        == model_name
    ].iloc[
        0
    ]


    wide_records.append(
        {
            "model":
                model_name,

            "pixel_precision":
                p[
                    "precision_pixel"
                ],

            "pixel_recall":
                p[
                    "recall_pixel"
                ],

            "pixel_f1":
                p[
                    "f1_pixel"
                ],

            "pixel_iou":
                p[
                    "iou"
                ],

            "predicted_objects":
                int(
                    p[
                        "predicted_objects"
                    ]
                ),

            "matched_reference_units":
                int(
                    p[
                        "matched_reference_units"
                    ]
                ),

            "unmatched_predicted_objects":
                int(
                    p[
                        "unmatched_predicted_objects"
                    ]
                ),

            "missed_reference_units":
                int(
                    p[
                        "missed_reference_units"
                    ]
                ),

            "object_precision":
                p[
                    "precision_object"
                ],

            "object_recall":
                p[
                    "recall_object"
                ],

            "object_f1":
                p[
                    "f1_object"
                ],

            "predicted_object_area_m2":
                p[
                    "predicted_object_area_m2"
                ],

            "reference_vector_area_m2":
                p[
                    "reference_vector_area_m2"
                ],

            "object_area_ratio_pred_ref":
                p[
                    "object_area_ratio_pred_ref"
                ],
        }
    )


primary_compact = pd.DataFrame(
    wide_records
)


PRIMARY_COMPACT_PATH = (
    RESULTS_DIR
    / "primary_comparison_compact.csv"
)


primary_compact.to_csv(
    PRIMARY_COMPACT_PATH,
    index=False,
)


# =============================================================================
# 18. CHANGE FROM BASELINE AT PRIMARY THRESHOLD
# =============================================================================

baseline_row = primary_compact[
    primary_compact[
        "model"
    ]
    == "baseline_march"
].iloc[
    0
]


change_metrics = [
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pixel_iou",
    "predicted_objects",
    "matched_reference_units",
    "unmatched_predicted_objects",
    "missed_reference_units",
    "object_precision",
    "object_recall",
    "object_f1",
    "predicted_object_area_m2",
    "object_area_ratio_pred_ref",
]


change_records = []


for comparison_model in [
    "complementary_refined",
    "error_directed_refined",
]:

    comparison_row = primary_compact[
        primary_compact[
            "model"
        ]
        == comparison_model
    ].iloc[
        0
    ]


    for metric in (
        change_metrics
    ):

        baseline_value = float(
            baseline_row[
                metric
            ]
        )


        comparison_value = float(
            comparison_row[
                metric
            ]
        )


        absolute_change = (
            comparison_value
            - baseline_value
        )


        relative_change_percent = (

            100.0
            * absolute_change
            / baseline_value

            if baseline_value
            != 0

            else np.nan

        )


        change_records.append(
            {
                "comparison_model":
                    comparison_model,

                "metric":
                    metric,

                "baseline":
                    baseline_value,

                "comparison":
                    comparison_value,

                "absolute_change":
                    absolute_change,

                "relative_change_percent":
                    relative_change_percent,
            }
        )


change_df = pd.DataFrame(
    change_records
)


CHANGE_PATH = (
    RESULTS_DIR
    / "change_from_baseline_threshold_0.50.csv"
)


change_df.to_csv(
    CHANGE_PATH,
    index=False,
)


# =============================================================================
# 19. RUN METADATA
# =============================================================================

metadata = {
    "seed":
        SEED,

    "orthomosaic":
        str(
            ORTHO_PATH
        ),

    "test_aoi":
        str(
            TEST_AOI_PATH
        ),

    "reference_waste":
        str(
            REFERENCE_WASTE_PATH
        ),

    "baseline_model":
        str(
            BASELINE_MODEL_PATH
        ),

    "baseline_model_sha256":
        sha256_file(
            BASELINE_MODEL_PATH
        ),

    "complementary_model":
        str(
            COMPLEMENTARY_MODEL_PATH
        ),

    "complementary_model_sha256":
        sha256_file(
            COMPLEMENTARY_MODEL_PATH
        ),

    "error_directed_model":
        str(
            ERROR_DIRECTED_MODEL_PATH
        ),

    "error_directed_model_sha256":
        sha256_file(
            ERROR_DIRECTED_MODEL_PATH
        ),

    "model_tile":
        MODEL_TILE,

    "retained_core":
        CORE_SIZE,

    "halo":
        HALO,

    "waste_class_index":
        WASTE_CLASS_IDX,

    "primary_threshold":
        PRIMARY_THRESHOLD,

    "descriptive_thresholds":
        THRESHOLDS,

    "minimum_mapping_unit_m2":
        MIN_OBJECT_AREA_M2,

    "connectivity":
        CONNECTIVITY,

    "minimum_reference_coverage_for_match":
        MIN_REFERENCE_COVERAGE,

    "reference_source_features":
        REFERENCE_SOURCE_FEATURES,

    "reference_contiguous_units":
        N_REFERENCE_UNITS,

    "reference_vector_area_m2":
        REFERENCE_VECTOR_AREA_M2,

    "reference_raster_area_m2":
        REFERENCE_RASTER_AREA_M2,

    "test_width_pixels":
        TEST_WIDTH,

    "test_height_pixels":
        TEST_HEIGHT,

    "test_aoi_pixels":
        N_AOI_PIXELS,

    "pixel_area_m2":
        PIXEL_AREA_M2,
}


METADATA_PATH = (
    RESULTS_DIR
    / "run_metadata.json"
)


with open(
    METADATA_PATH,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# =============================================================================
# 20. CONSOLE SUMMARY
# =============================================================================

print("\n")
print("=" * 80)
print("PRIMARY COMPARISON: FIXED THRESHOLD = 0.50")
print("=" * 80)


display_columns = [
    "model",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pixel_iou",
    "predicted_objects",
    "matched_reference_units",
    "unmatched_predicted_objects",
    "missed_reference_units",
    "object_precision",
    "object_recall",
    "object_f1",
    "predicted_object_area_m2",
    "reference_vector_area_m2",
    "object_area_ratio_pred_ref",
]


print(
    primary_compact[
        display_columns
    ].to_string(
        index=False
    )
)


print("\n")
print("=" * 80)
print("CHANGE FROM BASELINE")
print("=" * 80)


print(
    change_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 80)
print("OUTPUTS")
print("=" * 80)


print(
    f"\nPrimary compact comparison:\n"
    f"{PRIMARY_COMPACT_PATH}"
)


print(
    f"\nPrimary full comparison:\n"
    f"{PRIMARY_COMPARISON_PATH}"
)


print(
    f"\nChange from baseline:\n"
    f"{CHANGE_PATH}"
)


print(
    f"\nPixel threshold sensitivity:\n"
    f"{PIXEL_METRICS_PATH}"
)


print(
    f"\nObject threshold sensitivity:\n"
    f"{OBJECT_METRICS_PATH}"
)


print(
    f"\nProbability rasters:\n"
    f"{PROBABILITY_DIR}"
)


print(
    f"\nPrimary-threshold vectors:\n"
    f"{VECTOR_DIR}"
)


print(
    f"\nRun metadata:\n"
    f"{METADATA_PATH}"
)


print("\n")
print(
    "IMPORTANT: thresholds above 0.50 are descriptive sensitivity analyses "
    "only. Do not select a new operating threshold from the independent test."
)


# =============================================================================
# 21. CLEAN TEMPORARY MEMMAPS
# =============================================================================

try:

    del reference_mask

except Exception:

    pass


try:

    del aoi_mask

except Exception:

    pass


for temporary_path in [
    REFERENCE_MASK_PATH,
    AOI_MASK_PATH,
]:

    try:

        temporary_path.unlink()

    except Exception:

        pass


try:

    if TEMP_DIR.exists() and not any(
        TEMP_DIR.iterdir()
    ):

        TEMP_DIR.rmdir()

except Exception:

    pass


print("\nDone.")
