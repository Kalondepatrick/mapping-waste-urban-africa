# =============================================================================
# CLIP 18 CONFIRMED FALSE-POSITIVE AREAS,
# RUN MULTICLASS U-NET, AND VECTORISE PREDICTIONS
#
# INPUTS
# -------
# Full orthomosaic:
# C:/Users/Peter/Documents/GitHub/mapping-waste-urban-africa/data/optical.tiff
#
# Confirmed false-positive inspection areas:
# results/hard_example_object_sampling/confirmed_false_positives.gpkg
#
# Multiclass U-Net:
# models/filters_16_march_finetuned_best.keras
#
#
# OUTPUTS
# -------
# 1. Individual RGB image clips for the 18 areas
# 2. Multiclass prediction raster for each area
# 3. Model-confidence raster for each area
# 4. Vectorised multiclass prediction for each area
# 5. One combined editable GeoPackage containing all 18 areas
#
#
# IMPORTANT
# ---------
# The vector polygons are MODEL PREDICTIONS, not truth.
# They are intended to provide an editable starting point for manual
# correction against the original RGB imagery.
# =============================================================================


from pathlib import Path
import gc

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask, shapes

from shapely.geometry import shape, mapping
from shapely.ops import unary_union

import tensorflow as tf
from tensorflow.keras.models import load_model


# =============================================================================
# 1. PATHS
# =============================================================================

BASE_PATH = (
    Path.home()
    / "Documents"
    / "GitHub"
    / "mapping-waste-urban-africa"
)


# Full original orthomosaic
ORTHOMOSAIC_PATH = (
    BASE_PATH
    / "data"
    / "optical.tiff"
)


# Confirmed false-positive areas
FALSE_POSITIVE_PATH = (
    BASE_PATH
    / "results"
    / "hard_example_object_sampling"
    / "confirmed_false_positives.gpkg"
)


# Multiclass U-Net
MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_march_finetuned_best.keras"
)


# Main output directory
OUTPUT_DIR = (
    BASE_PATH
    / "results"
    / "confirmed_false_positive_annotation"
)


RGB_DIR = (
    OUTPUT_DIR
    / "01_rgb_clips"
)


CLASS_RASTER_DIR = (
    OUTPUT_DIR
    / "02_multiclass_rasters"
)


CONFIDENCE_DIR = (
    OUTPUT_DIR
    / "03_confidence_rasters"
)


VECTOR_DIR = (
    OUTPUT_DIR
    / "04_vector_predictions"
)


for folder in [
    OUTPUT_DIR,
    RGB_DIR,
    CLASS_RASTER_DIR,
    CONFIDENCE_DIR,
    VECTOR_DIR
]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# =============================================================================
# 2. CLASS DEFINITIONS
# =============================================================================

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

    10: "Unidentified"
}


N_CLASSES = 10


# =============================================================================
# 3. MODEL INFERENCE SETTINGS
# =============================================================================

# Model input
INPUT_SIZE = 256


# Retain only centre 128 x 128 pixels from each prediction
OUTPUT_SIZE = 128


MARGIN = (
    INPUT_SIZE
    - OUTPUT_SIZE
) // 2


BATCH_SIZE = 16


# =============================================================================
# 4. VECTORISATION SETTINGS
# =============================================================================

# Keep all contiguous predicted regions initially.
#
# Because these vectors will be manually edited, I would NOT impose
# an arbitrary minimum polygon size at this stage.
MIN_VECTOR_AREA_M2 = 0.0


# =============================================================================
# 5. CHECK INPUT FILES
# =============================================================================

print("\n")
print("=" * 80)
print("CHECKING INPUTS")
print("=" * 80)


for path in [
    ORTHOMOSAIC_PATH,
    FALSE_POSITIVE_PATH,
    MODEL_PATH
]:

    if not path.exists():

        raise FileNotFoundError(
            f"\nInput not found:\n{path}"
        )


print(
    f"\nOrthomosaic:\n"
    f"{ORTHOMOSAIC_PATH}"
)

print(
    f"\nFalse-positive areas:\n"
    f"{FALSE_POSITIVE_PATH}"
)

print(
    f"\nModel:\n"
    f"{MODEL_PATH}"
)


# =============================================================================
# 6. LOAD FALSE-POSITIVE AREAS
# =============================================================================

print("\n")
print("=" * 80)
print("LOADING CONFIRMED FALSE-POSITIVE AREAS")
print("=" * 80)


areas = gpd.read_file(
    FALSE_POSITIVE_PATH
)


areas = areas[
    areas.geometry.notna()
    & (~areas.geometry.is_empty)
].copy()


if areas.empty:

    raise ValueError(
        "No valid polygons were found."
    )


# Safety check:
# if fp field still exists, retain only fp == 1.
if "fp" in areas.columns:

    areas["fp"] = pd.to_numeric(
        areas["fp"],
        errors="coerce"
    )


    areas = areas[
        areas["fp"] == 1
    ].copy()


print(
    f"Confirmed false-positive areas: "
    f"{len(areas)}"
)


if len(areas) != 18:

    print(
        "\nWARNING:"
        f" Expected 18 areas but found {len(areas)}."
    )


# =============================================================================
# 7. LOAD MODEL
# =============================================================================

print("\n")
print("=" * 80)
print("LOADING MULTICLASS U-NET")
print("=" * 80)


tf.keras.backend.clear_session()


MODEL = load_model(
    MODEL_PATH,
    compile=False
)


print(
    f"Input shape: "
    f"{MODEL.input_shape}"
)


print(
    f"Output shape: "
    f"{MODEL.output_shape}"
)


model_n_classes = int(
    MODEL.output_shape[-1]
)


if model_n_classes != N_CLASSES:

    raise ValueError(
        "\nThis does not appear to be the expected "
        "10-class U-Net.\n"
        f"Model output classes: {model_n_classes}"
    )


# =============================================================================
# 8. IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(tile):

    """
    Convert rasterio format:

        bands x rows x cols

    to:

        rows x cols x bands

    and apply the same 0-1 scaling used in training.
    """

    tile = np.moveaxis(
        tile,
        0,
        -1
    )


    tile = tile.astype(
        np.float32
    )


    tile /= 255.0


    return tile


# =============================================================================
# 9. BATCH PREDICTION FUNCTION
# =============================================================================

def process_batch(
    tiles,
    metadata,
    class_output,
    confidence_output
):

    if len(tiles) == 0:
        return


    batch = np.stack(
        tiles,
        axis=0
    ).astype(
        np.float32
    )


    probabilities = MODEL.predict(
        batch,
        verbose=0
    )


    # -------------------------------------------------------------------------
    # Winning model channel
    # -------------------------------------------------------------------------

    model_class = np.argmax(
        probabilities,
        axis=-1
    )


    # Highest softmax probability
    model_confidence = np.max(
        probabilities,
        axis=-1
    )


    # -------------------------------------------------------------------------
    # Convert model channels:
    #
    # 0 -> raw label 1
    # 1 -> raw label 2
    # ...
    # 9 -> raw label 10
    # -------------------------------------------------------------------------

    raw_class = (
        model_class
        + 1
    ).astype(
        np.uint8
    )


    for (
        class_tile,
        confidence_tile,
        info
    ) in zip(

        raw_class,
        model_confidence,
        metadata

    ):

        (
            out_row,
            out_col,
            write_height,
            write_width
        ) = info


        # ---------------------------------------------------------------------
        # Retain centre of prediction only
        # ---------------------------------------------------------------------

        centre_class = class_tile[
            MARGIN:
            MARGIN + OUTPUT_SIZE,

            MARGIN:
            MARGIN + OUTPUT_SIZE
        ]


        centre_confidence = confidence_tile[
            MARGIN:
            MARGIN + OUTPUT_SIZE,

            MARGIN:
            MARGIN + OUTPUT_SIZE
        ]


        centre_class = centre_class[
            :write_height,
            :write_width
        ]


        centre_confidence = centre_confidence[
            :write_height,
            :write_width
        ]


        class_output[
            out_row:
            out_row + write_height,

            out_col:
            out_col + write_width
        ] = centre_class


        confidence_output[
            out_row:
            out_row + write_height,

            out_col:
            out_col + write_width
        ] = centre_confidence


# =============================================================================
# 10. OPEN FULL ORTHOMOSAIC
# =============================================================================

src = rasterio.open(
    ORTHOMOSAIC_PATH
)


if src.crs is None:

    raise ValueError(
        "Orthomosaic does not have a CRS."
    )


areas = areas.to_crs(
    src.crs
)


pixel_width_m = abs(
    src.transform.a
)


pixel_height_m = abs(
    src.transform.e
)


pixel_area_m2 = (
    pixel_width_m
    * pixel_height_m
)


print("\n")
print("=" * 80)
print("ORTHOMOSAIC INFORMATION")
print("=" * 80)


print(
    f"CRS: {src.crs}"
)


print(
    f"Dimensions: "
    f"{src.width:,} x "
    f"{src.height:,} pixels"
)


print(
    f"Pixel size: "
    f"{pixel_width_m:.4f} x "
    f"{pixel_height_m:.4f} m"
)


# =============================================================================
# 11. STORAGE FOR COMBINED OUTPUT
# =============================================================================

all_vector_records = []

class_summary_records = []


# =============================================================================
# 12. PROCESS EACH OF THE 18 AREAS
# =============================================================================

print("\n")
print("=" * 80)
print("PROCESSING FALSE-POSITIVE AREAS")
print("=" * 80)


for counter, (_, area) in enumerate(
    areas.iterrows(),
    start=1
):


    # -------------------------------------------------------------------------
    # IDs
    # -------------------------------------------------------------------------

    if "sample_id" in areas.columns:

        sample_id = int(
            area["sample_id"]
        )

    else:

        sample_id = counter


    if "object_id" in areas.columns:

        object_id = int(
            area["object_id"]
        )

    else:

        object_id = -1


    geometry = area.geometry


    print("\n")
    print("-" * 80)

    print(
        f"Processing {counter}/{len(areas)}"
    )

    print(
        f"Sample ID: {sample_id}"
    )

    print(
        f"Object ID: {object_id}"
    )


    # =========================================================================
    # 12.1 DETERMINE IMAGE WINDOW
    # =========================================================================

    minx, miny, maxx, maxy = (
        geometry.bounds
    )


    clip_window = rasterio.windows.from_bounds(

        minx,
        miny,
        maxx,
        maxy,

        transform=src.transform

    )


    clip_window = (
        clip_window
        .round_offsets()
        .round_lengths()
    )


    # Clip against whole raster
    clip_window = clip_window.intersection(

        Window(
            0,
            0,
            src.width,
            src.height
        )

    )


    clip_width = int(
        clip_window.width
    )


    clip_height = int(
        clip_window.height
    )


    clip_transform = (
        src.window_transform(
            clip_window
        )
    )


    print(
        f"Image clip: "
        f"{clip_width:,} x "
        f"{clip_height:,} pixels"
    )


    # =========================================================================
    # 12.2 READ RGB CLIP
    # =========================================================================

    rgb = src.read(
        [1, 2, 3],
        window=clip_window
    )


    # =========================================================================
    # 12.3 SAVE CLIPPED ORIGINAL IMAGERY
    # =========================================================================

    rgb_profile = (
        src.profile.copy()
    )


    rgb_profile.update(
        {

            "height":
                clip_height,

            "width":
                clip_width,

            "transform":
                clip_transform,

            "count":
                3,

            "compress":
                "lzw",

            "BIGTIFF":
                "IF_SAFER"

        }
    )


    rgb_output = (
        RGB_DIR
        / f"sample_{sample_id:03d}_rgb.tif"
    )


    with rasterio.open(
        rgb_output,
        "w",
        **rgb_profile
    ) as dst:

        dst.write(
            rgb
        )


    # =========================================================================
    # 12.4 OUTPUT ARRAYS
    # =========================================================================

    predicted_class = np.zeros(

        (
            clip_height,
            clip_width
        ),

        dtype=np.uint8

    )


    confidence = np.full(

        (
            clip_height,
            clip_width
        ),

        -1.0,

        dtype=np.float32

    )


    # =========================================================================
    # 12.5 RUN U-NET
    #
    # Use image context from the FULL orthomosaic rather than from the clipped
    # image itself. Therefore objects near the boundary of an inspection area
    # still receive surrounding context.
    # =========================================================================

    batch_tiles = []

    batch_metadata = []


    global_clip_row = int(
        clip_window.row_off
    )


    global_clip_col = int(
        clip_window.col_off
    )


    for local_row in range(
        0,
        clip_height,
        OUTPUT_SIZE
    ):

        write_height = min(
            OUTPUT_SIZE,
            clip_height - local_row
        )


        for local_col in range(
            0,
            clip_width,
            OUTPUT_SIZE
        ):

            write_width = min(
                OUTPUT_SIZE,
                clip_width - local_col
            )


            global_row = (
                global_clip_row
                + local_row
            )


            global_col = (
                global_clip_col
                + local_col
            )


            # -----------------------------------------------------------------
            # 256x256 model input centred around the output region
            # -----------------------------------------------------------------

            input_window = Window(

                global_col
                - MARGIN,

                global_row
                - MARGIN,

                INPUT_SIZE,

                INPUT_SIZE

            )


            tile = src.read(

                [1, 2, 3],

                window=input_window,

                boundless=True,

                fill_value=0

            )


            if tile.shape != (
                3,
                INPUT_SIZE,
                INPUT_SIZE
            ):

                raise RuntimeError(
                    f"Unexpected tile shape: "
                    f"{tile.shape}"
                )


            tile = preprocess_image(
                tile
            )


            batch_tiles.append(
                tile
            )


            batch_metadata.append(
                (
                    local_row,
                    local_col,
                    write_height,
                    write_width
                )
            )


            if len(
                batch_tiles
            ) >= BATCH_SIZE:

                process_batch(

                    batch_tiles,
                    batch_metadata,
                    predicted_class,
                    confidence

                )


                batch_tiles = []
                batch_metadata = []


    # Process remaining tiles
    if batch_tiles:

        process_batch(

            batch_tiles,
            batch_metadata,
            predicted_class,
            confidence

        )


    # =========================================================================
    # 12.6 MASK TO EXACT FALSE-POSITIVE INSPECTION POLYGON
    # =========================================================================

    inside_area = geometry_mask(

        [
            mapping(
                geometry
            )
        ],

        out_shape=(
            clip_height,
            clip_width
        ),

        transform=clip_transform,

        invert=True

    )


    # Class 0 = outside the manually selected inspection area
    predicted_class[
        ~inside_area
    ] = 0


    confidence[
        ~inside_area
    ] = -1.0


    # =========================================================================
    # 12.7 SAVE MULTICLASS RASTER
    # =========================================================================

    class_profile = (
        src.profile.copy()
    )


    class_profile.update(
        {

            "height":
                clip_height,

            "width":
                clip_width,

            "transform":
                clip_transform,

            "count":
                1,

            "dtype":
                "uint8",

            "nodata":
                0,

            "compress":
                "lzw",

            "BIGTIFF":
                "IF_SAFER"

        }
    )


    class_output = (

        CLASS_RASTER_DIR
        / f"sample_{sample_id:03d}_classes.tif"

    )


    with rasterio.open(
        class_output,
        "w",
        **class_profile
    ) as dst:

        dst.write(
            predicted_class,
            1
        )


    # =========================================================================
    # 12.8 SAVE CONFIDENCE RASTER
    # =========================================================================

    confidence_profile = (
        class_profile.copy()
    )


    confidence_profile.update(
        {

            "dtype":
                "float32",

            "nodata":
                -1.0

        }
    )


    confidence_output = (

        CONFIDENCE_DIR
        / f"sample_{sample_id:03d}_confidence.tif"

    )


    with rasterio.open(
        confidence_output,
        "w",
        **confidence_profile
    ) as dst:

        dst.write(
            confidence,
            1
        )


    # =========================================================================
    # 12.9 SUMMARISE PREDICTED CLASSES
    # =========================================================================

    print(
        "Predicted composition:"
    )


    for raw_label in range(
        1,
        N_CLASSES + 1
    ):

        class_mask = (
            predicted_class
            == raw_label
        )


        n_pixels = int(
            class_mask.sum()
        )


        if n_pixels == 0:
            continue


        area_m2 = (
            n_pixels
            * pixel_area_m2
        )


        mean_confidence = float(
            confidence[
                class_mask
            ].mean()
        )


        print(
            f"  {raw_label:2d} "
            f"{CLASS_NAMES[raw_label]:35s} "
            f"{area_m2:8.2f} m²   "
            f"confidence={mean_confidence:.3f}"
        )


        class_summary_records.append(
            {

                "sample_id":
                    sample_id,

                "object_id":
                    object_id,

                "predicted_label":
                    raw_label,

                "predicted_class":
                    CLASS_NAMES[
                        raw_label
                    ],

                "pixel_count":
                    n_pixels,

                "area_m2":
                    area_m2,

                "mean_confidence":
                    mean_confidence

            }
        )


    # =========================================================================
    # 12.10 VECTORISE MODEL CLASSES
    # =========================================================================

    print(
        "Vectorising prediction..."
    )


    sample_records = []


    valid_vector_mask = (
        inside_area
        & (
            predicted_class
            > 0
        )
    )


    for (
        geom_json,
        value
    ) in shapes(

        predicted_class.astype(
            np.int16
        ),

        mask=valid_vector_mask,

        transform=clip_transform,

        connectivity=8

    ):

        raw_label = int(
            value
        )


        if raw_label not in CLASS_NAMES:

            continue


        polygon = shape(
            geom_json
        )


        # Clip back to exact selected inspection polygon
        polygon = polygon.intersection(
            geometry
        )


        if polygon.is_empty:

            continue


        area_m2 = polygon.area


        if area_m2 < MIN_VECTOR_AREA_M2:

            continue


        # ---------------------------------------------------------------------
        # Mean confidence for this polygon
        #
        # Rasterise polygon to local clip coordinates
        # ---------------------------------------------------------------------

        polygon_mask = geometry_mask(

            [
                mapping(
                    polygon
                )
            ],

            out_shape=confidence.shape,

            transform=clip_transform,

            invert=True

        )


        probability_values = confidence[
            polygon_mask
            & (
                confidence >= 0
            )
        ]


        if len(
            probability_values
        ) > 0:

            mean_probability = float(
                np.mean(
                    probability_values
                )
            )

        else:

            mean_probability = np.nan


        record = {

            "sample_id":
                sample_id,

            "object_id":
                object_id,

            # -------------------------------------------------------------
            # MODEL PREDICTION
            # -------------------------------------------------------------

            "pred_label":
                raw_label,

            "pred_class":
                CLASS_NAMES[
                    raw_label
                ],

            "pred_conf":
                mean_probability,

            "area_m2":
                area_m2,


            # -------------------------------------------------------------
            # MANUAL EDITING FIELDS
            #
            # Fill these in QGIS.
            # -------------------------------------------------------------

            "manual_lbl":
                None,

            "manual_cls":
                None,

            "keep":
                None,

            "notes":
                None,

            "geometry":
                polygon

        }


        sample_records.append(
            record
        )


        all_vector_records.append(
            record.copy()
        )


    # =========================================================================
    # 12.11 SAVE INDIVIDUAL VECTOR FILE
    # =========================================================================

    if sample_records:

        sample_vectors = (
            gpd.GeoDataFrame(

                sample_records,

                geometry="geometry",

                crs=src.crs

            )
        )


        sample_vector_output = (

            VECTOR_DIR
            / f"sample_{sample_id:03d}_prediction.gpkg"

        )


        sample_vectors.to_file(

            sample_vector_output,

            layer="multiclass_prediction",

            driver="GPKG"

        )


        print(
            f"Vector regions created: "
            f"{len(sample_vectors):,}"
        )


    # =========================================================================
    # 12.12 MEMORY CLEANUP
    # =========================================================================

    del rgb
    del predicted_class
    del confidence
    del inside_area

    gc.collect()


# =============================================================================
# 13. CLOSE ORTHOMOSAIC
# =============================================================================

src.close()


# =============================================================================
# 14. CREATE ONE COMBINED EDITABLE VECTOR FILE
# =============================================================================

print("\n")
print("=" * 80)
print("CREATING COMBINED EDITABLE VECTOR DATASET")
print("=" * 80)


if not all_vector_records:

    raise RuntimeError(
        "No multiclass polygons were produced."
    )


all_vectors = gpd.GeoDataFrame(

    all_vector_records,

    geometry="geometry",

    crs=areas.crs

)


# =============================================================================
# 15. REMOVE INVALID / EMPTY GEOMETRIES
# =============================================================================

all_vectors = all_vectors[
    all_vectors.geometry.notna()
    & (~all_vectors.geometry.is_empty)
].copy()


# =============================================================================
# 16. ASSIGN UNIQUE POLYGON ID
# =============================================================================

all_vectors.insert(

    0,

    "polygon_id",

    np.arange(
        1,
        len(all_vectors) + 1
    )

)


# =============================================================================
# 17. SAVE COMBINED EDITABLE GPKG
# =============================================================================

combined_output = (

    OUTPUT_DIR
    / "confirmed_false_positive_multiclass_editable.gpkg"

)


all_vectors.to_file(

    combined_output,

    layer="multiclass_editable",

    driver="GPKG"

)


# =============================================================================
# 18. SAVE ORIGINAL FALSE-POSITIVE AREAS INTO SAME GPKG
# =============================================================================

areas.to_file(

    combined_output,

    layer="inspection_areas",

    driver="GPKG",

    mode="a"

)


# =============================================================================
# 19. SAVE CLASS LOOKUP TABLE
# =============================================================================

class_lookup = pd.DataFrame(

    [

        {

            "label":
                label,

            "class_name":
                class_name

        }

        for (
            label,
            class_name
        ) in CLASS_NAMES.items()

    ]

)


class_lookup.to_csv(

    OUTPUT_DIR
    / "class_lookup.csv",

    index=False

)


# =============================================================================
# 20. SAVE CLASS SUMMARY
# =============================================================================

class_summary = pd.DataFrame(
    class_summary_records
)


class_summary.to_csv(

    OUTPUT_DIR
    / "predicted_class_summary.csv",

    index=False

)


# =============================================================================
# 21. SAVE MANUAL CLASS LOOKUP
#
# Useful while editing in QGIS.
# =============================================================================

manual_lookup = pd.DataFrame(

    {

        "label": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10
        ],

        "class_name": [
            "Waste",
            "Soil",
            "Vegetation",
            "Built-up surface",
            "Built-up surface (paved area)",
            "Water Body",
            "Natural Surface - rocks",
            "Non-persistent object",
            "Ancillary class",
            "Unidentified"
        ]

    }

)


manual_lookup.to_csv(

    OUTPUT_DIR
    / "manual_class_lookup.csv",

    index=False

)


# =============================================================================
# 22. FINAL REPORT
# =============================================================================

print("\n")
print("=" * 80)
print("COMPLETE")
print("=" * 80)


print(
    f"\nFalse-positive areas processed: "
    f"{len(areas)}"
)


print(
    f"Total editable polygons: "
    f"{len(all_vectors):,}"
)


print(
    "\nRGB image clips:"
)

print(
    RGB_DIR
)


print(
    "\nMulticlass rasters:"
)

print(
    CLASS_RASTER_DIR
)


print(
    "\nConfidence rasters:"
)

print(
    CONFIDENCE_DIR
)


print(
    "\nIndividual vectors:"
)

print(
    VECTOR_DIR
)


print(
    "\nMAIN QGIS EDITING FILE:"
)

print(
    combined_output
)


print("\n")
print("Class coding:")


for label, name in CLASS_NAMES.items():

    print(
        f"  {label:2d} = {name}"
    )


print("\nDone.")