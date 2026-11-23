import csv
import math
import os
import warnings
from pathlib import Path

import fiona
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import shape, mapping


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa"
)

INPUT_RASTER = (
    PROJECT_DIR
    / "data"
    / "otherdata"
    / "subset_March_2024.tif"
)

MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "filters_16_best.keras"
)

THRESHOLD_CSV = (
    PROJECT_DIR
    / "results"
    / "filters_16_waste_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "whole_area_prediction"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# U-Net input size used during training.
TILE_SIZE = 256

# Predict on 256 x 256 tiles but retain only the central region.
# This reduces tile-edge artefacts.
HALO = 32
CORE_SIZE = TILE_SIZE - (2 * HALO)  # 192 pixels

RGB_BANDS = [1, 2, 3]

# Waste is raw raster label 1.
# During model training raw class 1 was mapped to model output channel 0.
WASTE_RAW_LABEL = 1
WASTE_CLASS_IDX = 0

# If None, read the validation-derived threshold from THRESHOLD_CSV.
# Set manually, e.g. 0.50, if you want to override the CSV value.
WASTE_THRESHOLD = None

# Remove predicted components containing 1-5 pixels.
# Therefore only objects with >5 pixels are retained.
MIN_PIXELS = 5

# Optional physical-area filter in square metres.
# Keep at 0.0 for the first pass.
MIN_AREA_M2 = 0.0

# Prediction batch size.
# Reduce if memory is limited.
BATCH_SIZE = 8

# Output products.
PROBABILITY_RASTER = OUTPUT_DIR / "waste_probability.tif"
BINARY_RASTER = OUTPUT_DIR / "waste_prediction_binary.tif"
ALL_VECTOR = OUTPUT_DIR / "waste_piles_all.gpkg"
FILTERED_VECTOR = OUTPUT_DIR / "waste_piles_filtered.gpkg"
OBJECT_CSV = OUTPUT_DIR / "waste_piles_all.csv"


# ============================================================
# UTILITIES
# ============================================================

def safe_delete(path):
    """
    Remove an existing output file.

    This is especially useful when a previous failed Rasterio/GDAL write
    left behind a corrupt or incomplete TIFF.
    """
    path = Path(path)

    if not path.exists():
        return

    try:
        path.unlink()
        print(f"Removed existing output: {path}")
    except PermissionError as exc:
        raise PermissionError(
            f"Could not delete:\n{path}\n\n"
            "The file may still be open in QGIS, ArcGIS, another Python "
            "session, or another program. Close it and rerun."
        ) from exc
    except OSError:
        # Some malformed TIFFs may be awkward for GDAL but can still normally
        # be deleted by the filesystem. Re-raise with a clearer message.
        raise RuntimeError(
            f"Could not remove existing output:\n{path}\n"
            "Delete it manually and rerun the script."
        )


def check_required_inputs():
    """Check that all required input files exist."""
    missing = []

    for path, label in [
        (INPUT_RASTER, "Input raster"),
        (MODEL_PATH, "Model"),
    ]:
        if not Path(path).exists():
            missing.append(f"{label}: {path}")

    if missing:
        raise FileNotFoundError(
            "Required input file(s) not found:\n\n" + "\n".join(missing)
        )


# ============================================================
# THRESHOLD
# ============================================================

def get_waste_threshold():
    """Obtain the probability threshold used to define waste."""

    if WASTE_THRESHOLD is not None:
        threshold = float(WASTE_THRESHOLD)
        print(f"Using manually specified waste threshold: {threshold:.4f}")
        return threshold

    if THRESHOLD_CSV.exists():
        df = pd.read_csv(THRESHOLD_CSV)

        if "Optimal_Threshold" in df.columns and len(df) > 0:
            threshold = float(df.loc[0, "Optimal_Threshold"])
            print(
                f"Using validation-derived waste threshold: "
                f"{threshold:.4f}"
            )
            return threshold

        warnings.warn(
            f"{THRESHOLD_CSV} does not contain 'Optimal_Threshold'. "
            "Falling back to 0.50."
        )
    else:
        warnings.warn(
            f"Threshold file not found:\n{THRESHOLD_CSV}\n"
            "Falling back to 0.50."
        )

    return 0.50


# ============================================================
# MODEL
# ============================================================

def load_unet():
    """
    Load the trained U-Net for inference only.

    compile=False is intentional because training metrics and losses are
    unnecessary for prediction.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found:\n{MODEL_PATH}"
        )

    print(f"\nLoading model:\n{MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print(f"Input shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    if tuple(model.input_shape[1:3]) != (TILE_SIZE, TILE_SIZE):
        raise ValueError(
            f"Model expects spatial input {model.input_shape[1:3]}, "
            f"but TILE_SIZE is {TILE_SIZE}."
        )

    n_classes = int(model.output_shape[-1])

    if WASTE_CLASS_IDX >= n_classes:
        raise ValueError(
            f"WASTE_CLASS_IDX={WASTE_CLASS_IDX}, "
            f"but model has only {n_classes} output classes."
        )

    return model


# ============================================================
# TILE GENERATION
# ============================================================

def core_windows(width, height):
    """
    Yield non-overlapping output windows of up to CORE_SIZE x CORE_SIZE.

    Each output core is predicted using a larger TILE_SIZE x TILE_SIZE
    context tile containing HALO pixels around it.
    """

    for row_off in range(0, height, CORE_SIZE):
        core_h = min(CORE_SIZE, height - row_off)

        for col_off in range(0, width, CORE_SIZE):
            core_w = min(CORE_SIZE, width - col_off)

            yield (
                row_off,
                col_off,
                core_h,
                core_w,
            )


def count_core_windows(width, height):
    """Return number of inference windows without building a full list."""
    n_rows = math.ceil(height / CORE_SIZE)
    n_cols = math.ceil(width / CORE_SIZE)
    return n_rows * n_cols


def read_context_tile(src, row_off, col_off):
    """
    Read one 256 x 256 RGB inference tile.

    At the external raster edge, Rasterio pads missing values with zero.
    """

    input_window = Window(
        col_off - HALO,
        row_off - HALO,
        TILE_SIZE,
        TILE_SIZE,
    )

    tile = src.read(
        RGB_BANDS,
        window=input_window,
        boundless=True,
        fill_value=0,
    )

    tile = tile.transpose(
        1,
        2,
        0,
    ).astype(np.float32)

    if tile.shape != (TILE_SIZE, TILE_SIZE, 3):
        raise RuntimeError(
            f"Unexpected tile shape {tile.shape}; "
            f"expected {(TILE_SIZE, TILE_SIZE, 3)}."
        )

    # Same scaling used during model training.
    tile /= 255.0

    return tile


# ============================================================
# WHOLE-AREA INFERENCE
# ============================================================

def predict_probability_raster(model):
    """
    Predict waste probability across the full orthomosaic.

    Predictions are streamed directly to disk. The full probability raster
    is never held in memory.
    """

    # Remove an incomplete/corrupt file from a previous failed run.
    safe_delete(PROBABILITY_RASTER)

    with rasterio.open(INPUT_RASTER) as src:

        if src.count < 3:
            raise ValueError(
                f"{INPUT_RASTER} has {src.count} band(s). "
                "At least three RGB bands are required."
            )

        if src.crs is None:
            warnings.warn(
                "Input raster has no CRS. Vector output will also have no CRS."
            )

        print("\n=========================================")
        print("          WHOLE-AREA INFERENCE")
        print("=========================================")
        print(f"Raster       : {INPUT_RASTER}")
        print(f"Width        : {src.width:,}")
        print(f"Height       : {src.height:,}")
        print(f"Bands        : {src.count}")
        print(f"CRS          : {src.crs}")
        print(f"Resolution   : {src.res}")
        print(f"Data type    : {src.dtypes}")
        print(f"Model tile   : {TILE_SIZE} x {TILE_SIZE}")
        print(f"Halo         : {HALO} pixels")
        print(f"Output core  : {CORE_SIZE} x {CORE_SIZE}")
        print("=========================================\n")

        n_windows = count_core_windows(
            src.width,
            src.height,
        )

        print(
            f"Total prediction windows: "
            f"{n_windows:,}"
        )

        # Start from the input profile because it safely preserves
        # georeferencing, dimensions and transform.
        out_profile = src.profile.copy()

        # IMPORTANT:
        # The source orthomosaic may be RGB + JPEG + YCbCr. Those creation
        # options are invalid for a one-band float32 DEFLATE GeoTIFF.
        for key in [
            "photometric",
            "interleave",
            "compress",
            "predictor",
            "tiled",
            "blockxsize",
            "blockysize",
        ]:
            out_profile.pop(key, None)

        out_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=None,
            compress="deflate",
            predictor=2,
            photometric="MINISBLACK",
            BIGTIFF="YES",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        with rasterio.open(
            PROBABILITY_RASTER,
            "w",
            **out_profile,
        ) as dst:

            batch_tiles = []
            batch_meta = []
            done = 0

            for meta in core_windows(
                src.width,
                src.height,
            ):
                row_off, col_off, core_h, core_w = meta

                batch_tiles.append(
                    read_context_tile(
                        src,
                        row_off,
                        col_off,
                    )
                )
                batch_meta.append(meta)

                # Run prediction whenever batch is full.
                if len(batch_tiles) == BATCH_SIZE:
                    done += predict_and_write_batch(
                        model=model,
                        dst=dst,
                        batch_tiles=batch_tiles,
                        batch_meta=batch_meta,
                    )

                    batch_tiles = []
                    batch_meta = []

                    if done % 200 < BATCH_SIZE or done == n_windows:
                        pct = 100.0 * done / n_windows
                        print(
                            f"Predicted {done:,} / {n_windows:,} "
                            f"tiles ({pct:.1f}%)"
                        )

            # Final partial batch.
            if batch_tiles:
                done += predict_and_write_batch(
                    model=model,
                    dst=dst,
                    batch_tiles=batch_tiles,
                    batch_meta=batch_meta,
                )

                pct = 100.0 * done / n_windows
                print(
                    f"Predicted {done:,} / {n_windows:,} "
                    f"tiles ({pct:.1f}%)"
                )

    print(
        f"\nProbability raster saved:\n"
        f"{PROBABILITY_RASTER}"
    )


def predict_and_write_batch(
    model,
    dst,
    batch_tiles,
    batch_meta,
):
    """Predict one batch and write its central cores to the output raster."""

    batch_array = np.stack(
        batch_tiles,
        axis=0,
    )

    pred = model.predict(
        batch_array,
        verbose=0,
    )

    waste_prob = pred[
        ...,
        WASTE_CLASS_IDX
    ]

    for probs, (
        row_off,
        col_off,
        core_h,
        core_w,
    ) in zip(
        waste_prob,
        batch_meta,
    ):

        core = probs[
            HALO : HALO + core_h,
            HALO : HALO + core_w,
        ].astype(np.float32)

        out_window = Window(
            col_off,
            row_off,
            core_w,
            core_h,
        )

        dst.write(
            core,
            1,
            window=out_window,
        )

    return len(batch_meta)


# ============================================================
# THRESHOLDING
# ============================================================

def create_binary_waste_raster(threshold):
    """
    Convert the probability raster to:
        1 = predicted waste
        0 = non-waste
    """

    print(
        f"\nApplying waste threshold "
        f"{threshold:.4f}..."
    )

    safe_delete(BINARY_RASTER)

    with rasterio.open(PROBABILITY_RASTER) as src:

        profile = src.profile.copy()

        # Explicitly clean/redefine the binary output profile too.
        for key in [
            "photometric",
            "interleave",
            "compress",
            "predictor",
        ]:
            profile.pop(key, None)

        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            predictor=1,
            photometric="MINISBLACK",
            BIGTIFF="YES",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        with rasterio.open(
            BINARY_RASTER,
            "w",
            **profile,
        ) as dst:

            for _, window in src.block_windows(1):

                prob = src.read(
                    1,
                    window=window,
                )

                binary = (
                    prob >= threshold
                ).astype(np.uint8)

                dst.write(
                    binary,
                    1,
                    window=window,
                )

    print(
        f"Binary waste raster saved:\n"
        f"{BINARY_RASTER}"
    )


# ============================================================
# VECTORISATION
# ============================================================

def vectorise_and_filter(threshold):
    """
    Polygonise predicted waste and save:

      1. every predicted component;
      2. filtered components with > MIN_PIXELS pixels;
      3. a CSV summary.

    Keeping the unfiltered output is useful for later hard-negative review.
    """

    print("\nVectorising predicted waste...")

    safe_delete(ALL_VECTOR)
    safe_delete(FILTERED_VECTOR)
    safe_delete(OBJECT_CSV)

    with rasterio.open(BINARY_RASTER) as src:

        pixel_width = abs(src.transform.a)
        pixel_height = abs(src.transform.e)
        pixel_area = pixel_width * pixel_height

        if pixel_area <= 0:
            raise ValueError(
                "Could not determine a valid raster pixel area."
            )

        projected = (
            src.crs is not None
            and not src.crs.is_geographic
        )

        if not projected:
            warnings.warn(
                "Raster CRS is not projected. Geometry areas will be in "
                "coordinate-system units rather than reliable square metres."
            )

        crs_wkt = (
            src.crs.to_wkt()
            if src.crs is not None
            else None
        )

        schema = {
            "geometry": "Polygon",
            "properties": {
                "object_id": "int",
                "class": "str:20",
                "raw_label": "int",
                "threshold": "float",
                "n_pixels": "int",
                "area_m2": "float",
                "kept": "int",
            },
        }

        with fiona.open(
            ALL_VECTOR,
            mode="w",
            driver="GPKG",
            layer="waste_piles_all",
            schema=schema,
            crs_wkt=crs_wkt,
        ) as all_dst, fiona.open(
            FILTERED_VECTOR,
            mode="w",
            driver="GPKG",
            layer="waste_piles_filtered",
            schema=schema,
            crs_wkt=crs_wkt,
        ) as filtered_dst, open(
            OBJECT_CSV,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "object_id",
                    "class",
                    "raw_label",
                    "threshold",
                    "n_pixels",
                    "area_m2",
                    "kept",
                ],
            )

            csv_writer.writeheader()

            n_all = 0
            n_kept = 0
            total_area = 0.0
            kept_area = 0.0

            # Rasterio band proxy avoids src.read(1) on the entire raster.
            band = rasterio.band(
                src,
                1,
            )

            for geom, value in shapes(
                band,
                transform=src.transform,
                connectivity=8,
            ):

                # Ignore background.
                if int(value) != 1:
                    continue

                polygon = shape(geom)

                if polygon.is_empty:
                    continue

                area = float(polygon.area)

                # Raster-derived polygons align to pixel boundaries, so
                # area / pixel_area gives their pixel count.
                n_pixels = int(
                    round(area / pixel_area)
                )

                keep = (
                    n_pixels > MIN_PIXELS
                    and area >= MIN_AREA_M2
                )

                n_all += 1
                object_id = n_all

                properties = {
                    "object_id": object_id,
                    "class": "Waste",
                    "raw_label": WASTE_RAW_LABEL,
                    "threshold": float(threshold),
                    "n_pixels": int(n_pixels),
                    "area_m2": float(area),
                    "kept": int(keep),
                }

                feature = {
                    "geometry": mapping(polygon),
                    "properties": properties,
                }

                # Save every prediction for later inspection and
                # hard-negative mining.
                all_dst.write(feature)

                if keep:
                    filtered_dst.write(feature)
                    n_kept += 1
                    kept_area += area

                csv_writer.writerow(
                    properties
                )

                total_area += area

                if n_all % 5000 == 0:
                    print(
                        f"Vectorised {n_all:,} predicted objects..."
                    )

    print("\n=========================================")
    print("          FILTERING SUMMARY")
    print("=========================================")
    print(f"All predicted objects   : {n_all:,}")
    print(f"Objects retained        : {n_kept:,}")
    print(f"Objects removed         : {n_all - n_kept:,}")
    print(f"Keep rule               : > {MIN_PIXELS} pixels")
    print(f"Minimum area            : {MIN_AREA_M2:.4f} m2")
    print(f"Total polygon area      : {total_area:,.2f} m2")
    print(f"Retained polygon area   : {kept_area:,.2f} m2")
    print("=========================================\n")

    print("Saved vector products:")
    print(f" - All predictions : {ALL_VECTOR}")
    print(f" - Filtered piles  : {FILTERED_VECTOR}")
    print(f" - Object table    : {OBJECT_CSV}")


# ============================================================
# MAIN
# ============================================================

def main():

    check_required_inputs()

    threshold = get_waste_threshold()

    model = load_unet()

    predict_probability_raster(
        model
    )

    create_binary_waste_raster(
        threshold
    )

    vectorise_and_filter(
        threshold
    )

    print("\n=========================================")
    print("          PROCESSING COMPLETE")
    print("=========================================")
    print(f"Probability raster: {PROBABILITY_RASTER}")
    print(f"Binary raster     : {BINARY_RASTER}")
    print(f"All polygons      : {ALL_VECTOR}")
    print(f"Filtered polygons : {FILTERED_VECTOR}")
    print(f"Object CSV        : {OBJECT_CSV}")
    print("=========================================")


if __name__ == "__main__":
    main()
