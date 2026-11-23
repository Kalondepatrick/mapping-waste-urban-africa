import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from skimage.transform import resize
from pathlib import Path
import gc
import fiona
from shapely.geometry import shape, mapping

# ---------------------------------------------------------
# 1. PATHS & CONFIG
# ---------------------------------------------------------
BASE_PATH = Path.home() / "Documents" / "GitHub" / "mapping-waste-urban-africa"

IMG_PATH = BASE_PATH / "data/optical.tiff"
MODEL_PATH = BASE_PATH / "data/trained_models/tensorflow/filters_16_hardnegative_finetuned.keras"

OUTPUT_PATH = BASE_PATH / "data/predictions/landcover_predicted_class_full_hn.tif"
WASTE_SHP_PATH = BASE_PATH / "data/predictions/waste_piles_vector/waste_piles_hn.shp"

# Final categorical raster code for Waste.
# The model channel is zero-based internally, but the output raster uses
# 1..NUM_CLASSES because 0 is reserved for NoData.
WASTE_CLASS_CODE = 1

# 8-connectivity treats diagonally touching waste pixels as one object.
WASTE_CONNECTIVITY = 8

# Set > 0 to remove very small polygons. This assumes a projected CRS
# whose horizontal units are metres (e.g. UTM). Keep 0.0 for no area filter.
MIN_WASTE_AREA_M2 = 0.0

TILE_SIZE  = 256
OVERLAP    = 32
BLOCK_SIZE = 2048
BATCH_SIZE = 32

# IMPORTANT:
# Must match the order of the model's output channels
CLASS_NAMES = [
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

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------
# 2. LOAD MODEL
# ---------------------------------------------------------
print("Loading model...")

tf.keras.backend.clear_session()
MODEL = load_model(MODEL_PATH, compile=False)

print("Model output shape:", MODEL.output_shape)

if MODEL.output_shape[-1] != NUM_CLASSES:
    raise ValueError(
        f"Model outputs {MODEL.output_shape[-1]} classes, "
        f"but CLASS_NAMES contains {NUM_CLASSES} classes."
    )

# ---------------------------------------------------------
# 3. CHECK INPUT
# ---------------------------------------------------------
if not IMG_PATH.exists():
    raise FileNotFoundError(f"Input image not found: {IMG_PATH}")

print(f"Processing: {IMG_PATH.name}")

# ---------------------------------------------------------
# 4. INFERENCE
# ---------------------------------------------------------
with rasterio.open(IMG_PATH) as src:

    H, W = src.height, src.width

    print(f"Image dimensions: {H} x {W} px")
    print(f"Output: {OUTPUT_PATH}")

    # One-band categorical raster
    profile = src.profile.copy()

    profile.update({
        "count": 1,
        "dtype": "uint8",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
        "nodata": 0,
        "BIGTIFF": "YES",
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    step = TILE_SIZE - OVERLAP

    with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:

        for row in range(0, H, BLOCK_SIZE):

            for col in range(0, W, BLOCK_SIZE):

                block_h = min(BLOCK_SIZE, H - row)
                block_w = min(BLOCK_SIZE, W - col)

                block_win = Window(
                    col,
                    row,
                    block_w,
                    block_h
                )

                # Probabilities are accumulated temporarily
                # but are NOT written to disk
                block_prob = np.zeros(
                    (block_h, block_w, NUM_CLASSES),
                    dtype=np.float32
                )

                block_weights = np.zeros(
                    (block_h, block_w),
                    dtype=np.float32
                )

                batch_tiles = []
                batch_metadata = []

                # -------------------------------------------------
                # Generate overlapping tiles
                # -------------------------------------------------
                for y in range(0, block_h, step):

                    for x in range(0, block_w, step):

                        global_y = row + y
                        global_x = col + x

                        read_h = min(
                            TILE_SIZE,
                            H - global_y
                        )

                        read_w = min(
                            TILE_SIZE,
                            W - global_x
                        )

                        if read_h < 10 or read_w < 10:
                            continue

                        read_win = Window(
                            global_x,
                            global_y,
                            read_w,
                            read_h
                        )

                        tile = src.read(
                            [1, 2, 3],
                            window=read_win
                        )

                        # Skip empty imagery
                        if np.max(tile) == 0:
                            continue

                        tile_tf = np.moveaxis(
                            tile,
                            0,
                            -1
                        )

                        # Resize edge tiles
                        if (read_h, read_w) != (
                            TILE_SIZE,
                            TILE_SIZE
                        ):

                            tile_input = resize(
                                tile_tf,
                                (
                                    TILE_SIZE,
                                    TILE_SIZE,
                                    tile_tf.shape[-1]
                                ),
                                preserve_range=True
                            )

                        else:
                            tile_input = tile_tf

                        # Same preprocessing used in training
                        x_in = (
                            tile_input.astype(np.float32)
                            / 255.0
                        )

                        batch_tiles.append(x_in)

                        batch_metadata.append(
                            (
                                y,
                                x,
                                read_h,
                                read_w
                            )
                        )

                        # -----------------------------------------
                        # Process full batch
                        # -----------------------------------------
                        if len(batch_tiles) == BATCH_SIZE:

                            preds = MODEL.predict(
                                np.array(batch_tiles),
                                verbose=0
                            )

                            for pred, (
                                by,
                                bx,
                                bh,
                                bw
                            ) in zip(
                                preds,
                                batch_metadata
                            ):

                                if (bh, bw) != (
                                    TILE_SIZE,
                                    TILE_SIZE
                                ):

                                    pred = resize(
                                        pred,
                                        (
                                            bh,
                                            bw,
                                            NUM_CLASSES
                                        ),
                                        preserve_range=True
                                    )

                                th = min(
                                    bh,
                                    block_h - by
                                )

                                tw = min(
                                    bw,
                                    block_w - bx
                                )

                                block_prob[
                                    by:by+th,
                                    bx:bx+tw,
                                    :
                                ] += pred[
                                    :th,
                                    :tw,
                                    :
                                ]

                                block_weights[
                                    by:by+th,
                                    bx:bx+tw
                                ] += 1

                            batch_tiles = []
                            batch_metadata = []

                # -------------------------------------------------
                # Process leftover batch
                # -------------------------------------------------
                if batch_tiles:

                    preds = MODEL.predict(
                        np.array(batch_tiles),
                        verbose=0
                    )

                    for pred, (
                        by,
                        bx,
                        bh,
                        bw
                    ) in zip(
                        preds,
                        batch_metadata
                    ):

                        if (bh, bw) != (
                            TILE_SIZE,
                            TILE_SIZE
                        ):

                            pred = resize(
                                pred,
                                (
                                    bh,
                                    bw,
                                    NUM_CLASSES
                                ),
                                preserve_range=True
                            )

                        th = min(
                            bh,
                            block_h - by
                        )

                        tw = min(
                            bw,
                            block_w - bx
                        )

                        block_prob[
                            by:by+th,
                            bx:bx+tw,
                            :
                        ] += pred[
                            :th,
                            :tw,
                            :
                        ]

                        block_weights[
                            by:by+th,
                            bx:bx+tw
                        ] += 1

                # -------------------------------------------------
                # Average overlapping predictions
                # -------------------------------------------------
                valid = block_weights > 0

                class_out = np.zeros(
                    (block_h, block_w),
                    dtype=np.uint8
                )

                if np.any(valid):

                    # Average class probabilities
                    averaged_probs = (
                        block_prob[valid, :]
                        / block_weights[valid, None]
                    )

                    # Select class with maximum probability
                    # +1 because 0 is reserved for NoData
                    class_out[valid] = (
                        np.argmax(
                            averaged_probs,
                            axis=1
                        ) + 1
                    ).astype(np.uint8)

                # -------------------------------------------------
                # WRITE ONLY FINAL CLASSIFICATION
                # -------------------------------------------------
                dst.write(
                    class_out,
                    1,
                    window=block_win
                )

                del (
                    block_prob,
                    block_weights,
                    class_out,
                    batch_tiles,
                    batch_metadata
                )

                gc.collect()

            print(f"Block row {row} / {H} done")

# ---------------------------------------------------------
# 5. EXTRACT WASTE CLASS TO SHAPEFILE
# ---------------------------------------------------------
print("\nExtracting waste polygons...")

# Remove an existing shapefile set so this run cleanly replaces it.
for sidecar in WASTE_SHP_PATH.parent.glob(f"{WASTE_SHP_PATH.stem}.*"):
    if sidecar.suffix.lower() in {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"}:
        sidecar.unlink(missing_ok=True)

with rasterio.open(OUTPUT_PATH) as src:
    if src.crs is None:
        raise ValueError(
            "The classified raster has no CRS. A CRS is required to create "
            "a correctly georeferenced waste shapefile."
        )

    crs_is_geographic = bool(src.crs.is_geographic)

    if MIN_WASTE_AREA_M2 > 0 and crs_is_geographic:
        raise ValueError(
            "MIN_WASTE_AREA_M2 is greater than zero, but the raster CRS is "
            "geographic (degrees). Reproject to a metre-based projected CRS "
            "or set MIN_WASTE_AREA_M2 = 0.0."
        )

    schema = {
        "geometry": "Polygon",
        "properties": {
            "pile_id": "int",
            "class_id": "int",
            "class_name": "str:20",
            "area_m2": "float:24.3",
        },
    }

    WASTE_SHP_PATH.parent.mkdir(parents=True, exist_ok=True)
    waste_count = 0

    with fiona.open(
        WASTE_SHP_PATH,
        mode="w",
        driver="ESRI Shapefile",
        crs_wkt=src.crs.to_wkt(),
        schema=schema,
    ) as shp:

        # Polygonise the categorical raster directly from disk.
        # Only connected regions whose class code is Waste are retained.
        for geom, value in shapes(
            rasterio.band(src, 1),
            transform=src.transform,
            connectivity=WASTE_CONNECTIVITY,
        ):
            if int(value) != WASTE_CLASS_CODE:
                continue

            waste_geom = shape(geom)
            if waste_geom.is_empty:
                continue

            # Valid m² only when the raster uses a metre-based projected CRS.
            area_m2 = -1.0 if crs_is_geographic else float(waste_geom.area)

            if MIN_WASTE_AREA_M2 > 0 and area_m2 < MIN_WASTE_AREA_M2:
                continue

            waste_count += 1
            shp.write({
                "geometry": mapping(waste_geom),
                "properties": {
                    "pile_id": waste_count,
                    "class_id": WASTE_CLASS_CODE,
                    "class_name": "Waste",
                    "area_m2": area_m2,
                },
            })

print(f"Waste polygons extracted: {waste_count}")
print(f"Waste shapefile saved to:\n{WASTE_SHP_PATH}")

# ---------------------------------------------------------
# 6. FINISHED
# ---------------------------------------------------------
print("\nDone.")

print(
    f"Predicted land-cover surface saved to:\n"
    f"{OUTPUT_PATH}"
)

print("\nClass codes:")

for i, name in enumerate(CLASS_NAMES, start=1):
    print(f"  {i} = {name}")
