import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import rasterio
from rasterio.windows import Window
from skimage.transform import resize
from pathlib import Path
import gc

# ---------------------------------------------------------
# 1. PATHS & CONFIG
# ---------------------------------------------------------
BASE_PATH = Path.home() / "Documents" / "GitHub" / "mapping-waste-urban-africa"

IMG_PATH    = BASE_PATH / "data/optical.tiff"
MODEL_PATH  = BASE_PATH / "data/trained_models/tensorflow/filters_16_march_finetuned_best.keras"
OUTPUT_PATH = BASE_PATH / "data/predictions/waste_probability_full.tif"

TILE_SIZE   = 256
OVERLAP     = 32
BLOCK_SIZE  = 2048
BATCH_SIZE  = 32

# Class 1 = waste → index 0 in the softmax output
WASTE_CLASS_INDEX = 0

# ---------------------------------------------------------
# 2. LOAD MODEL
# ---------------------------------------------------------
print("Loading model...")
tf.keras.backend.clear_session()
MODEL = load_model(MODEL_PATH, compile=False)

# ---------------------------------------------------------
# 3. INFERENCE — OUTPUT WASTE CLASS PROBABILITY
# ---------------------------------------------------------
if not IMG_PATH.exists():
    raise FileNotFoundError(f"Input image not found: {IMG_PATH}")

print(f"Processing: {IMG_PATH.name}")

with rasterio.open(IMG_PATH) as src:
    profile = src.profile.copy()
    H, W = src.height, src.width

    # Single-band float32 probability raster (values 0.0–1.0, nodata = -1)
    profile.update({
        "count":      1,
        "dtype":      "float32",
        "tiled":      True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress":   "lzw",
        "nodata":     -1,
        "BIGTIFF":    "YES",
    })

    print(f"Image dimensions: {H} x {W} px")
    print(f"Output: {OUTPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:

        for row in range(0, H, BLOCK_SIZE):
            for col in range(0, W, BLOCK_SIZE):

                block_h = min(BLOCK_SIZE, H - row)
                block_w = min(BLOCK_SIZE, W - col)
                block_win = Window(col, row, block_w, block_h)

                # Accumulate waste-class probabilities across overlapping tiles
                block_prob    = np.zeros((block_h, block_w), dtype=np.float32)
                block_weights = np.zeros((block_h, block_w), dtype=np.float32)

                step = TILE_SIZE - OVERLAP

                batch_tiles    = []
                batch_metadata = []

                for y in range(0, block_h, step):
                    for x in range(0, block_w, step):
                        global_y, global_x = row + y, col + x
                        read_h = min(TILE_SIZE, H - global_y)
                        read_w = min(TILE_SIZE, W - global_x)

                        if read_h < 10 or read_w < 10:
                            continue

                        read_win = Window(global_x, global_y, read_w, read_h)
                        tile = src.read([1, 2, 3], window=read_win)

                        if np.max(tile) == 0:
                            continue

                        tile_tf = np.moveaxis(tile, 0, -1)
                        if (read_h, read_w) != (TILE_SIZE, TILE_SIZE):
                            tile_input = resize(tile_tf, (TILE_SIZE, TILE_SIZE), preserve_range=True)
                        else:
                            tile_input = tile_tf

                        x_in = tile_input.astype(np.float32) / 255.0

                        batch_tiles.append(x_in)
                        batch_metadata.append((y, x, read_h, read_w))

                        if len(batch_tiles) == BATCH_SIZE:
                            preds = MODEL.predict(np.array(batch_tiles), verbose=0)
                            for pred, (by, bx, bh, bw) in zip(preds, batch_metadata):
                                # pred shape: (256, 256, NUM_CLASSES)
                                waste_prob = pred[:, :, WASTE_CLASS_INDEX]
                                if (bh, bw) != (TILE_SIZE, TILE_SIZE):
                                    waste_prob = resize(waste_prob, (bh, bw), preserve_range=True)

                                th = min(bh, block_h - by)
                                tw = min(bw, block_w - bx)
                                block_prob[by:by+th, bx:bx+tw]    += waste_prob[:th, :tw]
                                block_weights[by:by+th, bx:bx+tw] += 1

                            batch_tiles, batch_metadata = [], []

                # Leftover batch
                if batch_tiles:
                    preds = MODEL.predict(np.array(batch_tiles), verbose=0)
                    for pred, (by, bx, bh, bw) in zip(preds, batch_metadata):
                        waste_prob = pred[:, :, WASTE_CLASS_INDEX]
                        if (bh, bw) != (TILE_SIZE, TILE_SIZE):
                            waste_prob = resize(waste_prob, (bh, bw), preserve_range=True)

                        th = min(bh, block_h - by)
                        tw = min(bw, block_w - bx)
                        block_prob[by:by+th, bx:bx+tw]    += waste_prob[:th, :tw]
                        block_weights[by:by+th, bx:bx+tw] += 1

                # Average overlapping tile predictions
                valid = block_weights > 0
                block_out = np.full((block_h, block_w), -1, dtype=np.float32)
                block_out[valid] = block_prob[valid] / block_weights[valid]

                dst.write(block_out, 1, window=block_win)

                del block_prob, block_weights, block_out, batch_tiles, batch_metadata
                gc.collect()

            print(f"  Block row {row} / {H} done")

print(f"\nDone. Waste probability raster saved to: {OUTPUT_PATH}")
print("Load in QGIS and experiment with thresholds (e.g. > 0.5, > 0.7) to tune sensitivity.")