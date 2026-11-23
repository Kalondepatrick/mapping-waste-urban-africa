import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model


# Custom BatchNormalization handler for loading models with legacy attributes
class SafeBatchNormalization(BatchNormalization):

    def __init__(self, **kwargs):
        for key in ["renorm", "renorm_clipping", "renorm_momentum"]:
            kwargs.pop(key, None)
        super().__init__(**kwargs)


keras.utils.get_custom_objects()["BatchNormalization"] = SafeBatchNormalization

# --- CONFIGURATION & PATHS ---
INPUT_DIR = Path(
    r"C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa\data\otherdata"
)
MODEL_PATH = Path(
    "./results/filters_16_best.keras"
)
OUTPUT_DIR = INPUT_DIR / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 256  # Sliding window size expected by U-Net
OPTIMAL_THRESHOLD = 0.38  # Decision threshold for binary mask
WASTE_CHANNEL_IDX = 0  # Target class index for waste

# --- 1. LOAD MODEL ---
print(f"🚀 Loading model from: {MODEL_PATH}")
model = load_model(str(MODEL_PATH), compile=False)

# --- 2. WINDOWED MULTI-OUTPUT SPATIAL INFERENCE ---
tif_files = [
    f for f in INPUT_DIR.glob("*.tif") if not f.name.endswith(".aux.xml")
]
print(f"🔮 Processing {len(tif_files)} rasters in {INPUT_DIR}...")

for img_path in tif_files:
    print(f"\nProcessing raster: {img_path.name}")

    # Define 3 separate output GeoTIFF file paths
    path_multiclass = OUTPUT_DIR / f"multiclass_map_{img_path.name}"
    path_prob = OUTPUT_DIR / f"prob_class1_{img_path.name}"
    path_binary = OUTPUT_DIR / f"binary_class1_{img_path.name}"

    # Delete pre-existing outputs to prevent GDAL read errors on open
    for path in [path_multiclass, path_prob, path_binary]:
        if path.exists():
            path.unlink()

    with rasterio.open(img_path) as src:
        height, width = src.height, src.width
        num_bands = src.count
        print(
            f" - Resolution: {width}x{height} | Source Bands: {num_bands} (using 3 bands)"
        )

        # Common metadata configuration
        base_profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": 1,
            "crs": src.crs,
            "transform": src.transform,
            "tiled": True,
            "blockxsize": TILE_SIZE,
            "blockysize": TILE_SIZE,
            "compress": "lzw",
        }

        # Profiles tailored for each output type
        profile_multiclass = {
            **base_profile,
            "dtype": rasterio.uint8,
            "nodata": 0,
        }
        profile_prob = {
            **base_profile,
            "dtype": rasterio.float32,
            "nodata": 0.0,
        }
        profile_binary = {**base_profile, "dtype": rasterio.uint8, "nodata": 0}

        # Open destination writers simultaneously
        with (
            rasterio.open(path_multiclass, "w", **profile_multiclass) as dst_multi,
            rasterio.open(path_prob, "w", **profile_prob) as dst_prob,
            rasterio.open(path_binary, "w", **profile_binary) as dst_bin,
        ):

            for y in range(0, height, TILE_SIZE):
                for x in range(0, width, TILE_SIZE):
                    # Define current spatial window
                    win_w = min(TILE_SIZE, width - x)
                    win_h = min(TILE_SIZE, height - y)
                    window = Window(
                        col_off=x, row_off=y, width=win_w, height=win_h
                    )

                    # Read RGB channels (bands 1, 2, 3), dropping 4th band if present
                    tile_data = (
                        src.read([1, 2, 3], window=window)
                        .transpose(1, 2, 0)
                        .astype("float32")
                    )

                    # Zero-pad edge tiles to 256x256 if needed
                    if win_h < TILE_SIZE or win_w < TILE_SIZE:
                        padded_tile = np.zeros(
                            (TILE_SIZE, TILE_SIZE, 3), dtype="float32"
                        )
                        padded_tile[:win_h, :win_w, :] = tile_data
                        input_tile = padded_tile
                    else:
                        input_tile = tile_data

                    # Preprocess and run model inference
                    input_tile = input_tile / 255.0
                    input_batch = np.expand_dims(input_tile, axis=0)
                    pred_probs = model.predict(input_batch, verbose=0)[0]

                    # --- COMPUTE THE 3 OUTPUT TYPES ---
                    if pred_probs.ndim == 3 and pred_probs.shape[-1] > 1:
                        # 1. Multi-class classification via argmax across channel axis
                        multiclass_tile = np.argmax(pred_probs, axis=-1).astype(
                            np.uint8
                        )
                        # 2. Continuous probabilities for Class 1 (Waste)
                        prob_tile = pred_probs[:, :, WASTE_CHANNEL_IDX].astype(
                            np.float32
                        )
                    else:
                        prob_tile = pred_probs.squeeze().astype(np.float32)
                        multiclass_tile = (prob_tile >= 0.5).astype(np.uint8)

                    # 3. Hard-coded thresholded binary mask for Class 1
                    binary_tile = (prob_tile >= OPTIMAL_THRESHOLD).astype(
                        np.uint8
                    )

                    # Crop padding from tile boundaries
                    multiclass_tile = multiclass_tile[:win_h, :win_w]
                    prob_tile = prob_tile[:win_h, :win_w]
                    binary_tile = binary_tile[:win_h, :win_w]

                    # Stream tiles directly to GeoTIFF outputs
                    dst_multi.write(multiclass_tile, 1, window=window)
                    dst_prob.write(prob_tile, 1, window=window)
                    dst_bin.write(binary_tile, 1, window=window)

    print(f"  ✓ Exported Multiclass Map : {path_multiclass.name}")
    print(f"  ✓ Exported Class 1 Probs : {path_prob.name}")
    print(f"  ✓ Exported Binary Mask   : {path_binary.name}")

print(f"\n Spatial reconstruction completed. Results exported to:\n{OUTPUT_DIR}")