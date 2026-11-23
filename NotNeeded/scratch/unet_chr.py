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
WASTE_CHANNEL_IDX = 1  # Target class index for waste

# 1. Load model
print(f"Loading model: {MODEL_PATH}")
model = load_model(str(MODEL_PATH), compile=False)

# 2. Grab the first raster for inspection
sample_tif = next(INPUT_DIR.glob("*.tif"))
print(f"Inspecting sample raster: {sample_tif.name}\n")

with rasterio.open(sample_tif) as src:
    # Read a central 256x256 tile to avoid empty margin borders
    center_y = (src.height // 2) - 128
    center_x = (src.width // 2) - 128
    window = rasterio.windows.Window(center_x, center_y, 256, 256)

    tile_data = (
        src.read([1, 2, 3], window=window).transpose(1, 2, 0).astype("float32")
    )
    input_tile = tile_data / 255.0
    input_batch = np.expand_dims(input_tile, axis=0)

# 3. Predict probabilities
pred_probs = model.predict(input_batch, verbose=0)[0]

# --- DIAGNOSTIC OUTPUT ---
num_channels = pred_probs.shape[-1]
print("=" * 55)
print(f"MODEL OUTPUT SHAPE : {pred_probs.shape}")
print(f"NUMBER OF CLASSES  : {num_channels}")
print("=" * 55)

print("\n--- PER-CHANNEL PROBABILITY STATISTICS ---")
for ch in range(num_channels):
    ch_data = pred_probs[:, :, ch]
    print(
        f"Channel {ch}: Min = {ch_data.min():.4f} | Max = {ch_data.max():.4f} | Mean = {ch_data.mean():.4f}"
    )

print("\n--- ARGMAX CLASS DISTRIBUTION (Pixel Counts) ---")
argmax_map = np.argmax(pred_probs, axis=-1)
unique_classes, counts = np.unique(argmax_map, return_counts=True)

total_pixels = argmax_map.size
for cls, count in zip(unique_classes, counts):
    pct = (count / total_pixels) * 100
    print(f"Class {cls}: {count} pixels ({pct:.2f}%)")
print("=" * 55)