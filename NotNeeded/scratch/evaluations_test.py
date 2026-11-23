import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras as keras
import rasterio
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, precision_recall_fscore_support

# ---------------------------------------------------------
# 0. GLOBAL COMPATIBILITY FIX
# ---------------------------------------------------------
class SafeBatchNormalization(BatchNormalization):
    def __init__(self, **kwargs):
        for key in ['renorm', 'renorm_clipping', 'renorm_momentum']:
            kwargs.pop(key, None)
        super().__init__(**kwargs)

keras.utils.get_custom_objects()['BatchNormalization'] = SafeBatchNormalization

# ---------------------------------------------------------
# 1. PATHS & CONFIGURATION
# ---------------------------------------------------------
BASE_PATH = Path("/head/NFS/pkalonde/ImageClassification/mapping-waste-urban-africa")
MODEL_PATH = BASE_PATH / "03_evaluations/models/best_unet_model.keras"
VAL_IMG_DIR = BASE_PATH / "data/tiled_data/images"
VAL_MASK_DIR = BASE_PATH / "data/tiled_data/masks"
SAVE_VIZ_PATH = BASE_PATH / "03_evaluations/professional_prediction_comparison.png"

TILE_SIZE = 256
WASTE_CLASS_ID = 1  # Pixel value in your .tif ground truth
WASTE_INDEX = 0     # Channel index in model output (usually 0 or 1)
THRESHOLD = 0.5     # Probability threshold for binary classification

# ---------------------------------------------------------
# 2. LOAD MODEL
# ---------------------------------------------------------
print("🚀 Loading Model...")
model = load_model(str(MODEL_PATH), compile=False)
print("✅ SUCCESS: Model loaded!")

# ---------------------------------------------------------
# 3. EVALUATION & VISUALIZATION
# ---------------------------------------------------------
img_files = list(VAL_IMG_DIR.glob("*.[tT][iI][fF]"))
precisions, recalls, f1s, ious = [], [], [], []
viz_saved = False

print(f"📊 Evaluating {len(img_files)} tiles...")

for img_path in img_files:
    mask_path = VAL_MASK_DIR / img_path.name
    if not mask_path.exists(): continue

    with rasterio.open(img_path) as src:
        img = np.moveaxis(src.read([1, 2, 3]), 0, -1)
    with rasterio.open(mask_path) as src:
        true_mask = src.read(1)

    # Preprocess
    img_resized = cv2.resize(img, (TILE_SIZE, TILE_SIZE))
    x = np.expand_dims(img_resized.astype(np.float32) / 255.0, 0)
    
    # Predict
    pred_probs = model.predict(x, verbose=0)[0]
    
    # If model is multiclass, extract the waste channel specifically
    if pred_probs.shape[-1] > 1:
        waste_probs = pred_probs[:, :, WASTE_INDEX]
    else:
        waste_probs = pred_probs.squeeze()

    # Thresholding for binary map
    pred_waste = (waste_probs > THRESHOLD).astype(np.uint8)
    
    # Ground Truth processing
    true_waste = (true_mask == WASTE_CLASS_ID).astype(np.uint8)
    true_waste_res = cv2.resize(true_waste, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_NEAREST)

    # Metrics calculation
    p_flat = pred_waste.flatten()
    t_flat = true_waste_res.flatten()

    if np.sum(t_flat) > 0 or np.sum(p_flat) > 0:
        p, r, f, _ = precision_recall_fscore_support(t_flat, p_flat, average='binary', zero_division=0)
        iou = jaccard_score(t_flat, p_flat, zero_division=0)
        
        precisions.append(p); recalls.append(r); f1s.append(f); ious.append(iou)

        # PROFESSIONAL PLOTTING
        if not viz_saved:
            fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor='#f0f0f0')
            plt.subplots_adjust(top=0.85)

            # 1. RGB Satellite
            axes[0].imshow(img_resized)
            axes[0].set_title("Satellite Imagery (RGB)", fontsize=14, fontweight='bold', pad=10)
            axes[0].axis('off')

            # 2. Ground Truth
            axes[1].imshow(img_resized) # Underlay RGB
            axes[1].imshow(true_waste_res, cmap='Reds', alpha=0.5) # Overlay Mask
            axes[1].set_title("Ground Truth Overlay", fontsize=14, fontweight='bold', pad=10)
            axes[1].axis('off')

            # 3. Prediction
            axes[2].imshow(img_resized) # Underlay RGB
            axes[2].imshow(pred_waste, cmap='Blues', alpha=0.5) # Overlay Pred
            axes[2].set_title("Model Prediction Overlay", fontsize=14, fontweight='bold', pad=10)
            axes[2].axis('off')

            # Add Local Metrics to the figure
            fig.suptitle(f"Waste Detection Analysis | Tile: {img_path.name}\n"
                         f"Local IoU: {iou:.3f} | Local F1: {f:.3f}", 
                         fontsize=18, fontweight='black', y=0.98)
            
            plt.tight_layout()
            plt.savefig(SAVE_VIZ_PATH, dpi=300, bbox_inches='tight')
            print(f"✅ Professional visualization saved to {SAVE_VIZ_PATH}")
            viz_saved = True

# ---------------------------------------------------------
# 4. FINAL REPORT
# ---------------------------------------------------------
print("\n" + "═"*60)
print(f"            FINAL VALIDATION SUMMARY")
print("═"*60)
if len(ious) > 0:
    print(f" Tiles Processed: {len(ious)}")
    print(f" Mean Precision:  {np.mean(precisions):.4f}")
    print(f" Mean Recall:     {np.mean(recalls):.4f}")
    print(f" Mean F1-Score:   {np.mean(f1s):.4f}")
    print(f" Mean IoU:        {np.mean(ious):.4f}")
else:
    print(" No matching waste tiles found for evaluation.")
print("═"*60)