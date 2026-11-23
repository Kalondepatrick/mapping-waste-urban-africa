import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras as keras
import rasterio
from pathlib import Path
import cv2
import pandas as pd
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
MODEL_PATH = BASE_PATH / "data/trained_models/tensorflow/best_unet_model_evaluation_copy.keras"
VAL_IMG_DIR = BASE_PATH / "data/tiled_data/images"
VAL_MASK_DIR = BASE_PATH / "data/tiled_data/masks"
RESULT_DIR = BASE_PATH / "results/02_unet_tensorflow/evaluation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_VIZ_PATH = RESULT_DIR / "professional_prediction_comparison.png"
CSV_LOG_PATH = RESULT_DIR / "per_tile_metrics.csv"

TILE_SIZE = 256
WASTE_CLASS_ID = 1  # The pixel value for 'Waste' in your ground truth .tif
WASTE_INDEX = 0     # The channel index for 'Waste' in your model output
THRESHOLD = 0.5     # Probability threshold (Adjust to 0.3 to increase Recall)

# ---------------------------------------------------------
# 2. LOAD MODEL
# ---------------------------------------------------------
print("🚀 Loading Model...")
model = load_model(str(MODEL_PATH), compile=False)
print("✅ SUCCESS: Model loaded!")

# ---------------------------------------------------------
# 3. EVALUATION LOOP
# ---------------------------------------------------------
img_files = list(VAL_IMG_DIR.glob("*.[tT][iI][fF]"))
results_log = []
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
    
    # Predict Probabilities
    pred_probs = model.predict(x, verbose=0)[0]
    
    # Extract Waste Channel & Threshold
    # If multiclass, pred_probs has shape (H, W, Classes)
    if pred_probs.ndim == 3:
        waste_probs = pred_probs[:, :, WASTE_INDEX]
    else:
        waste_probs = pred_probs

    pred_waste = (waste_probs > THRESHOLD).astype(np.uint8)
    
    # Process Ground Truth
    true_waste = (true_mask == WASTE_CLASS_ID).astype(np.uint8)
    true_waste_res = cv2.resize(true_waste, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_NEAREST)

    # Metrics (flattened)
    p_flat, t_flat = pred_waste.flatten(), true_waste_res.flatten()

    # Only evaluate tiles with some content (Waste in truth or prediction)
    if np.sum(t_flat) > 0 or np.sum(p_flat) > 0:
        prec, rec, f1, _ = precision_recall_fscore_support(t_flat, p_flat, average='binary', zero_division=0)
        iou = jaccard_score(t_flat, p_flat, zero_division=0)
        
        results_log.append({
            'tile': img_path.name,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'iou': iou
        })

        # Save a high-quality visualization for the first "hit"
        if not viz_saved and iou > 0.1:
            fig, axes = plt.subplots(1, 3, figsize=(20, 8), facecolor='#f8f8f8')
            
            # 1. RGB
            axes[0].imshow(img_resized)
            axes[0].set_title("Satellite Image (RGB)", fontsize=14, fontweight='bold')
            axes[0].axis('off')

            # 2. Ground Truth Overlay
            axes[1].imshow(img_resized)
            axes[1].imshow(true_waste_res, cmap='Reds', alpha=0.4)
            axes[1].set_title("Ground Truth (Red)", fontsize=14, fontweight='bold')
            axes[1].axis('off')

            # 3. Model Prediction Overlay
            axes[2].imshow(img_resized)
            axes[2].imshow(pred_waste, cmap='Blues', alpha=0.4)
            axes[2].set_title("Model Prediction (Blue)", fontsize=14, fontweight='bold')
            axes[2].axis('off')

            fig.suptitle(f"Detection Analysis: {img_path.name}\nIoU: {iou:.3f} | F1: {f1:.3f}", 
                         fontsize=18, fontweight='black', y=0.95)
            
            plt.savefig(SAVE_VIZ_PATH, dpi=300, bbox_inches='tight')
            print(f"🎨 Professional visualization saved to {SAVE_VIZ_PATH}")
            viz_saved = True

# ---------------------------------------------------------
# 4. FINAL REPORT & EXPORT
# ---------------------------------------------------------
df_results = pd.DataFrame(results_log)
df_results.to_csv(CSV_LOG_PATH, index=False)

print("\n" + "═"*60)
print(f"            FINAL VALIDATION SUMMARY")
print("═"*60)
if not df_results.empty:
    print(f" Total Tiles with Waste: {len(df_results)}")
    print(f" Mean Precision:        {df_results['precision'].mean():.4f}")
    print(f" Mean Recall:           {df_results['recall'].mean():.4f}")
    print(f" Mean F1-Score:         {df_results['f1_score'].mean():.4f}")
    print(f" Mean IoU:              {df_results['iou'].mean():.4f}")
    print(f" Metrics saved to:      {CSV_LOG_PATH}")
else:
    print(" No waste was detected in any of the processed tiles.")
print("═"*60)