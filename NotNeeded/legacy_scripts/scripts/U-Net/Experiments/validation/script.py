import os
import random
import numpy as np
import rasterio
from skimage.transform import resize
from scipy.ndimage import rotate
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 8
EPOCHS = 500


BASE_DATA_PATH = "/head/NFS/pkalonde/ImageClassification/mapping-waste-urban-africa/data/tiled_data"
IMG_DIR = os.path.join(BASE_DATA_PATH, "images")
MASK_DIR = os.path.join(BASE_DATA_PATH, "masks")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Pre-Scan: Dynamic Label Discovery
# -----------------------------
print("Scanning masks for unique labels...")
mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith(".tif") and not f.endswith(".aux.xml")]
if not mask_files:
    raise FileNotFoundError(f"No .tif files found in {MASK_DIR}")

unique_labels = set()
for f in mask_files:
    with rasterio.open(os.path.join(MASK_DIR, f)) as src:
        band = src.read(1)
        unique_labels.update(np.unique(band))

LABEL_LIST = sorted(list(unique_labels))
N_CLASSES = len(LABEL_LIST)
LABEL_MAP = {val: i for i, val in enumerate(LABEL_LIST)}
print(f"Detected {N_CLASSES} classes: {LABEL_LIST}")
print(f"Training Map (Original -> Internal): {LABEL_MAP}")

# -----------------------------
# Custom Metrics
# -----------------------------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# -----------------------------
# Data Generator
# -----------------------------
class RasterDataGenerator(Sequence):
    def __init__(self, img_dir, mask_dir, img_files, mask_files, batch_size=BATCH_SIZE, shuffle=True, augment=True, **kwargs):
        super().__init__(**kwargs)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = img_files
        self.mask_files = mask_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        mask_set = set(mask_files)
        self.valid_indices = [i for i, f in enumerate(img_files) if f in mask_set]
        print(f"Generator initialized with {len(self.valid_indices)} matching pairs.")

        self.indices = np.array(self.valid_indices)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgs, masks = [], []

        for i in batch_idx:
            file_name = self.img_files[i]
            with rasterio.open(os.path.join(self.img_dir, file_name)) as src:
                img = src.read([1,2,3]).transpose(1,2,0).astype("float32")
            with rasterio.open(os.path.join(self.mask_dir, file_name)) as src:
                mask = src.read(1)

            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True)
            mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), order=0, preserve_range=True, anti_aliasing=False)

            if self.augment:
                if random.random() < 0.5:
                    img, mask = np.flipud(img), np.flipud(mask)
                if random.random() < 0.5:
                    angle = random.uniform(-15, 15)
                    img = rotate(img, angle, reshape=False, order=1, mode="reflect")
                    mask = rotate(mask, angle, reshape=False, order=0, mode="nearest")

            img = img / 255.0
            mapped_mask = np.zeros(mask.shape, dtype="int32")
            for raw_val, idx_map in LABEL_MAP.items():
                mapped_mask[mask == raw_val] = idx_map

            imgs.append(img)
            masks.append(to_categorical(mapped_mask, num_classes=N_CLASSES))

        return np.array(imgs), np.array(masks)
    
    # -----------------------------
# U-Net Model
# -----------------------------

def build_unet(n_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))
    
    def conv_block(x, f):
        x = Conv2D(f, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(f, 3, padding="same", activation="relu")(x)
        return BatchNormalization()(x)

    # Encoder
    s1 = conv_block(inputs, 64)
    p1 = MaxPooling2D((2,2))(s1)
    s2 = conv_block(p1, 128)
    p2 = MaxPooling2D((2,2))(s2)
    s3 = conv_block(p2, 256)
    p3 = MaxPooling2D((2,2))(s3)
    
    # Bottleneck (The deepest representation)
    b1 = conv_block(p3, 512)

    # NO DECODER: Direct prediction from bottleneck
    # The bottleneck is 1/8th the size of the original (256 -> 128 -> 64 -> 32)
    # We apply the final 1x1 convolution here
    outputs = Conv2D(n_classes, 1, activation="softmax", padding="same")(b1)
    
    # Simple mathematical upsampling (bilinear) to get back to 256x256
    # size=(8, 8) because 32 * 8 = 256
    outputs = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(outputs)
    
    return Model(inputs, outputs)


# -----------------------------
# Train-Validation Split
# -----------------------------
all_img = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".tif")])
all_msk = sorted([f for f in os.listdir(MASK_DIR) if f.endswith(".tif")])
combined = list(zip(all_img, all_msk))
random.shuffle(combined)
split_idx = int(len(combined) * 0.8)
train_pairs = combined[:split_idx]
val_pairs = combined[split_idx:]
train_imgs, train_msks = zip(*train_pairs)
val_imgs, val_msks = zip(*val_pairs)

train_gen = RasterDataGenerator(IMG_DIR, MASK_DIR, train_imgs, train_msks)
val_gen = RasterDataGenerator(IMG_DIR, MASK_DIR, val_imgs, val_msks, shuffle=False, augment=False)

import glob

# -----------------------------
# Evaluation Loop
# -----------------------------
results = []

# Find all .keras models in the directory
model_paths = glob.glob(os.path.join(MODEL_DIR, "*.keras"))
print(f"Found {len(model_paths)} models for evaluation.")

# We need to pass custom metrics to load_model
custom_objects = {
    'iou_score': iou_score,
    'dice_coef': dice_coef
}

for model_path in model_paths:
    model_name = os.path.basename(model_path)
    print(f"\n--- Evaluating Model: {model_name} ---")
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        continue

    all_preds = []
    all_trues = []

    # Iterate through the validation generator
    for i in range(len(val_gen)):
        x_val, y_true = val_gen[i]
        y_pred = model.predict(x_val, verbose=0)
        
        # Convert from one-hot/softmax to class indices
        # Shape: (Batch, H, W)
        y_true_idx = np.argmax(y_true, axis=-1)
        y_pred_idx = np.argmax(y_pred, axis=-1)
        
        all_trues.append(y_true_idx.flatten())
        all_preds.append(y_pred_idx.flatten())

    # Flatten everything for sklearn metrics
    flat_trues = np.concatenate(all_trues)
    flat_preds = np.concatenate(all_preds)

    # Calculate Metrics (Weighted handles class imbalance)
    precision = precision_score(flat_trues, flat_preds, average='weighted', zero_division=0)
    recall = recall_score(flat_trues, flat_preds, average='weighted', zero_division=0)
    f1 = f1_score(flat_trues, flat_preds, average='weighted', zero_division=0)
    
    # Calculate IOU (Mean Intersection over Union)
    # Using a simple logic for multiclass IOU
    iou_list = []
    for cls in range(N_CLASSES):
        intersection = np.logical_and(flat_trues == cls, flat_preds == cls).sum()
        union = np.logical_or(flat_trues == cls, flat_preds == cls).sum()
        if union > 0:
            iou_list.append(intersection / union)
    
    mean_iou = np.mean(iou_list) if iou_list else 0

    results.append({
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "mIOU": mean_iou
    })

# -----------------------------
# Summary and Saving
# -----------------------------
df_results = pd.DataFrame(results)
print("\n--- Final Results ---")
print(df_results)

# Save to CSV in the model directory
output_csv = os.path.join(MODEL_DIR, "ablation_results.csv")
df_results.to_csv(output_csv, index=False)
print(f"\nResults saved to {output_csv}")