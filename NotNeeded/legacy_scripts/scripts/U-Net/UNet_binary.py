import os
import random
import numpy as np
import rasterio
from skimage.transform import resize
from scipy.ndimage import rotate
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 8
EPOCHS = 500

IMG_DIR = "data/tiled_data/images"
MASK_DIR = "data/tiled_data/masks_bin"  # Binary masks (0=background, 1=waste)
MODEL_DIR = "models_binary"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Custom Metrics
# -----------------------------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
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
                mask = src.read(1).astype("float32")
                mask[mask > 0] = 1  # ensure binary

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
            mask = np.expand_dims(mask, axis=-1)  # shape: HxWx1
            imgs.append(img)
            masks.append(mask)

        return np.array(imgs), np.array(masks)

# -----------------------------
# U-Net Model (Binary)
# -----------------------------
def build_unet_binary():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))
    def conv_block(x, f):
        x = Conv2D(f, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(f, 3, padding="same", activation="relu")(x)
        return BatchNormalization()(x)
    s1 = conv_block(inputs, 64)
    p1 = MaxPooling2D((2,2))(s1)
    s2 = conv_block(p1, 128)
    p2 = MaxPooling2D((2,2))(s2)
    s3 = conv_block(p2, 256)
    p3 = MaxPooling2D((2,2))(s3)
    b1 = conv_block(p3, 512)
    d1 = Conv2DTranspose(256, 2, strides=2, padding="same")(b1)
    d1 = Concatenate()([d1, s3])
    d1 = conv_block(d1, 256)
    d2 = Conv2DTranspose(128, 2, strides=2, padding="same")(d1)
    d2 = Concatenate()([d2, s2])
    d2 = conv_block(d2, 128)
    d3 = Conv2DTranspose(64, 2, strides=2, padding="same")(d2)
    d3 = Concatenate()([d3, s1])
    d3 = conv_block(d3, 64)
    outputs = Conv2D(1, 1, activation="sigmoid")(d3)
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

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "best_unet_binary.h5"),
                             monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

# -----------------------------
# Compile & Train
# -----------------------------
model = build_unet_binary()
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy", dice_coef, iou_score])

history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=EPOCHS,
                    callbacks=[early_stop, checkpoint, reduce_lr])

# -----------------------------
# Evaluate on Validation Set
# -----------------------------
y_true_all, y_pred_all = [], []

for X, Y in val_gen:
    y_pred_batch = model.predict(X)
    y_true_batch = Y.flatten()
    y_pred_batch = (y_pred_batch.flatten() > 0.5).astype(np.uint8)
    y_true_all.extend(y_true_batch)
    y_pred_all.extend(y_pred_batch)

precision = precision_score(y_true_all, y_pred_all, zero_division=0)
recall = recall_score(y_true_all, y_pred_all, zero_division=0)
f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall:    {recall:.4f}")
print(f"Validation F1-score:  {f1:.4f}")

# -----------------------------
# Save Final Model
# -----------------------------
save_path = os.path.join(MODEL_DIR, "unet_binary_final.h5")
model.save(save_path)
print(f"Final binary model saved at {save_path}")

# -----------------------------
# Plot Training History
# -----------------------------
PLOT_DIR = os.path.join(MODEL_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Loss
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "loss.png"))
plt.close()

# Accuracy
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "accuracy.png"))
plt.close()

# Dice
plt.figure(figsize=(8,6))
plt.plot(history.history['dice_coef'], label='Training Dice')
plt.plot(history.history['val_dice_coef'], label='Validation Dice')
plt.title("Dice Coefficient")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "dice.png"))
plt.close()

# IoU
plt.figure(figsize=(8,6))
plt.plot(history.history['iou_score'], label='Training IoU')
plt.plot(history.history['val_iou_score'], label='Validation IoU')
plt.title("IoU Score")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "iou.png"))
plt.close()

print(f"Training plots saved in {PLOT_DIR}")
