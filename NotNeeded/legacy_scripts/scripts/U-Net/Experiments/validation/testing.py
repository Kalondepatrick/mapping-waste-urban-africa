
import os
import random
import numpy as np
import rasterio
import pandas as pd
import tensorflow as tf
from multiprocessing import Process, Queue
from skimage.transform import resize
from scipy.ndimage import rotate
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint

# -----------------------------
# Configuration
# -----------------------------
IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 8
EPOCHS = 10000
BASE_DATA_PATH = "C:\\Users\\Peter\\Documents\\GitHub\\mapping-waste-urban-africa\\data\\tiled_data"
IMG_DIR = os.path.join(BASE_DATA_PATH, "images")
MASK_DIR = os.path.join(BASE_DATA_PATH, "masks")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Define these globally so child processes can see them
LABEL_LIST = [0, 1, 2] 
N_CLASSES = len(LABEL_LIST)
LABEL_MAP = {val: i for i, val in enumerate(LABEL_LIST)}

# -----------------------------
# Fixed Metrics & Losses (No Lambdas!)
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

# Define explicit functions for the ablation study
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return cce + (1 - iou_score(y_true, y_pred))

# -----------------------------
# Fixed Data Generator
# -----------------------------
class RasterDataGenerator(Sequence):
    def __init__(self, img_dir, mask_dir, img_files, mask_files, batch_size=BATCH_SIZE, shuffle=True, augment=True, **kwargs):
        # Fix for the UserWarning
        super().__init__(**kwargs)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = img_files
        self.mask_files = mask_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.img_files))
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
            # ... (Rest of your rasterio loading logic remains the same)
            with rasterio.open(os.path.join(self.img_dir, file_name)) as src:
                img = src.read([1,2,3]).transpose(1,2,0).astype("float32")
            with rasterio.open(os.path.join(self.mask_dir, file_name)) as src:
                mask = src.read(1)
            
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True) / 255.0
            mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), order=0, preserve_range=True, anti_aliasing=False)
            
            mapped_mask = np.zeros(mask.shape, dtype="int32")
            for raw_val, idx_map in LABEL_MAP.items():
                mapped_mask[mask == raw_val] = idx_map
                
            imgs.append(img)
            masks.append(to_categorical(mapped_mask, num_classes=N_CLASSES))
        return np.array(imgs), np.array(masks)

# -----------------------------
# Model & Worker
# -----------------------------
def build_unet(n_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))
    x = Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    b1 = Conv2D(128, 3, padding="same", activation="relu")(x)
    outputs = Conv2D(n_classes, 1, activation="softmax", padding="same")(b1)
    outputs = UpSampling2D(size=(2, 2), interpolation='bilinear')(outputs)
    return Model(inputs, outputs)

def train_worker(loss_name, loss_fn_name, gpu_id, train_info, val_info, results_queue):
    # Map the string name back to the actual function inside the worker
    loss_map = {
        "CCE": "categorical_crossentropy",
        "DiceLoss": dice_loss,
        "Combined": combined_loss
    }
    loss_fn = loss_map[loss_fn_name]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Important for multiple processes on one GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_gen = RasterDataGenerator(IMG_DIR, MASK_DIR, train_info[0], train_info[1])
    val_gen = RasterDataGenerator(IMG_DIR, MASK_DIR, val_info[0], val_info[1], shuffle=False, augment=False)

    model = build_unet(N_CLASSES)
    model.compile(optimizer='adam', loss=loss_fn, metrics=[iou_score, dice_coef])

    model_path = os.path.join(MODEL_DIR, f"ablation_{loss_name}.keras")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, verbose=0)
    
    eval_results = model.evaluate(val_gen, verbose=0)
    results_queue.put({"Exp": loss_name, "mIOU": eval_results[1]})

# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    # Ensure this block exists to prevent EOFError/Infinite spawn loops
    all_img = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".tif")])
    all_msk = sorted([f for f in os.listdir(MASK_DIR) if f.endswith(".tif")])
    
    # Split logic...
    train_info = (all_img[:10], all_msk[:10]) # Simplified for example
    val_info = (all_img[10:15], all_msk[10:15])

    results_queue = Queue()
    # Pass function NAMES as strings to avoid pickling lambdas
    experiments = [("Exp1", "CCE"), ("Exp2", "DiceLoss")]
    
    processes = []
    for name, fn_name in experiments:
        p = Process(target=train_worker, args=(name, fn_name, 0, train_info, val_info, results_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    while not results_queue.empty():
        print(results_queue.get())