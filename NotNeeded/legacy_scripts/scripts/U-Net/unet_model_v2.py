import os
import numpy as np
import tifffile as tiff
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation, Concatenate

IMG_HEIGHT = 576
IMG_WIDTH  = 576
N_CLASSES  = 1  # binary mask


class TiffDataGenerator(Sequence):
    def __init__(self, img_dir, mask_dir, batch_size=4, shuffle=True):
        self.img_files  = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.tif')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.tif')])
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.img_files))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.img_files) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        imgs, masks = [], []
        for i in batch_idx:
            img  = tiff.imread(os.path.join(self.img_dir,  self.img_files[i]))
            mask = tiff.imread(os.path.join(self.mask_dir, self.mask_files[i]))

            # Keep only first 3 channels if RGB or RGBA
            if img.ndim == 3 and img.shape[-1] >= 4:
                img = img[..., :3]

            # Expand grayscale images to 3 channels
            if img.ndim == 2:
                img = np.repeat(np.expand_dims(img, -1), 3, axis=-1)

            # Resize
            img  = resize(img,  (IMG_HEIGHT, IMG_WIDTH), preserve_range=True, anti_aliasing=True)
            mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True, anti_aliasing=True)

            img  = img.astype('float32') / 255.
            mask = (mask > 0).astype('float32')

            imgs.append(img)
            masks.append(np.expand_dims(mask, -1))

        return np.array(imgs), np.array(masks)

# Instantiate generator
train_gen = TiffDataGenerator("Images", "mask", batch_size=4)

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, skip_tensor, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_tensor)
    x = Concatenate()([x, skip_tensor])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model
model = build_unet()
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train for, e.g., 20 epochs
tf.keras.backend.clear_session()
EPOCHS = 5

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    verbose=1
)


# ===============================
# 5. Predict and visualize
# ===============================
# Clear previous session
tf.keras.backend.clear_session()

# Pick a random image from first batch
images, masks = train_gen[0]
idx = np.random.randint(len(images))

sample_image = images[idx]
sample_mask  = masks[idx]

# Predict mask
predicted_mask = model.predict(sample_image[np.newaxis, ...])[0]
threshold = 0.5
predicted_mask_bin = (predicted_mask > threshold).astype(np.float32)

# Plot
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(sample_image)
ax[0].set_title("Image")
ax[0].axis('off')

ax[1].imshow(sample_mask.squeeze(), cmap="gray")
ax[1].set_title("Ground Truth Mask")
ax[1].axis('off')

ax[2].imshow(predicted_mask_bin.squeeze(), cmap="gray")
ax[2].set_title(f"Predicted Mask (threshold={threshold})")
ax[2].axis('off')
plt.show()
