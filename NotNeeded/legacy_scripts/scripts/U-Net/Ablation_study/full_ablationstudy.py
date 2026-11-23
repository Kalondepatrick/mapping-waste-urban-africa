import os
import csv
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from main_train_script import (
    IMG_HEIGHT, IMG_WIDTH, EPOCHS,
    train_gen, val_gen,
    MODEL_DIR, N_CLASSES,
    dice_coef, iou_score
)

# ===============================================================
# Shared Conv Block
# ===============================================================
def conv_block(x, filters, use_bn=True):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)

    return x


# ===============================================================
# Build Variant Models
# ===============================================================
def build_unet_base(n_classes):
    return _build_modified(n_classes)

def build_unet_no_bn(n_classes):
    return _build_modified(n_classes, use_bn=False)

def build_unet_no_skips(n_classes):
    return _build_modified(n_classes, use_skips=False)

def build_unet_shallow_bottleneck(n_classes):
    return _build_modified(n_classes, bottleneck_filters=256)

def build_unet_no_transpose(n_classes):
    return _build_modified(n_classes, use_transpose=False)

def build_unet_sgd(n_classes):
    model = _build_modified(n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["accuracy", dice_coef, iou_score]
    )
    return model


# ===============================================================
# Internal Builder
# ===============================================================
def _build_modified(
    n_classes,
    use_bn=True,
    use_skips=True,
    use_transpose=True,
    bottleneck_filters=512
):
    inputs = tf.keras.Input((IMG_HEIGHT, IMG_WIDTH, 3))

    # Encoder
    s1 = conv_block(inputs, 64, use_bn)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(s1)

    s2 = conv_block(p1, 128, use_bn)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(s2)

    s3 = conv_block(p2, 256, use_bn)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(s3)

    # Bottleneck
    b1 = conv_block(p3, bottleneck_filters, use_bn)

    # Decoder (Up blocks)
    def upsample(x, filters):
        if use_transpose:
            return tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
        else:
            x = tf.keras.layers.UpSampling2D((2,2), interpolation="bilinear")(x)
            return tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)

    d1 = upsample(b1, 256)
    if use_skips:
        d1 = tf.keras.layers.Concatenate()([d1, s3])
    d1 = conv_block(d1, 256, use_bn)

    d2 = upsample(d1, 128)
    if use_skips:
        d2 = tf.keras.layers.Concatenate()([d2, s2])
    d2 = conv_block(d2, 128, use_bn)

    d3 = upsample(d2, 64)
    if use_skips:
        d3 = tf.keras.layers.Concatenate()([d3, s1])
    d3 = conv_block(d3, 64, use_bn)

    outputs = tf.keras.layers.Conv2D(n_classes, 1, activation="softmax")(d3)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", dice_coef, iou_score]
    )

    return model


# ===============================================================
# Evaluate Best Model — Class 1 Only + Save CSV
# ===============================================================
def evaluate_best_model_class1(model_path, val_gen, save_csv_path):
    print(f"Evaluating best model: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"dice_coef": dice_coef, "iou_score": iou_score}
    )

    y_true_all = []
    y_pred_all = []

    for batch_imgs, batch_masks in val_gen:
        preds = model.predict(batch_imgs, verbose=0)

        true_idx = np.argmax(batch_masks, axis=-1).reshape(-1)
        pred_idx = np.argmax(preds, axis=-1).reshape(-1)

        y_true_all.extend(true_idx)
        y_pred_all.extend(pred_idx)

    # class-wise metrics
    precision = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)

    # isolate class 1 (waste)
    precision_1 = float(precision[1]) if len(precision) > 1 else 0.0
    recall_1 = float(recall[1]) if len(recall) > 1 else 0.0
    f1_1 = float(f1[1]) if len(f1) > 1 else 0.0

    # Save CSV
    with open(save_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["precision_class_1", precision_1])
        writer.writerow(["recall_class_1", recall_1])
        writer.writerow(["f1_class_1", f1_1])

    print(f"Saved metrics → {save_csv_path}")

    return precision_1, recall_1, f1_1


# ===============================================================
# Training Wrapper
# ===============================================================
def train_model(model, name):
    exp_path = os.path.join(MODEL_DIR, name)
    os.makedirs(exp_path, exist_ok=True)

    checkpoint_path = os.path.join(exp_path, "best_model.keras")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    print(f"\n===== Training Experiment: {name} =====")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # Save final model
    model.save(os.path.join(exp_path, "final_model.keras"))

    # Evaluate best model on class 1
    csv_path = os.path.join(exp_path, "class1_metrics.csv")
    evaluate_best_model_class1(checkpoint_path, val_gen, csv_path)

    return history


# ===============================================================
# Run All Experiments
# ===============================================================
def run_all_experiments(n_classes):
    experiments = [
        ("baseline", build_unet_base),
        ("no_bn", build_unet_no_bn),
        ("no_skips", build_unet_no_skips),
        ("shallow_bottleneck", build_unet_shallow_bottleneck),
        ("no_transpose", build_unet_no_transpose),
        ("sgd_optimizer", build_unet_sgd),
    ]

    for name, builder in experiments:
        model = builder(n_classes)
        train_model(model, name)


# ===============================================================
# Entry Point
# ===============================================================
if __name__ == "__main__":
    run_all_experiments(N_CLASSES)