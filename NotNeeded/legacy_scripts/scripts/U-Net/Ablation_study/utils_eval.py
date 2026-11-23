import csv
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from main_train_script import dice_coef, iou_score


def evaluate_best_model_class1(model_path, val_gen, save_csv_path):
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

    precision = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)

    precision_1 = float(precision[1]) if len(precision) > 1 else 0.0
    recall_1 = float(recall[1]) if len(recall) > 1 else 0.0
    f1_1 = float(f1[1]) if len(f1) > 1 else 0.0

    with open(save_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["precision_class_1", precision_1])
        writer.writerow(["recall_class_1", recall_1])
        writer.writerow(["f1_class_1", f1_1])

    print(f"Saved Class 1 metrics to {save_csv_path}")