from main_train_script import build_unet, train_gen, val_gen, N_CLASSES, EPOCHS, MODEL_DIR
from utils_eval import evaluate_best_model_class1
import os
import tensorflow as tf

exp_name = "baseline"
save_dir = os.path.join(MODEL_DIR, exp_name)
os.makedirs(save_dir, exist_ok=True)

model = build_unet(N_CLASSES)

checkpoint_path = os.path.join(save_dir, "best_model.keras")
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint, early])

model.save(os.path.join(save_dir, "final_model.keras"))

evaluate_best_model_class1(checkpoint_path, val_gen, os.path.join(save_dir, "class1_metrics.csv"))