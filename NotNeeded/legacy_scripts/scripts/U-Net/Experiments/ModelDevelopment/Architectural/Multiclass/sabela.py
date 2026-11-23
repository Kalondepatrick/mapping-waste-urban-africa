import os
import re

# -----------------------------
# Configuration
# -----------------------------
SOURCE_FILE = "scripts/U-Net/Experiments/ModelDevelopment/Architectural/Multiclass/main_train_script.py"
OUTPUT_ROOT = "scripts/U-Net/Experiments/ModelDevelopment/Architectural/Multiclass/ablation_study"

def create_experiment(category, exp_name, modifications):
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: {SOURCE_FILE} not found.")
        return

    with open(SOURCE_FILE, "r") as f:
        content = f.read()

    # Define path - Use forward slashes to avoid regex escape errors
    exp_dir = f"{OUTPUT_ROOT}/{category}/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Use a raw string replacement logic instead of re.sub to avoid escape issues
    # We find the line and replace it manually
    model_dir_pattern = r'MODEL_DIR = .+'
    replacement_line = f'MODEL_DIR = "{exp_dir}"'
    content = re.sub(model_dir_pattern, lambda m: replacement_line, content)

    # Apply all code modifications
    for find_str, replace_str in modifications.items():
        content = content.replace(find_str, replace_str)

    # Save the specialized training script
    script_path = os.path.join(exp_dir, f"train_{exp_name}.py")
    with open(script_path, "w") as f:
        f.write(content)
    
    print(f"✅ Created: {category}/{exp_name}")
# =========================================================
# 1. BASELINE
# =========================================================
create_experiment("reference", "baseline", {})

# =========================================================
# 2. STRUCTURAL ABLATIONS (Removing Components)
# =========================================================

# A. No Skip Connections (Pure Encoder-Decoder)
create_experiment("structural", "no_skip_connections", {
    'd1 = Concatenate()([d1, s3])': 'd1 = d1 # Skip removed',
    'd2 = Concatenate()([d2, s2])': 'd2 = d2 # Skip removed',
    'd3 = Concatenate()([d3, s1])': 'd3 = d3 # Skip removed'
})

# B. No Batch Normalization
create_experiment("structural", "no_batchnorm", {
    "x = BatchNormalization()(x)": "x = x # BN removed"
})

# C. Simplified Conv Block (Double Conv -> Single Conv)
simplified_block = """
def conv_block(x, f):
    x = Conv2D(f, 3, padding="same", activation="relu")(x)
    return BatchNormalization()(x)
"""
# Note: This logic assumes your source file has the standard double-conv block
create_experiment("structural", "simplified_conv_block", {
    'def conv_block(x, f):': '# Original block replaced',
    'return BatchNormalization()(x)': 'return x # Placeholder', 
    '# Original block replaced': simplified_block
})

# D. No Decoder Upsampling (Bottleneck prediction)
create_experiment("structural", "no_decoder_upsampling", {
    'd1 = Conv2DTranspose': '# Decoder logic removed\n    # d1 = Conv2DTranspose',
    'outputs = Conv2D(n_classes, 1, activation="softmax")(d3)': 
    'outputs = Conv2D(n_classes, 1, activation="softmax", padding="same")(b1)\n    outputs = tf.keras.layers.UpSampling2D(size=(8, 8))(outputs)'
})

# =========================================================
# 3. QUANTITATIVE SWEEPS (Hyperparameters)
# =========================================================

# A. Filter Depth Sweep (16, 32, 128)
for size in [16, 32, 128]:
    create_experiment("filter_sweep", f"filters_{size}", {
        "conv_block(inputs, 64)": f"conv_block(inputs, {size})",
        "conv_block(p1, 128)": f"conv_block(p1, {size*2})",
        "conv_block(p2, 256)": f"conv_block(p2, {size*4})",
        "conv_block(p3, 512)": f"conv_block(p3, {size*8})",
        "Conv2DTranspose(256": f"Conv2DTranspose({size*4}",
        "Conv2DTranspose(128": f"Conv2DTranspose({size*2}",
        "Conv2DTranspose(64": f"Conv2DTranspose({size}",
        "conv_block(d1, 256)": f"conv_block(d1, {size*4})",
        "conv_block(d2, 128)": f"conv_block(d2, {size*2})",
        "conv_block(d3, 64)": f"conv_block(d3, {size})"
    })

# B. Kernel Size Variation (5x5 Context)
create_experiment("kernel_sweep", "kernel_5x5", {
    'Conv2D(f, 3,': 'Conv2D(f, 5,' 
})

# =========================================================
# 4. LOSS FUNCTION VARIATIONS
# =========================================================

# A. Dice Loss
create_experiment("loss_comparisons", "dice_loss_only", {
    'loss="categorical_crossentropy"': 'loss=dice_loss_function'
})

# B. Weighted Waste Bias (Class 1 weight = 5.0)
weighted_loss_code = """
def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        weights_tensor = tf.reduce_sum(weights * y_true, axis=-1)
        unweighted_losses = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(unweighted_losses * weights_tensor)
    return loss

# High importance for Class 1 (Waste)
weights = np.array([1.0] * N_CLASSES)
weights[1] = 5.0 
"""
create_experiment("loss_comparisons", "weighted_waste_bias", {
    '# Compile & Train': weighted_loss_code + '\n# Compile & Train',
    'loss="categorical_crossentropy"': 'loss=weighted_categorical_crossentropy(weights)'
})