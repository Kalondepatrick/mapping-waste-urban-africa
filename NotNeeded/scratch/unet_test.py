import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

# =====================================================================
# 1. THRESHOLD EVALUATION FUNCTION
# =====================================================================


def evaluate_f1_thresholds(y_true, y_pred_raw):
    """Calculates PR curve and returns optimal probability threshold for max F1.

    :param y_true: Ground truth binary masks (0 or 1)
    :param y_pred_raw: Raw model outputs (logits or probabilities)
    """
    # Flatten arrays for pixel-wise metrics
    y_true = np.asarray(y_true).ravel()
    y_pred_raw = np.asarray(y_pred_raw).ravel()

    # Apply Sigmoid if outputs are raw unscaled logits
    if y_pred_raw.min() < 0 or y_pred_raw.max() > 1:
        probs = 1 / (1 + np.exp(-y_pred_raw))
    else:
        probs = y_pred_raw

    # Compute Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    ap_score = average_precision_score(y_true, probs)

    # Note: precision and recall have size N; thresholds has size N-1.
    # We slice precision and recall to N-1 to match thresholds cleanly.
    p_matched = precision[:-1]
    r_matched = recall[:-1]

    # Calculate F1 scores safely
    f1_scores = (2 * p_matched * r_matched) / (p_matched + r_matched + 1e-7)

    # Pick index with highest F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_p = p_matched[best_idx]
    best_r = r_matched[best_idx]
    best_f1 = f1_scores[best_idx]

    # Print summary
    print("\n" + "=" * 45)
    print("      OPTIMAL THRESHOLD SEARCH RESULTS      ")
    print("=" * 45)
    print(f"Average Precision (AP)      : {ap_score:.4f}")
    print(f"Optimal Decision Threshold  : {best_threshold:.4f}")
    print(f"Precision at Best Threshold : {best_p:.4f}")
    print(f"Recall at Best Threshold    : {best_r:.4f}")
    print(f"Max F1-Score                : {best_f1:.4f}")
    print("=" * 45 + "\n")

    # Plot PR Curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        color="blue",
        lw=2,
        label=f"Waste Class PR (AP = {ap_score:.4f})",
    )
    plt.plot(
        best_r,
        best_p,
        "ro",
        markersize=8,
        label=f"Best Threshold = {best_threshold:.4f}",
    )
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curve (Waste Class)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    # Save plot
    save_path = "pr_curve_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to: {os.path.abspath(save_path)}")

    return best_threshold


# =====================================================================
# 2. RUN SCRIPT
# =====================================================================

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # OPTION A: Load your pre-saved evaluation numpy arrays (if available)
    # -----------------------------------------------------------------
    # y_true = np.load("y_true.npy")
    # y_pred = np.load("y_pred.npy")

    # -----------------------------------------------------------------
    # OPTION B: Dummy test data (Remove this block when connecting your PyTorch model)
    # -----------------------------------------------------------------
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=(100000,), p=[0.95, 0.05])
    y_pred = np.random.normal(loc=-2.0, scale=1.5, size=(100000,))
    y_pred[y_true == 1] += 2.0  # simulate U-Net waste detection

    # Execute search
    optimal_threshold = evaluate_f1_thresholds(y_true, y_pred)

    # Use 'optimal_threshold' to binarize future U-Net outputs:
    # binary_mask = (y_pred_probs >= optimal_threshold).astype(np.uint8)