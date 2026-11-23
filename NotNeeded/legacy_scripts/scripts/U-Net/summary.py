import os
import numpy as np
import rasterio

# -----------------------------
# Configuration
# -----------------------------
MASK_DIR = "data/tiled_data/masks"

def summarize_mask_classes(mask_dir):
    # Filter for .tif files
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
    
    if not mask_files:
        print(f"No .tif files found in {mask_dir}")
        return

    unique_classes_global = set()
    total_files = len(mask_files)
    
    print(f"Scanning {total_files} masks...")

    for idx, filename in enumerate(mask_files):
        file_path = os.path.join(mask_dir, filename)
        
        with rasterio.open(file_path) as src:
            # Read the first band
            mask_data = src.read(1)
            # Find unique values in this specific file
            unique_in_file = np.unique(mask_data)
            # Add to the global set
            unique_classes_global.update(unique_in_file)
        
        # Optional progress pulse
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_files} files...")

    # Final Summary
    sorted_classes = sorted(list(unique_classes_global))
    
    print("\n--- Mask Class Summary ---")
    print(f"Total files analyzed: {total_files}")
    print(f"Unique class values found: {sorted_classes}")
    print(f"Total number of unique classes: {len(sorted_classes)}")
    print("--------------------------")

    return sorted_classes

if __name__ == "__main__":
    classes = summarize_mask_classes(MASK_DIR)