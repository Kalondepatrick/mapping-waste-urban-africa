import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import rasterio
from rasterio.windows import Window
from pathlib import Path
import gc


# =============================================================================
# 1. PATHS
# =============================================================================

BASE_PATH = (
    Path.home()
    / "Documents"
    / "GitHub"
    / "mapping-waste-urban-africa"
)

IMG_PATH = (
    BASE_PATH
    / "data"
    / "optical.tiff"
)

MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_march_finetuned_best.keras"
)

OUTPUT_PATH = (
    BASE_PATH
    / "outputs"
    / "waste_probability_full_seamfree.tif"
)


# =============================================================================
# 2. INFERENCE SETTINGS
# =============================================================================

# Model input size
INPUT_SIZE = 256

# Only retain the central 128 x 128 pixels from each prediction.
#
# This means predictions made close to the edge of the input tile
# are discarded.
OUTPUT_SIZE = 128

# Context discarded from every side:
#
# (256 - 128) / 2 = 64 pixels
MARGIN = (INPUT_SIZE - OUTPUT_SIZE) // 2

BATCH_SIZE = 32

# Waste is channel 0 in the model output
WASTE_CLASS_INDEX = 0


# =============================================================================
# 3. CHECK PATHS
# =============================================================================

if not IMG_PATH.exists():
    raise FileNotFoundError(
        f"Input image not found:\n{IMG_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 4. LOAD MODEL
# =============================================================================

print("=" * 75)
print("LOADING MODEL")
print("=" * 75)

tf.keras.backend.clear_session()

MODEL = load_model(
    MODEL_PATH,
    compile=False
)

print(f"Model: {MODEL_PATH.name}")
print(f"Input tile: {INPUT_SIZE} x {INPUT_SIZE}")
print(f"Retained centre: {OUTPUT_SIZE} x {OUTPUT_SIZE}")
print(f"Discarded border: {MARGIN} pixels per side")


# =============================================================================
# 5. FUNCTION FOR RUNNING A BATCH
# =============================================================================

def process_batch(
    batch_tiles,
    batch_metadata,
    dst
):

    if len(batch_tiles) == 0:
        return

    batch_array = np.stack(
        batch_tiles,
        axis=0
    ).astype(np.float32)

    preds = MODEL.predict(
        batch_array,
        verbose=0
    )

    # ---------------------------------------------------------
    # Check model output
    # ---------------------------------------------------------

    if preds.ndim != 4:
        raise ValueError(
            f"Unexpected model output shape: {preds.shape}"
        )

    if preds.shape[-1] == 1:

        # Single-channel sigmoid model
        waste_probs = preds[..., 0]

    else:

        # Multi-class softmax model
        waste_probs = preds[
            ...,
            WASTE_CLASS_INDEX
        ]

    # ---------------------------------------------------------
    # Write only CENTRAL prediction of each tile
    # ---------------------------------------------------------

    for waste_prob, meta in zip(
        waste_probs,
        batch_metadata
    ):

        (
            out_row,
            out_col,
            out_h,
            out_w,
            valid_mask
        ) = meta

        # Central 128 x 128 prediction
        central_prob = waste_prob[
            MARGIN:MARGIN + OUTPUT_SIZE,
            MARGIN:MARGIN + OUTPUT_SIZE
        ]

        # Crop at right/bottom image boundaries
        central_prob = central_prob[
            :out_h,
            :out_w
        ].astype(
            np.float32
        )

        # Initialise with nodata
        output = np.full(
            (out_h, out_w),
            -1.0,
            dtype=np.float32
        )

        output[
            valid_mask
        ] = central_prob[
            valid_mask
        ]

        dst.write(
            output,
            1,
            window=Window(
                out_col,
                out_row,
                out_w,
                out_h
            )
        )


# =============================================================================
# 6. RUN INFERENCE
# =============================================================================

print("\n")
print("=" * 75)
print("CENTRE-CROP WHOLE-IMAGE INFERENCE")
print("=" * 75)

with rasterio.open(
    IMG_PATH
) as src:

    H = src.height
    W = src.width

    print(
        f"Image dimensions: "
        f"{W:,} x {H:,} pixels"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        "\nEach 128 x 128 output region is predicted "
        "from a 256 x 256 image neighbourhood."
    )

    profile = src.profile.copy()

    profile.update(
        {
            "count": 1,
            "dtype": "float32",
            "nodata": -1.0,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "YES"
        }
    )

    # Determine what to use outside raster boundary
    input_nodata = (
        src.nodata
        if src.nodata is not None
        else 0
    )

    # Number of output rows to process
    n_output_rows = int(
        np.ceil(
            H / OUTPUT_SIZE
        )
    )

    with rasterio.open(
        OUTPUT_PATH,
        "w",
        **profile
    ) as dst:

        batch_tiles = []
        batch_metadata = []

        for row_index, out_row in enumerate(
            range(
                0,
                H,
                OUTPUT_SIZE
            ),
            start=1
        ):

            out_h = min(
                OUTPUT_SIZE,
                H - out_row
            )

            for out_col in range(
                0,
                W,
                OUTPUT_SIZE
            ):

                out_w = min(
                    OUTPUT_SIZE,
                    W - out_col
                )

                # ---------------------------------------------------------
                # The output region starts at:
                #
                # out_row, out_col
                #
                # but we read 64 pixels around it on every side.
                # ---------------------------------------------------------

                input_row = (
                    out_row
                    - MARGIN
                )

                input_col = (
                    out_col
                    - MARGIN
                )

                input_window = Window(
                    input_col,
                    input_row,
                    INPUT_SIZE,
                    INPUT_SIZE
                )

                # ---------------------------------------------------------
                # Always read exactly 256 x 256.
                #
                # boundless=True pads only at the OUTER edge of the
                # complete orthomosaic.
                # ---------------------------------------------------------

                tile = src.read(
                    [1, 2, 3],
                    window=input_window,
                    boundless=True,
                    fill_value=input_nodata
                )

                if tile.shape != (
                    3,
                    INPUT_SIZE,
                    INPUT_SIZE
                ):
                    raise RuntimeError(
                        "Unexpected tile shape: "
                        f"{tile.shape}"
                    )

                # ---------------------------------------------------------
                # Determine which pixels in the CENTRAL output region
                # represent valid imagery.
                # ---------------------------------------------------------

                central_input = tile[
                    :,
                    MARGIN:MARGIN + out_h,
                    MARGIN:MARGIN + out_w
                ]

                if src.nodata is not None:

                    valid_mask = ~np.all(
                        central_input
                        == src.nodata,
                        axis=0
                    )

                else:

                    # No explicit nodata value supplied by raster
                    valid_mask = np.ones(
                        (
                            out_h,
                            out_w
                        ),
                        dtype=bool
                    )

                # ---------------------------------------------------------
                # Normalisation
                # ---------------------------------------------------------

                tile_input = np.moveaxis(
                    tile,
                    0,
                    -1
                )

                tile_input = (
                    tile_input.astype(
                        np.float32
                    )
                    / 255.0
                )

                batch_tiles.append(
                    tile_input
                )

                batch_metadata.append(
                    (
                        out_row,
                        out_col,
                        out_h,
                        out_w,
                        valid_mask
                    )
                )

                # ---------------------------------------------------------
                # Run batch
                # ---------------------------------------------------------

                if (
                    len(batch_tiles)
                    >= BATCH_SIZE
                ):

                    process_batch(
                        batch_tiles,
                        batch_metadata,
                        dst
                    )

                    batch_tiles = []
                    batch_metadata = []

            # -------------------------------------------------------------
            # Progress
            # -------------------------------------------------------------

            completed_y = min(
                out_row + OUTPUT_SIZE,
                H
            )

            pct = (
                100
                * completed_y
                / H
            )

            print(
                f"\rRows processed: "
                f"{completed_y:,} / {H:,} "
                f"({pct:5.1f}%)",
                end="",
                flush=True
            )

            gc.collect()

        # -------------------------------------------------------------
        # Remaining tiles
        # -------------------------------------------------------------

        if batch_tiles:

            process_batch(
                batch_tiles,
                batch_metadata,
                dst
            )


# =============================================================================
# 7. FINISHED
# =============================================================================

print("\n")
print("\n" + "=" * 75)
print("INFERENCE COMPLETE")
print("=" * 75)

print(
    f"\nProbability raster saved to:\n"
    f"{OUTPUT_PATH}"
)

print(
    "\nThis raster should be used for the "
    "threshold/object analysis instead of the old "
    "waste_probability_full.tif."
)