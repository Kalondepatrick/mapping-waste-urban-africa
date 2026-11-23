# =============================================================================
# THRESHOLD SENSITIVITY ANALYSIS FOR U-NET WASTE OBJECT DETECTION
#
# For each probability threshold:
#   - Number of predicted waste objects
#   - TP, FP, FN
#   - Object precision, recall, F1
#   - Mean / SD / total predicted waste area
#   - Area error relative to manually digitised waste
#
# Then:
#   - Select threshold with highest object F1
#   - Export predicted objects, false positives, and false negatives
#
# IMPORTANT:
# The evaluation area must be exhaustively digitised for waste.
# Ideally provide TEST_AREA_PATH.
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape, box


from pathlib import Path

# ---------------------------------------------------------
# 1. PATHS & CONFIG
# ---------------------------------------------------------

BASE_PATH = (
    Path.home()
    / "GitHub"
    / "waste_spatio_temporal"
)

IMG_PATH = (
    BASE_PATH
    / "intermediaries"
    / "co-registration"
    / "October_2024_coregistered.tif"
)

MODEL_PATH = (
    BASE_PATH
    / "models"
    / "filters_16_march_finetuned_best.keras"
)

OUTPUT_DIR = (
    BASE_PATH
    / "outputs"
    / "October_2024"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "October_2024_predicted_classes.tif"
)

WASTE_SHP_PATH = (
    OUTPUT_DIR
    / "October_2024_waste_piles.shp"
)


# =============================================================================
# 2. PARAMETERS
# =============================================================================

THRESHOLDS = np.arange(
    0.10,
    0.96,
    0.05
)

# Remove tiny connected components
MIN_PREDICTED_AREA_M2 = 0.25

# Minimum fraction of reference pile that must overlap prediction
# to count as detected
MIN_REFERENCE_COVERAGE = 0.01

# 8-neighbour connectivity
STRUCTURE = np.ones(
    (3, 3),
    dtype=np.uint8
)


# =============================================================================
# 3. LOAD MANUAL WASTE
# =============================================================================

print("\nLoading manually digitised waste...")

manual = gpd.read_file(
    MANUAL_WASTE_PATH
)

manual = manual[
    manual.geometry.notna()
].copy()

manual = manual[
    ~manual.geometry.is_empty
].copy()

if len(manual) == 0:
    raise ValueError(
        "No polygons found in waste.shp"
    )

manual["pile_id"] = np.arange(
    1,
    len(manual) + 1
)

print(
    f"Manual waste polygons loaded: "
    f"{len(manual)}"
)


# =============================================================================
# 4. OPEN PROBABILITY RASTER
# =============================================================================

print("\nOpening probability raster...")

with rasterio.open(
    PROBABILITY_PATH
) as src:

    raster_crs = src.crs

    if raster_crs is None:
        raise ValueError(
            "Probability raster has no CRS."
        )

    manual = manual.to_crs(
        raster_crs
    )

    # ---------------------------------------------------------------------
    # Evaluation area
    # ---------------------------------------------------------------------

    if TEST_AREA_PATH.exists():

        print(
            "Using explicit test-area polygon:"
        )

        print(
            TEST_AREA_PATH
        )

        test_area = gpd.read_file(
            TEST_AREA_PATH
        )

        test_area = test_area.to_crs(
            raster_crs
        )

        # dissolve if multiple polygons
        test_geom = (
            test_area.geometry
            .union_all()
        )

    else:

        print(
            "\nWARNING:"
        )

        print(
            "test_area.shp not found."
        )

        print(
            "Using bounding box of manual "
            "waste polygons instead."
        )

        minx, miny, maxx, maxy = (
            manual.total_bounds
        )

        test_geom = box(
            minx,
            miny,
            maxx,
            maxy
        )

        test_area = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[test_geom],
            crs=raster_crs
        )

    # ---------------------------------------------------------------------
    # Crop raster to evaluation area
    # ---------------------------------------------------------------------

    minx, miny, maxx, maxy = (
        test_geom.bounds
    )

    window = from_bounds(
        minx,
        miny,
        maxx,
        maxy,
        transform=src.transform
    )

    window = (
        window
        .round_offsets()
        .round_lengths()
    )

    probability = src.read(
        1,
        window=window
    ).astype(
        np.float32
    )

    transform = src.window_transform(
        window
    )

    nodata = src.nodata


# =============================================================================
# 5. CLIP MANUAL WASTE TO TEST AREA
# =============================================================================

manual = gpd.clip(
    manual,
    test_area
)

manual = manual[
    manual.geometry.notna()
].copy()

manual = manual[
    ~manual.geometry.is_empty
].copy()

manual["manual_area_m2"] = (
    manual.geometry.area
)

manual_count = len(manual)

manual_total_area = (
    manual["manual_area_m2"]
    .sum()
)

manual_mean_area = (
    manual["manual_area_m2"]
    .mean()
)

manual_sd_area = (
    manual["manual_area_m2"]
    .std(ddof=1)
)

print("\nManual reference:")
print(
    f"  piles: "
    f"{manual_count}"
)

print(
    f"  total area: "
    f"{manual_total_area:.3f} m²"
)


# =============================================================================
# 6. RASTER INFORMATION
# =============================================================================

pixel_width = abs(
    transform.a
)

pixel_height = abs(
    transform.e
)

pixel_area_m2 = (
    pixel_width
    * pixel_height
)

minimum_pixels = max(
    1,
    int(
        np.ceil(
            MIN_PREDICTED_AREA_M2
            / pixel_area_m2
        )
    )
)

print(
    f"\nPixel area: "
    f"{pixel_area_m2:.6f} m²"
)

print(
    f"Minimum predicted object: "
    f"{MIN_PREDICTED_AREA_M2} m² "
    f"({minimum_pixels} pixels)"
)


# =============================================================================
# 7. VALID RASTER MASK
# =============================================================================

valid = np.isfinite(
    probability
)

if nodata is not None:

    valid &= (
        probability != nodata
    )


# =============================================================================
# 8. FUNCTION: CREATE PREDICTED OBJECTS
# =============================================================================

def make_predicted_objects(
    threshold
):

    binary = (
        valid
        & (
            probability
            >= threshold
        )
    )

    labels, n_objects = (
        ndimage.label(
            binary,
            structure=STRUCTURE
        )
    )

    if n_objects == 0:

        return gpd.GeoDataFrame(
            columns=[
                "patch_id",
                "predicted_area_m2",
                "geometry"
            ],
            crs=raster_crs
        )

    # ---------------------------------------------------------------------
    # Remove tiny objects efficiently
    # ---------------------------------------------------------------------

    counts = np.bincount(
        labels.ravel()
    )

    keep = (
        counts
        >= minimum_pixels
    )

    keep[0] = False

    keep_mask = keep[
        labels
    ]

    labels_filtered, n_filtered = (
        ndimage.label(
            keep_mask,
            structure=STRUCTURE
        )
    )

    if n_filtered == 0:

        return gpd.GeoDataFrame(
            columns=[
                "patch_id",
                "predicted_area_m2",
                "geometry"
            ],
            crs=raster_crs
        )

    # ---------------------------------------------------------------------
    # Vectorise
    # ---------------------------------------------------------------------

    records = []

    for geom, value in shapes(
        labels_filtered.astype(
            np.int32
        ),
        mask=(
            labels_filtered > 0
        ),
        transform=transform
    ):

        patch_id = int(
            value
        )

        if patch_id == 0:
            continue

        records.append(
            {
                "patch_id":
                    patch_id,
                "geometry":
                    shape(geom)
            }
        )

    if len(records) == 0:

        return gpd.GeoDataFrame(
            columns=[
                "patch_id",
                "predicted_area_m2",
                "geometry"
            ],
            crs=raster_crs
        )

    predicted = (
        gpd.GeoDataFrame(
            records,
            crs=raster_crs
        )
    )

    predicted = predicted.dissolve(
        by="patch_id",
        as_index=False
    )

    predicted = gpd.clip(
        predicted,
        test_area
    )

    predicted = predicted[
        predicted.geometry.notna()
    ].copy()

    predicted = predicted[
        ~predicted.geometry.is_empty
    ].copy()

    predicted[
        "predicted_area_m2"
    ] = (
        predicted.geometry.area
    )

    # Clipping at the test-area boundary can
    # create tiny remnants.
    predicted = predicted[
        predicted[
            "predicted_area_m2"
        ]
        >= MIN_PREDICTED_AREA_M2
    ].copy()

    return predicted


# =============================================================================
# 9. FUNCTION: MATCH REFERENCE AND PREDICTION
# =============================================================================

def match_objects(
    manual_gdf,
    predicted_gdf
):

    if len(
        predicted_gdf
    ) == 0:

        return pd.DataFrame()

    candidates = []

    pred_sindex = (
        predicted_gdf.sindex
    )

    for _, ref in (
        manual_gdf.iterrows()
    ):

        ref_id = int(
            ref["pile_id"]
        )

        ref_geom = (
            ref.geometry
        )

        ref_area = float(
            ref[
                "manual_area_m2"
            ]
        )

        possible_ids = list(
            pred_sindex.intersection(
                ref_geom.bounds
            )
        )

        for pos in possible_ids:

            pred = (
                predicted_gdf
                .iloc[pos]
            )

            pred_geom = (
                pred.geometry
            )

            if not (
                ref_geom
                .intersects(
                    pred_geom
                )
            ):
                continue

            intersection = (
                ref_geom.intersection(
                    pred_geom
                )
            )

            intersection_area = (
                intersection.area
            )

            if (
                intersection_area
                <= 0
            ):
                continue

            pred_area = float(
                pred[
                    "predicted_area_m2"
                ]
            )

            reference_coverage = (
                intersection_area
                / ref_area
            )

            if (
                reference_coverage
                <
                MIN_REFERENCE_COVERAGE
            ):
                continue

            union_area = (
                ref_area
                + pred_area
                - intersection_area
            )

            iou = (
                intersection_area
                / union_area
                if union_area > 0
                else 0
            )

            candidates.append(
                {
                    "pile_id":
                        ref_id,

                    "patch_id":
                        int(
                            pred[
                                "patch_id"
                            ]
                        ),

                    "reference_area_m2":
                        ref_area,

                    "predicted_area_m2":
                        pred_area,

                    "intersection_area_m2":
                        intersection_area,

                    "reference_coverage":
                        reference_coverage,

                    "iou":
                        iou
                }
            )

    if len(candidates) == 0:

        return pd.DataFrame()

    candidates = (
        pd.DataFrame(
            candidates
        )
        .sort_values(
            [
                "iou",
                "reference_coverage",
                "intersection_area_m2"
            ],
            ascending=False
        )
    )

    # ---------------------------------------------------------------------
    # Greedy one-to-one matching
    # ---------------------------------------------------------------------

    matches = []

    used_ref = set()
    used_pred = set()

    for _, row in (
        candidates.iterrows()
    ):

        pile_id = int(
            row["pile_id"]
        )

        patch_id = int(
            row["patch_id"]
        )

        if pile_id in used_ref:
            continue

        if patch_id in used_pred:
            continue

        matches.append(
            row.to_dict()
        )

        used_ref.add(
            pile_id
        )

        used_pred.add(
            patch_id
        )

    return pd.DataFrame(
        matches
    )


# =============================================================================
# 10. RUN ALL THRESHOLDS
# =============================================================================

results = []

predictions_by_threshold = {}
matches_by_threshold = {}

print("\n")
print("=" * 75)
print("RUNNING THRESHOLD ANALYSIS")
print("=" * 75)


for threshold in THRESHOLDS:

    threshold = float(
        np.round(
            threshold,
            2
        )
    )

    print(
        f"\nThreshold: "
        f"{threshold:.2f}"
    )

    predicted = (
        make_predicted_objects(
            threshold
        )
    )

    matches = (
        match_objects(
            manual,
            predicted
        )
    )

    predicted_count = len(
        predicted
    )

    TP = len(
        matches
    )

    FP = (
        predicted_count
        - TP
    )

    FN = (
        manual_count
        - TP
    )

    precision = (
        TP
        / (TP + FP)
        if (TP + FP) > 0
        else np.nan
    )

    recall = (
        TP
        / (TP + FN)
        if (TP + FN) > 0
        else np.nan
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            np.isfinite(
                precision
            )
            and np.isfinite(
                recall
            )
            and (
                precision
                + recall
            ) > 0
        )
        else np.nan
    )

    if predicted_count > 0:

        predicted_total_area = (
            predicted[
                "predicted_area_m2"
            ].sum()
        )

        predicted_mean_area = (
            predicted[
                "predicted_area_m2"
            ].mean()
        )

        predicted_sd_area = (
            predicted[
                "predicted_area_m2"
            ].std(
                ddof=1
            )
            if predicted_count > 1
            else np.nan
        )

    else:

        predicted_total_area = 0.0
        predicted_mean_area = np.nan
        predicted_sd_area = np.nan

    signed_area_error = (
        predicted_total_area
        - manual_total_area
    )

    percent_area_error = (
        100
        * signed_area_error
        / manual_total_area
        if manual_total_area > 0
        else np.nan
    )

    results.append(
        {
            "threshold":
                threshold,

            "manual_piles":
                manual_count,

            "predicted_piles":
                predicted_count,

            "TP":
                TP,

            "FP":
                FP,

            "FN":
                FN,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "manual_mean_area_m2":
                manual_mean_area,

            "manual_sd_area_m2":
                manual_sd_area,

            "manual_total_area_m2":
                manual_total_area,

            "predicted_mean_area_m2":
                predicted_mean_area,

            "predicted_sd_area_m2":
                predicted_sd_area,

            "predicted_total_area_m2":
                predicted_total_area,

            "signed_area_error_m2":
                signed_area_error,

            "percent_area_error":
                percent_area_error
        }
    )

    predictions_by_threshold[
        threshold
    ] = predicted

    matches_by_threshold[
        threshold
    ] = matches

    print(
        f"  predicted piles: "
        f"{predicted_count}"
    )

    print(
        f"  TP / FP / FN: "
        f"{TP} / {FP} / {FN}"
    )

    print(
        f"  precision: "
        f"{precision:.3f}"
    )

    print(
        f"  recall:    "
        f"{recall:.3f}"
    )

    print(
        f"  F1:        "
        f"{f1:.3f}"
    )

    print(
        f"  area:      "
        f"{predicted_total_area:.1f} m²"
    )


# =============================================================================
# 11. SAVE THRESHOLD TABLE
# =============================================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_DIR
    / "threshold_metrics.csv",
    index=False
)


# =============================================================================
# 12. IDENTIFY BEST F1 THRESHOLD
# =============================================================================

valid_f1 = (
    results_df[
        "f1"
    ].notna()
)

if not valid_f1.any():

    raise RuntimeError(
        "No valid F1 values were produced."
    )

best_row = (
    results_df.loc[
        valid_f1
    ]
    .sort_values(
        "f1",
        ascending=False
    )
    .iloc[0]
)

best_threshold = float(
    best_row[
        "threshold"
    ]
)

print("\n")
print("=" * 75)
print("BEST OBJECT-F1 THRESHOLD")
print("=" * 75)

print(
    f"Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Precision: "
    f"{best_row['precision']:.3f}"
)

print(
    f"Recall:    "
    f"{best_row['recall']:.3f}"
)

print(
    f"F1:        "
    f"{best_row['f1']:.3f}"
)

print(
    f"Predicted piles: "
    f"{int(best_row['predicted_piles'])}"
)

print(
    f"Manual piles:    "
    f"{manual_count}"
)

print(
    f"Predicted area:   "
    f"{best_row['predicted_total_area_m2']:.2f} m²"
)

print(
    f"Manual area:      "
    f"{manual_total_area:.2f} m²"
)


# =============================================================================
# 13. EXPORT BEST-THRESHOLD FP/FN LAYERS
# =============================================================================

best_predicted = (
    predictions_by_threshold[
        best_threshold
    ].copy()
)

best_matches = (
    matches_by_threshold[
        best_threshold
    ].copy()
)

if len(best_matches) > 0:

    matched_ref_ids = set(
        best_matches[
            "pile_id"
        ].astype(int)
    )

    matched_pred_ids = set(
        best_matches[
            "patch_id"
        ].astype(int)
    )

else:

    matched_ref_ids = set()
    matched_pred_ids = set()


# False negatives
false_negatives = (
    manual[
        ~manual[
            "pile_id"
        ].isin(
            matched_ref_ids
        )
    ]
    .copy()
)

false_negatives[
    "error_type"
] = "false_negative"


# False positives
false_positives = (
    best_predicted[
        ~best_predicted[
            "patch_id"
        ].isin(
            matched_pred_ids
        )
    ]
    .copy()
)

false_positives[
    "error_type"
] = "false_positive"


# True positives / matched predictions
true_positive_predictions = (
    best_predicted[
        best_predicted[
            "patch_id"
        ].isin(
            matched_pred_ids
        )
    ]
    .copy()
)

true_positive_predictions[
    "error_type"
] = "true_positive"


# =============================================================================
# 14. SAVE VECTOR OUTPUTS
# =============================================================================

best_predicted.to_file(
    OUTPUT_DIR
    / "best_threshold_predictions.gpkg",
    layer="predictions",
    driver="GPKG"
)

false_positives.to_file(
    OUTPUT_DIR
    / "best_threshold_false_positives.gpkg",
    layer="false_positives",
    driver="GPKG"
)

false_negatives.to_file(
    OUTPUT_DIR
    / "best_threshold_false_negatives.gpkg",
    layer="false_negatives",
    driver="GPKG"
)

true_positive_predictions.to_file(
    OUTPUT_DIR
    / "best_threshold_true_positives.gpkg",
    layer="true_positives",
    driver="GPKG"
)

manual.to_file(
    OUTPUT_DIR
    / "manual_reference.gpkg",
    layer="manual_waste",
    driver="GPKG"
)

test_area.to_file(
    OUTPUT_DIR
    / "evaluation_area.gpkg",
    layer="evaluation_area",
    driver="GPKG"
)


if len(best_matches) > 0:

    best_matches.to_csv(
        OUTPUT_DIR
        / "best_threshold_matches.csv",
        index=False
    )


# =============================================================================
# 15. ALSO IDENTIFY SOME USEFUL ALTERNATIVE THRESHOLDS
# =============================================================================

# Highest recall threshold with precision >= 0.20
subset = results_df[
    results_df[
        "precision"
    ] >= 0.20
]

if len(subset) > 0:

    best_recall_given_precision = (
        subset
        .sort_values(
            [
                "recall",
                "f1"
            ],
            ascending=False
        )
        .iloc[0]
    )

    print("\n")
    print(
        "Highest recall with "
        "precision >= 0.20:"
    )

    print(
        f"  threshold: "
        f"{best_recall_given_precision['threshold']:.2f}"
    )

    print(
        f"  precision: "
        f"{best_recall_given_precision['precision']:.3f}"
    )

    print(
        f"  recall: "
        f"{best_recall_given_precision['recall']:.3f}"
    )


# Threshold with smallest absolute total-area error
area_row = (
    results_df.assign(
        abs_area_error=lambda x:
        abs(
            x[
                "signed_area_error_m2"
            ]
        )
    )
    .sort_values(
        "abs_area_error"
    )
    .iloc[0]
)

print("\nThreshold with smallest total-area error:")

print(
    f"  threshold: "
    f"{area_row['threshold']:.2f}"
)

print(
    f"  predicted area: "
    f"{area_row['predicted_total_area_m2']:.2f} m²"
)

print(
    f"  area error: "
    f"{area_row['signed_area_error_m2']:.2f} m²"
)

print(
    f"  precision: "
    f"{area_row['precision']:.3f}"
)

print(
    f"  recall: "
    f"{area_row['recall']:.3f}"
)


# =============================================================================
# 16. FINAL SUMMARY
# =============================================================================

print("\n")
print("=" * 75)
print("DONE")
print("=" * 75)

print(
    "\nResults written to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nMain table:"
)

print(
    "  threshold_metrics.csv"
)

print(
    "\nQGIS layers for inspecting errors:"
)

print(
    "  best_threshold_false_positives.gpkg"
)

print(
    "  best_threshold_false_negatives.gpkg"
)

print(
    "  best_threshold_true_positives.gpkg"
)

print(
    "  best_threshold_predictions.gpkg"
)