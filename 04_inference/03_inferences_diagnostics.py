from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.features import shapes
from shapely.geometry import shape, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union


# ============================================================
# 1. PATHS
# ============================================================

BASE_PATH = (
    Path.home()
    / "Documents"
    / "GitHub"
    / "mapping-waste-urban-africa"
)

# CHANGE THIS if another probability raster is your final/current output
PROB_RASTER = (
    BASE_PATH
    / "outputs"
    / "waste_probability_full_seamfree.tif"
)

REFERENCE_PATH = (
    BASE_PATH
    / "data"
    / "waste"
    / "waste_outlines.geojson"
)

OUTPUT_DIR = (
    BASE_PATH
    / "results"
    / "whole_area_probability_sweep"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

MIN_AREAS_M2 = [
    0.25,
    0.50,
    1.00,
    2.00,
]

# Same general matching rule used previously
MIN_REFERENCE_COVERAGE = 0.01

# Optional operating criterion
TARGET_RECALL = 0.90


# ============================================================
# 3. HELPERS
# ============================================================

def safe_divide(a, b):
    return a / b if b > 0 else np.nan


def get_polygon_parts(geom):

    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)

    if isinstance(geom, GeometryCollection):

        parts = []

        for g in geom.geoms:
            parts.extend(
                get_polygon_parts(g)
            )

        return parts

    return []


def build_reference_units(reference_gdf):

    union_geom = unary_union(
        reference_gdf.geometry
    )

    parts = get_polygon_parts(
        union_geom
    )

    units = gpd.GeoDataFrame(
        {
            "reference_id": range(
                1,
                len(parts) + 1
            )
        },
        geometry=parts,
        crs=reference_gdf.crs,
    )

    units["reference_area_m2"] = (
        units.geometry.area
    )

    return units


def evaluate_objects(
    predictions,
    references
):

    if len(predictions) == 0:

        return {
            "predicted_objects": 0,
            "matched_predictions": 0,
            "reference_units_detected": 0,
            "reference_units_total": len(references),
            "unmatched_predictions": 0,
            "missed_reference_units": len(references),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    ref_index = references.sindex

    matches = []

    for pred_idx, pred_row in predictions.iterrows():

        pred_geom = pred_row.geometry

        if pred_geom is None or pred_geom.is_empty:
            continue

        candidates = list(
            ref_index.query(
                pred_geom,
                predicate="intersects"
            )
        )

        for ref_idx in candidates:

            ref_geom = references.iloc[
                ref_idx
            ].geometry

            intersection = (
                pred_geom.intersection(
                    ref_geom
                )
            )

            intersection_area = (
                intersection.area
            )

            if intersection_area <= 0:
                continue

            pred_area = pred_geom.area
            ref_area = ref_geom.area

            reference_coverage = (
                intersection_area / ref_area
                if ref_area > 0
                else 0.0
            )

            union_area = (
                pred_area
                + ref_area
                - intersection_area
            )

            iou = (
                intersection_area / union_area
                if union_area > 0
                else 0.0
            )

            if (
                reference_coverage
                >= MIN_REFERENCE_COVERAGE
            ):

                matches.append(
                    {
                        "pred_idx": pred_idx,
                        "ref_idx": ref_idx,
                        "iou": iou,
                        "reference_coverage":
                            reference_coverage,
                        "intersection_area":
                            intersection_area,
                    }
                )

    # Highest quality matches first
    matches.sort(
        key=lambda x: (
            x["iou"],
            x["reference_coverage"],
            x["intersection_area"],
        ),
        reverse=True,
    )

    used_predictions = set()
    used_references = set()

    accepted = []

    for match in matches:

        p = match["pred_idx"]
        r = match["ref_idx"]

        if p in used_predictions:
            continue

        if r in used_references:
            continue

        used_predictions.add(p)
        used_references.add(r)

        accepted.append(match)

    tp = len(accepted)

    fp = (
        len(predictions)
        - tp
    )

    fn = (
        len(references)
        - tp
    )

    precision = safe_divide(
        tp,
        tp + fp
    )

    recall = safe_divide(
        tp,
        tp + fn
    )

    f1 = (
        safe_divide(
            2 * precision * recall,
            precision + recall
        )
        if precision + recall > 0
        else 0.0
    )

    return {
        "predicted_objects":
            len(predictions),

        "matched_predictions":
            tp,

        "reference_units_detected":
            tp,

        "reference_units_total":
            len(references),

        "unmatched_predictions":
            fp,

        "missed_reference_units":
            fn,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,
    }


# ============================================================
# 4. OPEN PROBABILITY RASTER
# ============================================================

print(
    f"\nProbability raster:\n{PROB_RASTER}"
)

if not PROB_RASTER.exists():
    raise FileNotFoundError(
        PROB_RASTER
    )


with rasterio.open(
    PROB_RASTER
) as src:

    if src.count != 1:
        raise ValueError(
            "Expected a single-band "
            "waste probability raster."
        )

    if src.crs is None:
        raise ValueError(
            "Probability raster has no CRS."
        )

    raster_crs = src.crs

    transform = src.transform

    nodata = src.nodata

    print(
        f"Raster dimensions: "
        f"{src.width:,} x "
        f"{src.height:,}"
    )

    print(
        f"CRS: {src.crs}"
    )

    print(
        "Reading probability raster..."
    )

    probability = src.read(1)


# ============================================================
# 5. CHECK PROBABILITY RANGE
# ============================================================

valid = np.isfinite(
    probability
)

if nodata is not None:
    valid &= (
        probability != nodata
    )


print(
    "\nProbability statistics"
)

print(
    "Minimum:",
    float(
        np.nanmin(
            probability[valid]
        )
    )
)

print(
    "Maximum:",
    float(
        np.nanmax(
            probability[valid]
        )
    )
)

print(
    "Mean:",
    float(
        np.nanmean(
            probability[valid]
        )
    )
)


# ============================================================
# 6. REFERENCE WASTE
# ============================================================

print(
    "\nReading reference waste..."
)

reference = gpd.read_file(
    REFERENCE_PATH
)

reference = reference.to_crs(
    raster_crs
)

reference_units = (
    build_reference_units(
        reference
    )
)

print(
    f"Reference source features: "
    f"{len(reference):,}"
)

print(
    f"Contiguous reference units: "
    f"{len(reference_units):,}"
)


# ============================================================
# 7. THRESHOLD SWEEP
# ============================================================

results = []

all_candidates = {}


for threshold in THRESHOLDS:

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"THRESHOLD = {threshold:.2f}"
    )

    print(
        "=" * 70
    )

    binary = (
        valid
        & (
            probability
            >= threshold
        )
    )

    print(
        "Polygonising..."
    )

    records = []

    object_id = 0

    # Polygonise all connected
    # thresholded regions once.
    for geom, value in shapes(
        binary.astype(
            np.uint8
        ),
        mask=binary,
        transform=transform,
        connectivity=8,
    ):

        if int(value) != 1:
            continue

        polygon = shape(
            geom
        )

        if (
            polygon is None
            or polygon.is_empty
        ):
            continue

        area_m2 = float(
            polygon.area
        )

        # Do not retain objects smaller than
        # the smallest area threshold.
        if (
            area_m2
            < min(
                MIN_AREAS_M2
            )
        ):
            continue

        object_id += 1

        records.append(
            {
                "object_id":
                    object_id,

                "threshold":
                    threshold,

                "area_m2":
                    area_m2,

                "geometry":
                    polygon,
            }
        )

    candidates = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=raster_crs,
    )

    print(
        f"Objects >= "
        f"{min(MIN_AREAS_M2)} m²: "
        f"{len(candidates):,}"
    )

    all_candidates[
        threshold
    ] = candidates


    # ------------------------------------
    # Evaluate each area criterion
    # ------------------------------------

    for min_area in MIN_AREAS_M2:

        selected = (
            candidates[
                candidates[
                    "area_m2"
                ]
                >= min_area
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        metrics = (
            evaluate_objects(
                selected,
                reference_units,
            )
        )

        row = {
            "threshold":
                threshold,

            "min_area_m2":
                min_area,

            **metrics,
        }

        results.append(
            row
        )

        print(
            f"\nMin area "
            f"{min_area:.2f} m²"
        )

        print(
            f"  Predicted objects : "
            f"{metrics['predicted_objects']:,}"
        )

        print(
            f"  References detected: "
            f"{metrics['reference_units_detected']}"
            f"/"
            f"{metrics['reference_units_total']}"
        )

        print(
            f"  Precision : "
            f"{metrics['precision']:.6f}"
        )

        print(
            f"  Recall    : "
            f"{metrics['recall']:.6f}"
        )

        print(
            f"  F1        : "
            f"{metrics['f1']:.6f}"
        )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

RESULTS_CSV = (
    OUTPUT_DIR
    / "threshold_area_sweep.csv"
)

results_df.to_csv(
    RESULTS_CSV,
    index=False
)


# Sort table for easier inspection
display_cols = [
    "threshold",
    "min_area_m2",
    "predicted_objects",
    "reference_units_detected",
    "reference_units_total",
    "precision",
    "recall",
    "f1",
]

print(
    "\n\n"
    + "=" * 70
)

print(
    "FULL RESULTS"
)

print(
    "=" * 70
)

print(
    results_df[
        display_cols
    ].to_string(
        index=False
    )
)


# ============================================================
# 9. SELECT OPERATING POINT
# ============================================================

eligible = (
    results_df[
        results_df[
            "recall"
        ]
        >= TARGET_RECALL
    ]
    .copy()
)


if len(eligible) > 0:

    # Main objective:
    # fewest objects requiring review.
    #
    # Tie-break:
    # higher precision.
    best = (
        eligible
        .sort_values(
            by=[
                "predicted_objects",
                "precision",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .iloc[0]
    )

else:

    # If no configuration reaches
    # target recall, retain the
    # configuration with highest recall,
    # then fewest candidates.
    best = (
        results_df
        .sort_values(
            by=[
                "recall",
                "predicted_objects",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )


best_threshold = float(
    best[
        "threshold"
    ]
)

best_min_area = float(
    best[
        "min_area_m2"
    ]
)


print(
    "\n"
    + "=" * 70
)

print(
    "SELECTED OPERATING POINT"
)

print(
    "=" * 70
)

print(
    f"Threshold       : "
    f"{best_threshold:.2f}"
)

print(
    f"Minimum area    : "
    f"{best_min_area:.2f} m²"
)

print(
    f"Candidate objects: "
    f"{int(best['predicted_objects']):,}"
)

print(
    f"Reference units : "
    f"{int(best['reference_units_detected'])}"
    f"/"
    f"{int(best['reference_units_total'])}"
)

print(
    f"Precision       : "
    f"{best['precision']:.6f}"
)

print(
    f"Recall          : "
    f"{best['recall']:.6f}"
)

print(
    f"F1              : "
    f"{best['f1']:.6f}"
)


# ============================================================
# 10. EXPORT CANDIDATES FOR SHINY
# ============================================================

candidate_gdf = (
    all_candidates[
        best_threshold
    ]
)

candidate_gdf = (
    candidate_gdf[
        candidate_gdf[
            "area_m2"
        ]
        >= best_min_area
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

candidate_gdf[
    "candidate_id"
] = np.arange(
    1,
    len(candidate_gdf) + 1
)

candidate_gdf[
    "review"
] = "unreviewed"


# Put useful fields first
candidate_gdf = (
    candidate_gdf[
        [
            "candidate_id",
            "threshold",
            "area_m2",
            "review",
            "geometry",
        ]
    ]
)


CANDIDATE_GPKG = (
    OUTPUT_DIR
    / "candidates_for_shiny.gpkg"
)

candidate_gdf.to_file(
    CANDIDATE_GPKG,
    layer="candidate_waste",
    driver="GPKG",
)


# ============================================================
# 11. SAVE OPERATING-POINT SUMMARY
# ============================================================

SUMMARY_TXT = (
    OUTPUT_DIR
    / "selected_operating_point.txt"
)

with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        f"Probability raster: "
        f"{PROB_RASTER}\n"
    )

    f.write(
        f"Threshold: "
        f"{best_threshold:.2f}\n"
    )

    f.write(
        f"Minimum area: "
        f"{best_min_area:.2f} m2\n"
    )

    f.write(
        f"Predicted objects: "
        f"{int(best['predicted_objects'])}\n"
    )

    f.write(
        f"Reference units detected: "
        f"{int(best['reference_units_detected'])}"
        f"/"
        f"{int(best['reference_units_total'])}\n"
    )

    f.write(
        f"Precision: "
        f"{best['precision']:.6f}\n"
    )

    f.write(
        f"Recall: "
        f"{best['recall']:.6f}\n"
    )

    f.write(
        f"F1: "
        f"{best['f1']:.6f}\n"
    )


print(
    "\nSaved:"
)

print(
    RESULTS_CSV
)

print(
    CANDIDATE_GPKG
)

print(
    SUMMARY_TXT
)

print(
    "\nDone."
)