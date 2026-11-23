from pathlib import Path
import math

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.features import shapes
from rasterio.windows import Window
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

# Fine-tuned waste-probability raster.
# Change only this path if you want to evaluate another model/raster.
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

TEMP_DIR = OUTPUT_DIR / "temporary_threshold_rasters"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

# Probability thresholds to evaluate.
THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

# Minimum predicted-object areas to evaluate.
# Table 4 corresponds to min_area_m2 == 0.25.
MIN_AREAS_M2 = [
    0.25,
    0.50,
    1.00,
    2.00,
]

# A reference waste unit is counted as detected when at least 1%
# of its area overlaps a predicted object.
MIN_REFERENCE_COVERAGE = 0.01

# Optional criterion used only when selecting an operating point.
TARGET_RECALL = 0.90

# Read the very large probability raster in manageable windows.
BLOCK_SIZE = 2048

# Delete the temporary thresholded uint8 raster after polygonisation.
DELETE_TEMP_RASTERS = True


# ============================================================
# 3. HELPERS
# ============================================================

def safe_divide(a, b):
    return a / b if b > 0 else np.nan


def get_polygon_parts(geom):
    """Return individual Polygon geometries from polygonal geometry."""

    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)

    if isinstance(geom, GeometryCollection):
        parts = []
        for g in geom.geoms:
            parts.extend(get_polygon_parts(g))
        return parts

    return []


def build_reference_units(reference_gdf):
    """
    Dissolve overlapping/touching reference polygons and split the result
    into contiguous reference waste units.
    """

    union_geom = unary_union(reference_gdf.geometry)
    parts = get_polygon_parts(union_geom)

    units = gpd.GeoDataFrame(
        {
            "reference_id": np.arange(1, len(parts) + 1)
        },
        geometry=parts,
        crs=reference_gdf.crs,
    )

    units["reference_area_m2"] = units.geometry.area

    return units


def iter_windows(src, block_size=2048):
    """Yield raster windows without reading the whole raster into memory."""

    for row_off in range(0, src.height, block_size):
        block_h = min(block_size, src.height - row_off)

        for col_off in range(0, src.width, block_size):
            block_w = min(block_size, src.width - col_off)

            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=block_w,
                height=block_h,
            )


def probability_statistics_blockwise(probability_raster, block_size=2048):
    """
    Calculate minimum, maximum and mean probability without loading the
    complete raster into RAM.
    """

    global_min = np.inf
    global_max = -np.inf
    total_sum = 0.0
    total_count = 0

    with rasterio.open(probability_raster) as src:
        total_windows = (
            math.ceil(src.height / block_size)
            * math.ceil(src.width / block_size)
        )

        for i, window in enumerate(
            iter_windows(src, block_size),
            start=1,
        ):
            arr = src.read(
                1,
                window=window,
                masked=True,
            )

            data = arr.data
            invalid = np.ma.getmaskarray(arr) | ~np.isfinite(data)
            valid_values = data[~invalid]

            if valid_values.size > 0:
                block_min = float(valid_values.min())
                block_max = float(valid_values.max())

                global_min = min(global_min, block_min)
                global_max = max(global_max, block_max)
                total_sum += float(valid_values.sum(dtype=np.float64))
                total_count += int(valid_values.size)

            if i % 250 == 0 or i == total_windows:
                print(
                    f"  Statistics: processed {i:,}/{total_windows:,} blocks"
                )

    if total_count == 0:
        raise ValueError("Probability raster contains no valid pixels.")

    return {
        "minimum": global_min,
        "maximum": global_max,
        "mean": total_sum / total_count,
        "valid_pixels": total_count,
    }


def create_binary_threshold_raster(
    probability_raster,
    output_raster,
    threshold,
    block_size=2048,
):
    """
    Threshold the probability raster block-by-block.

    Output coding:
        1   = valid pixel with probability >= threshold
        0   = valid pixel with probability < threshold
        255 = NoData / invalid probability

    The output is uint8, tiled, compressed and BigTIFF-enabled.
    """

    print(
        f"\nCreating temporary binary raster for p >= {threshold:.2f}"
    )

    with rasterio.open(probability_raster) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=255,
            compress="DEFLATE",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="YES",
        )

        total_windows = (
            math.ceil(src.height / block_size)
            * math.ceil(src.width / block_size)
        )

        with rasterio.open(output_raster, "w", **profile) as dst:
            for i, window in enumerate(
                iter_windows(src, block_size),
                start=1,
            ):
                arr = src.read(
                    1,
                    window=window,
                    masked=True,
                )

                data = arr.data
                invalid = np.ma.getmaskarray(arr) | ~np.isfinite(data)

                binary = np.zeros(data.shape, dtype=np.uint8)
                binary[invalid] = 255

                is_waste = (
                    (~invalid)
                    & (data >= threshold)
                )

                binary[is_waste] = 1

                dst.write(
                    binary,
                    1,
                    window=window,
                )

                if i % 250 == 0 or i == total_windows:
                    print(
                        f"  Thresholding: processed {i:,}/{total_windows:,} blocks"
                    )


def polygonise_binary_raster(
    binary_raster,
    threshold,
    minimum_area=0.25,
):
    """
    Polygonise contiguous pixels equal to 1 directly from the on-disk
    threshold raster. Uses 8-neighbour connectivity, matching the previous
    analysis.
    """

    print(
        f"Polygonising threshold {threshold:.2f} "
        f"with 8-neighbour connectivity..."
    )

    records = []
    object_id = 0

    with rasterio.open(binary_raster) as src:
        raster_crs = src.crs

        # Passing rasterio.band(src, 1) lets GDAL/rasterio process the
        # raster from disk rather than first creating a giant NumPy array.
        source_band = rasterio.band(src, 1)

        for geom, value in shapes(
            source_band,
            transform=src.transform,
            connectivity=8,
        ):
            # Ignore background (0) and NoData (255).
            if int(value) != 1:
                continue

            polygon = shape(geom)

            if polygon is None or polygon.is_empty:
                continue

            area_m2 = float(polygon.area)

            if area_m2 < minimum_area:
                continue

            object_id += 1

            records.append(
                {
                    "object_id": object_id,
                    "threshold": float(threshold),
                    "area_m2": area_m2,
                    "geometry": polygon,
                }
            )

    if len(records) == 0:
        return gpd.GeoDataFrame(
            {
                "object_id": pd.Series(dtype="int64"),
                "threshold": pd.Series(dtype="float64"),
                "area_m2": pd.Series(dtype="float64"),
                "geometry": gpd.GeoSeries([], crs=raster_crs),
            },
            geometry="geometry",
            crs=raster_crs,
        )

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=raster_crs,
    )


def evaluate_objects(predictions, references):
    """
    One-to-one object matching used for the manuscript evaluation.

    A prediction/reference pair is eligible when the prediction overlaps at
    least MIN_REFERENCE_COVERAGE of the reference area. Eligible pairs are
    sorted by IoU, then reference coverage, then intersection area, and
    greedily matched one-to-one.
    """

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
                predicate="intersects",
            )
        )

        for ref_idx in candidates:
            ref_geom = references.iloc[ref_idx].geometry
            intersection = pred_geom.intersection(ref_geom)
            intersection_area = intersection.area

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

            if reference_coverage >= MIN_REFERENCE_COVERAGE:
                matches.append(
                    {
                        "pred_idx": pred_idx,
                        "ref_idx": ref_idx,
                        "iou": iou,
                        "reference_coverage": reference_coverage,
                        "intersection_area": intersection_area,
                    }
                )

    # Highest-quality candidate matches first.
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
    fp = len(predictions) - tp
    fn = len(references) - tp

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    f1 = (
        safe_divide(
            2 * precision * recall,
            precision + recall,
        )
        if precision + recall > 0
        else 0.0
    )

    return {
        "predicted_objects": len(predictions),
        "matched_predictions": tp,
        "reference_units_detected": tp,
        "reference_units_total": len(references),
        "unmatched_predictions": fp,
        "missed_reference_units": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# 4. CHECK INPUTS AND RASTER METADATA
# ============================================================

print("\n" + "=" * 70)
print("MEMORY-SAFE WHOLE-AREA PROBABILITY SWEEP")
print("=" * 70)

print(f"\nProbability raster:\n{PROB_RASTER}")
print(f"\nReference polygons:\n{REFERENCE_PATH}")
print(f"\nOutput directory:\n{OUTPUT_DIR}")

if not PROB_RASTER.exists():
    raise FileNotFoundError(PROB_RASTER)

if not REFERENCE_PATH.exists():
    raise FileNotFoundError(REFERENCE_PATH)

with rasterio.open(PROB_RASTER) as src:
    if src.count != 1:
        raise ValueError(
            "Expected a single-band waste probability raster."
        )

    if src.crs is None:
        raise ValueError(
            "Probability raster has no CRS."
        )

    raster_crs = src.crs

    print("\nRaster metadata")
    print("---------------")
    print(f"Width       : {src.width:,}")
    print(f"Height      : {src.height:,}")
    print(f"Data type   : {src.dtypes[0]}")
    print(f"NoData      : {src.nodata}")
    print(f"CRS         : {src.crs}")
    print(f"Resolution  : {src.res}")


# ============================================================
# 5. BLOCK-WISE PROBABILITY STATISTICS
# ============================================================

print("\nCalculating probability statistics block-by-block...")

prob_stats = probability_statistics_blockwise(
    PROB_RASTER,
    block_size=BLOCK_SIZE,
)

print("\nProbability statistics")
print("----------------------")
print(f"Minimum      : {prob_stats['minimum']:.6f}")
print(f"Maximum      : {prob_stats['maximum']:.6f}")
print(f"Mean         : {prob_stats['mean']:.6f}")
print(f"Valid pixels : {prob_stats['valid_pixels']:,}")


# ============================================================
# 6. READ AND PREPARE REFERENCE WASTE
# ============================================================

print("\nReading reference waste polygons...")

reference = gpd.read_file(REFERENCE_PATH)

if reference.empty:
    raise ValueError(
        "Reference waste file contains no features."
    )

reference = reference.to_crs(raster_crs)

reference_units = build_reference_units(reference)

print(f"Reference source features : {len(reference):,}")
print(f"Contiguous reference units: {len(reference_units):,}")


# ============================================================
# 7. THRESHOLD AND AREA SWEEP
# ============================================================

results = []
all_candidates = {}

smallest_area = min(MIN_AREAS_M2)

for threshold in THRESHOLDS:
    print("\n" + "=" * 70)
    print(f"THRESHOLD = {threshold:.2f}")
    print("=" * 70)

    threshold_tag = f"{threshold:.2f}".replace(".", "p")

    binary_raster = (
        TEMP_DIR
        / f"binary_threshold_{threshold_tag}.tif"
    )

    # --------------------------------------------------------
    # 7a. Create threshold raster block-by-block
    # --------------------------------------------------------

    create_binary_threshold_raster(
        probability_raster=PROB_RASTER,
        output_raster=binary_raster,
        threshold=threshold,
        block_size=BLOCK_SIZE,
    )

    # --------------------------------------------------------
    # 7b. Convert contiguous thresholded pixels to objects
    # --------------------------------------------------------

    candidates = polygonise_binary_raster(
        binary_raster=binary_raster,
        threshold=threshold,
        minimum_area=smallest_area,
    )

    print(
        f"Objects >= {smallest_area:.2f} m²: "
        f"{len(candidates):,}"
    )

    # Candidate object collections are small compared with the raster,
    # so keeping them in memory is safe and lets us export the selected
    # operating point later.
    all_candidates[threshold] = candidates

    # --------------------------------------------------------
    # 7c. Evaluate each minimum-area criterion
    # --------------------------------------------------------

    for min_area in MIN_AREAS_M2:
        selected = (
            candidates[
                candidates["area_m2"] >= min_area
            ]
            .copy()
            .reset_index(drop=True)
        )

        metrics = evaluate_objects(
            selected,
            reference_units,
        )

        row = {
            "threshold": float(threshold),
            "min_area_m2": float(min_area),
            **metrics,
        }

        results.append(row)

        print(f"\nMinimum area = {min_area:.2f} m²")
        print(
            f"  Predicted objects   : "
            f"{metrics['predicted_objects']:,}"
        )
        print(
            f"  References detected : "
            f"{metrics['reference_units_detected']}"
            f"/{metrics['reference_units_total']}"
        )
        print(
            f"  Precision           : "
            f"{metrics['precision']:.6f}"
        )
        print(
            f"  Recall              : "
            f"{metrics['recall']:.6f}"
        )
        print(
            f"  F1                  : "
            f"{metrics['f1']:.6f}"
        )

    # --------------------------------------------------------
    # 7d. Delete the large temporary binary raster
    # --------------------------------------------------------

    if DELETE_TEMP_RASTERS:
        try:
            binary_raster.unlink()
            print(
                f"\nDeleted temporary raster: "
                f"{binary_raster.name}"
            )
        except Exception as exc:
            print(
                f"\nWarning: could not delete "
                f"{binary_raster}: {exc}"
            )


# ============================================================
# 8. SAVE FULL SWEEP RESULTS
# ============================================================

results_df = pd.DataFrame(results)

RESULTS_CSV = (
    OUTPUT_DIR
    / "threshold_area_sweep.csv"
)

results_df.to_csv(
    RESULTS_CSV,
    index=False,
)

# Columns shown in the console and useful for manuscript reporting.
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

print("\n\n" + "=" * 70)
print("FULL RESULTS")
print("=" * 70)
print(
    results_df[display_cols].to_string(index=False)
)


# ============================================================
# 9. SAVE TABLE-4-STYLE RESULTS
# ============================================================

# The manuscript table uses the 0.25 m² minimum object size and
# thresholds 0.50 to 0.90 for the fine-tuned model.
table4_df = (
    results_df[
        (np.isclose(results_df["min_area_m2"], 0.25))
        & (results_df["threshold"] >= 0.50)
    ]
    .copy()
    .sort_values("threshold")
)

TABLE4_CSV = (
    OUTPUT_DIR
    / "table4_finetuned_thresholds.csv"
)

table4_df.to_csv(
    TABLE4_CSV,
    index=False,
)

print("\n" + "=" * 70)
print("TABLE 4 STYLE RESULTS: MINIMUM AREA = 0.25 m²")
print("=" * 70)
print(
    table4_df[display_cols].to_string(index=False)
)


# ============================================================
# 10. SELECT OPERATING POINT
# ============================================================

eligible = (
    results_df[
        results_df["recall"] >= TARGET_RECALL
    ]
    .copy()
)

if len(eligible) > 0:
    # Primary objective: fewest objects requiring manual review.
    # Tie-break: higher precision.
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
    # If target recall is unattainable, retain highest recall,
    # then the configuration with the fewest candidate objects.
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

best_threshold = float(best["threshold"])
best_min_area = float(best["min_area_m2"])

print("\n" + "=" * 70)
print("SELECTED OPERATING POINT")
print("=" * 70)
print(f"Threshold         : {best_threshold:.2f}")
print(f"Minimum area      : {best_min_area:.2f} m²")
print(f"Candidate objects : {int(best['predicted_objects']):,}")
print(
    f"Reference units   : "
    f"{int(best['reference_units_detected'])}"
    f"/{int(best['reference_units_total'])}"
)
print(f"Precision         : {best['precision']:.6f}")
print(f"Recall            : {best['recall']:.6f}")
print(f"F1                : {best['f1']:.6f}")


# ============================================================
# 11. EXPORT CANDIDATES FOR SHINY / MANUAL REVIEW
# ============================================================

candidate_gdf = (
    all_candidates[best_threshold]
    [
        all_candidates[best_threshold]["area_m2"]
        >= best_min_area
    ]
    .copy()
    .reset_index(drop=True)
)

candidate_gdf["candidate_id"] = np.arange(
    1,
    len(candidate_gdf) + 1,
)

candidate_gdf["review"] = "unreviewed"

candidate_gdf = candidate_gdf[
    [
        "candidate_id",
        "threshold",
        "area_m2",
        "review",
        "geometry",
    ]
]

CANDIDATE_GPKG = (
    OUTPUT_DIR
    / "candidates_for_shiny.gpkg"
)

# Remove old file so reruns do not fail or leave stale layers.
if CANDIDATE_GPKG.exists():
    CANDIDATE_GPKG.unlink()

candidate_gdf.to_file(
    CANDIDATE_GPKG,
    layer="candidate_waste",
    driver="GPKG",
)


# ============================================================
# 12. SAVE OPERATING-POINT SUMMARY
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
    f.write(f"Probability raster: {PROB_RASTER}\n")
    f.write(f"Reference file: {REFERENCE_PATH}\n")
    f.write(f"Threshold: {best_threshold:.2f}\n")
    f.write(f"Minimum area: {best_min_area:.2f} m2\n")
    f.write(
        f"Predicted objects: "
        f"{int(best['predicted_objects'])}\n"
    )
    f.write(
        f"Reference units detected: "
        f"{int(best['reference_units_detected'])}"
        f"/{int(best['reference_units_total'])}\n"
    )
    f.write(f"Precision: {best['precision']:.6f}\n")
    f.write(f"Recall: {best['recall']:.6f}\n")
    f.write(f"F1: {best['f1']:.6f}\n")


# ============================================================
# 13. FINISH
# ============================================================

print("\n" + "=" * 70)
print("SAVED OUTPUTS")
print("=" * 70)
print(RESULTS_CSV)
print(TABLE4_CSV)
print(CANDIDATE_GPKG)
print(SUMMARY_TXT)
print("\nDone.")
