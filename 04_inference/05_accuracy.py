from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio

from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
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

PRED_RASTER = (
    BASE_PATH
    / "outputs"
    / "landcover_predicted_class_full.tif"
)

PRED_OBJECTS = (
    BASE_PATH
    / "outputs"
    / "waste_piles.shp"
)

REFERENCE = (
    BASE_PATH
    / "data"
    / "waste"
    / "waste_outlines.geojson"
)


# ============================================================
# 2. SETTINGS
# ============================================================

# Waste code in categorical prediction raster
WASTE_CODE = 1

# Evaluate in manageable raster blocks
BLOCK_SIZE = 2048

# Same minimum object size used in earlier spatial evaluation
MIN_PRED_AREA_M2 = 0.25

# A reference pile is considered detected when at least 1%
# of its area overlaps a prediction.
MIN_REFERENCE_COVERAGE = 0.01


# ============================================================
# 3. HELPER
# ============================================================

def safe_divide(a, b):
    return a / b if b > 0 else np.nan


def extract_polygon_parts(geom):
    """Return individual Polygon geometries."""
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)

    if isinstance(geom, GeometryCollection):
        polygons = []
        for g in geom.geoms:
            polygons.extend(extract_polygon_parts(g))
        return polygons

    return []


# ============================================================
# 4. READ REFERENCE
# ============================================================

print("\nReading reference waste polygons...")

reference = gpd.read_file(REFERENCE)

if reference.empty:
    raise ValueError("Reference waste file contains no features.")

print(f"Reference features read: {len(reference):,}")


# ============================================================
# 5. PIXEL-LEVEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("PIXEL-LEVEL EVALUATION")
print("=" * 70)

TP = 0
FP = 0
FN = 0
TN = 0


with rasterio.open(PRED_RASTER) as src:

    if src.crs is None:
        raise ValueError("Prediction raster has no CRS.")

    reference_raster_crs = reference.to_crs(src.crs)

    # Spatial index makes block-wise rasterisation much faster
    ref_sindex = reference_raster_crs.sindex

    height = src.height
    width = src.width

    for row in range(0, height, BLOCK_SIZE):

        block_h = min(BLOCK_SIZE, height - row)

        for col in range(0, width, BLOCK_SIZE):

            block_w = min(BLOCK_SIZE, width - col)

            win = Window(
                col_off=col,
                row_off=row,
                width=block_w,
                height=block_h,
            )

            pred = src.read(1, window=win)

            valid = pred != src.nodata if src.nodata is not None else np.ones(
                pred.shape,
                dtype=bool
            )

            pred_waste = pred == WASTE_CODE

            # Geographic extent of this raster block
            left, bottom, right, top = window_bounds(
                win,
                src.transform
            )

            block_box = box(
                left,
                bottom,
                right,
                top
            )

            candidate_idx = list(
                ref_sindex.query(
                    block_box,
                    predicate="intersects"
                )
            )

            if candidate_idx:

                candidate_geometries = (
                    reference_raster_crs
                    .iloc[candidate_idx]
                    .geometry
                )

                ref_mask = rasterize(
                    (
                        (geom, 1)
                        for geom in candidate_geometries
                        if geom is not None and not geom.is_empty
                    ),
                    out_shape=(block_h, block_w),
                    transform=src.window_transform(win),
                    fill=0,
                    dtype=np.uint8,
                    all_touched=False,
                ).astype(bool)

            else:

                ref_mask = np.zeros(
                    (block_h, block_w),
                    dtype=bool
                )

            TP += int(
                np.count_nonzero(
                    pred_waste & ref_mask & valid
                )
            )

            FP += int(
                np.count_nonzero(
                    pred_waste & ~ref_mask & valid
                )
            )

            FN += int(
                np.count_nonzero(
                    ~pred_waste & ref_mask & valid
                )
            )

            TN += int(
                np.count_nonzero(
                    ~pred_waste & ~ref_mask & valid
                )
            )

        print(
            f"Processed raster row "
            f"{row:,} / {height:,}"
        )


pixel_precision = safe_divide(TP, TP + FP)
pixel_recall = safe_divide(TP, TP + FN)
pixel_f1 = safe_divide(
    2 * pixel_precision * pixel_recall,
    pixel_precision + pixel_recall
)
pixel_iou = safe_divide(
    TP,
    TP + FP + FN
)


print("\nPixel confusion counts")
print("----------------------")
print(f"TP: {TP:,}")
print(f"FP: {FP:,}")
print(f"FN: {FN:,}")
print(f"TN: {TN:,}")

print("\nPixel metrics")
print("-------------")
print(f"Precision : {pixel_precision:.6f}")
print(f"Recall    : {pixel_recall:.6f}")
print(f"F1        : {pixel_f1:.6f}")
print(f"IoU       : {pixel_iou:.6f}")


# ============================================================
# 6. OBJECT-LEVEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("OBJECT-LEVEL EVALUATION")
print("=" * 70)

pred_objects = gpd.read_file(PRED_OBJECTS)

if pred_objects.empty:
    raise ValueError("Prediction shapefile contains no objects.")


# Use raster CRS as common projected CRS
with rasterio.open(PRED_RASTER) as src:
    common_crs = src.crs


pred_objects = pred_objects.to_crs(common_crs)
reference_objects = reference.to_crs(common_crs)


# ------------------------------------------------------------
# Filter extremely small predictions
# ------------------------------------------------------------

pred_objects["calc_area_m2"] = (
    pred_objects.geometry.area
)

pred_objects = pred_objects[
    pred_objects["calc_area_m2"]
    >= MIN_PRED_AREA_M2
].copy()

pred_objects = pred_objects.reset_index(drop=True)

print(
    f"Predicted objects >= "
    f"{MIN_PRED_AREA_M2} m²: "
    f"{len(pred_objects):,}"
)


# ------------------------------------------------------------
# Convert reference polygons into contiguous waste units
# ------------------------------------------------------------

reference_union = unary_union(
    reference_objects.geometry
)

reference_parts = extract_polygon_parts(
    reference_union
)

reference_units = gpd.GeoDataFrame(
    {
        "reference_id": np.arange(
            1,
            len(reference_parts) + 1
        )
    },
    geometry=reference_parts,
    crs=common_crs,
)

reference_units["reference_area"] = (
    reference_units.geometry.area
)

print(
    f"Contiguous reference waste units: "
    f"{len(reference_units):,}"
)


# ------------------------------------------------------------
# Find possible prediction-reference matches
# ------------------------------------------------------------

ref_index = reference_units.sindex

possible_matches = []


for pred_id, pred_row in pred_objects.iterrows():

    pred_geom = pred_row.geometry

    if pred_geom is None or pred_geom.is_empty:
        continue

    candidate_refs = list(
        ref_index.query(
            pred_geom,
            predicate="intersects"
        )
    )

    for ref_idx in candidate_refs:

        ref_geom = reference_units.iloc[
            ref_idx
        ].geometry

        intersection = pred_geom.intersection(
            ref_geom
        )

        inter_area = intersection.area

        if inter_area <= 0:
            continue

        pred_area = pred_geom.area
        ref_area = ref_geom.area

        ref_coverage = (
            inter_area / ref_area
            if ref_area > 0
            else 0.0
        )

        union_area = (
            pred_area
            + ref_area
            - inter_area
        )

        iou = (
            inter_area / union_area
            if union_area > 0
            else 0.0
        )

        if ref_coverage >= MIN_REFERENCE_COVERAGE:

            possible_matches.append(
                {
                    "pred_id": pred_id,
                    "ref_idx": ref_idx,
                    "iou": iou,
                    "ref_coverage": ref_coverage,
                    "intersection_area": inter_area,
                }
            )


# ------------------------------------------------------------
# Greedy one-to-one matching
# ------------------------------------------------------------

possible_matches.sort(
    key=lambda x: (
        x["iou"],
        x["ref_coverage"],
        x["intersection_area"],
    ),
    reverse=True,
)

matched_predictions = set()
matched_references = set()
accepted_matches = []


for match in possible_matches:

    pred_id = match["pred_id"]
    ref_idx = match["ref_idx"]

    if pred_id in matched_predictions:
        continue

    if ref_idx in matched_references:
        continue

    matched_predictions.add(pred_id)
    matched_references.add(ref_idx)

    accepted_matches.append(match)


object_tp = len(accepted_matches)
object_fp = len(pred_objects) - object_tp
object_fn = len(reference_units) - object_tp

object_precision = safe_divide(
    object_tp,
    object_tp + object_fp
)

object_recall = safe_divide(
    object_tp,
    object_tp + object_fn
)

object_f1 = safe_divide(
    2 * object_precision * object_recall,
    object_precision + object_recall
)


print("\nObject confusion counts")
print("-----------------------")
print(f"Matched predicted objects : {object_tp:,}")
print(f"Unmatched predictions     : {object_fp:,}")
print(f"Missed reference units    : {object_fn:,}")

print("\nObject metrics")
print("--------------")
print(f"Precision : {object_precision:.6f}")
print(f"Recall    : {object_recall:.6f}")
print(f"F1        : {object_f1:.6f}")

print(
    f"\nReference units detected: "
    f"{object_tp}/{len(reference_units)}"
)

print("\nFinished.")