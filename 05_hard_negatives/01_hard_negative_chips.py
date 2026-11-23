# =============================================================================
# CREATE COMPLEMENTARY HARD-NEGATIVE IMAGE / MASK PAIRS
# =============================================================================
#
# Reconstructs confirmed false-positive OBJECTS from the Shiny review database.
# Complementary mask meaning:
#   1 = confirmed NOT waste
#   0 = unknown / ignore
#
# IMPORTANT: TEST_AOI_PATH is an EXCLUSION area. No training chip whose focal
# negative object intersects the independent test AOI is created.
# =============================================================================

from pathlib import Path
import sqlite3
import random
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from shapely.geometry import box
from shapely.validation import make_valid

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 20260919
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# PATHS
# =============================================================================

BASE_PATH = Path.home() / "Documents" / "GitHub" / "mapping-waste-urban-africa"

ORTHO_PATH = BASE_PATH / "data" / "optical.tiff"
PREDICTION_PATH = BASE_PATH / "data" / "predictions" / "waste_piles_vector" / "waste_piles.shp"

APP_DIR = BASE_PATH / "data" / "annotations" / "waste_review_app"
DB_PATH = APP_DIR / "review_log.sqlite"
REVIEW_GRID_PATH = APP_DIR / "data" / "review_grid.gpkg"
OBJECT_CHIP_MAP_PATH = APP_DIR / "data" / "object_chip_map.csv"

# Independent test area: excluded from fine-tuning data.
TEST_AOI_PATH = Path(
    r"C:\Users\Peter\Documents\GitHub\waste_spatio_temporal\inputs\finetuning\AOI\AOI.shp"
)

# Reference waste in the test area. Read only for audit / path checking.
TEST_WASTE_PATH = Path(
    r"C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa\data\vectors\waste_probability_step0\waste.shp"
)

OUTPUT_DIR = BASE_PATH / "data" / "tiled_data" / "complementary_hard_negatives_v2"
IMAGE_DIR = OUTPUT_DIR / "images"
NEGATIVE_MASK_DIR = OUTPUT_DIR / "negative_masks"
QA_DIR = OUTPUT_DIR / "qa"
MANIFEST_PATH = OUTPUT_DIR / "finetuning_manifest.csv"
OBJECT_AUDIT_PATH = OUTPUT_DIR / "negative_object_audit.csv"
EXCLUDED_TEST_OBJECTS_PATH = OUTPUT_DIR / "excluded_test_area_negative_objects.gpkg"

# =============================================================================
# SETTINGS
# =============================================================================

TILE_SIZE = 256
MIN_OBJECT_AREA_M2 = 0.25
LARGE_OBJECT_STRIDE = 128
MIN_NEGATIVE_PIXELS = 5
N_QA = 30
COMPRESS = "lzw"

for d in [OUTPUT_DIR, IMAGE_DIR, NEGATIVE_MASK_DIR, QA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def clean_gdf(gdf):
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    return gdf[~gdf.geometry.is_empty].copy()


def classify_stage1_status(statuses):
    """Exact logic used by the review app at object level."""
    statuses = sorted(set(str(x) for x in statuses))
    if "unreviewed" in statuses:
        return "pending"
    if any(x in {"mixed", "not_sure"} for x in statuses):
        return "second_review"
    if statuses and all(x == "waste" for x in statuses):
        return "retained"
    if statuses and all(x == "not_waste" for x in statuses):
        return "rejected"
    return "second_review"


def centred_window(src, geom):
    p = geom.representative_point()
    row, col = src.index(p.x, p.y)
    row_off = max(0, min(int(row - TILE_SIZE // 2), src.height - TILE_SIZE))
    col_off = max(0, min(int(col - TILE_SIZE // 2), src.width - TILE_SIZE))
    return Window(col_off, row_off, TILE_SIZE, TILE_SIZE)


def windows_for_geometry(src, geom):
    """One centred chip for small objects; overlapping chips for large objects."""
    w0 = centred_window(src, geom)
    fp0 = box(*rasterio.windows.bounds(w0, src.transform))
    if fp0.contains(geom):
        return [w0]

    minx, miny, maxx, maxy = geom.bounds
    r1, c1 = src.index(minx, maxy)
    r2, c2 = src.index(maxx, miny)
    rmin, rmax = sorted((r1, r2))
    cmin, cmax = sorted((c1, c2))

    rmin = max(0, rmin - TILE_SIZE // 4)
    cmin = max(0, cmin - TILE_SIZE // 4)
    rmax = min(src.height - TILE_SIZE, rmax)
    cmax = min(src.width - TILE_SIZE, cmax)

    rows = list(range(rmin, max(rmin + 1, rmax + 1), LARGE_OBJECT_STRIDE)) or [int(w0.row_off)]
    cols = list(range(cmin, max(cmin + 1, cmax + 1), LARGE_OBJECT_STRIDE)) or [int(w0.col_off)]

    rows.append(max(0, min(src.height - TILE_SIZE, rmax)))
    cols.append(max(0, min(src.width - TILE_SIZE, cmax)))

    out, seen = [], set()
    for r in rows:
        for c in cols:
            r = max(0, min(int(r), src.height - TILE_SIZE))
            c = max(0, min(int(c), src.width - TILE_SIZE))
            key = (r, c)
            if key in seen:
                continue
            seen.add(key)
            w = Window(c, r, TILE_SIZE, TILE_SIZE)
            fp = box(*rasterio.windows.bounds(w, src.transform))
            if fp.intersects(geom):
                out.append(w)
    return out or [w0]


def write_rgb(path, data, profile, transform):
    p = profile.copy()
    p.update(height=TILE_SIZE, width=TILE_SIZE, count=3,
             dtype=data.dtype, transform=transform, compress=COMPRESS)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data)


def write_mask(path, mask, profile, transform):
    p = profile.copy()
    p.update(height=TILE_SIZE, width=TILE_SIZE, count=1,
             dtype="uint8", transform=transform, compress=COMPRESS, nodata=0)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(mask.astype(np.uint8), 1)

# =============================================================================
# INPUT CHECKS
# =============================================================================

required = [ORTHO_PATH, PREDICTION_PATH, DB_PATH, REVIEW_GRID_PATH,
            OBJECT_CHIP_MAP_PATH, TEST_AOI_PATH, TEST_WASTE_PATH]
missing = [p for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required input(s):\n" + "\n".join(map(str, missing)))

print("\n" + "=" * 80)
print("RECONSTRUCTING CONFIRMED FALSE-POSITIVE OBJECTS")
print("=" * 80)
print(f"\nReview database:\n{DB_PATH}")
print(f"\nIndependent test AOI (EXCLUDED):\n{TEST_AOI_PATH}")

# =============================================================================
# READ CURRENT REVIEW DECISIONS
# =============================================================================

con = sqlite3.connect(DB_PATH)
try:
    annotations = pd.read_sql_query(
        "SELECT chip_id, label, annotator, timestamp FROM annotations", con
    )
    second_review = pd.read_sql_query(
        "SELECT candidate_id, label, annotator, timestamp FROM second_review", con
    )
finally:
    con.close()

annotations["chip_id"] = annotations["chip_id"].astype(str)
second_review["candidate_id"] = second_review["candidate_id"].astype(str)

print("\nStage-1 decisions:")
print(annotations["label"].value_counts(dropna=False).to_string())
print("\nStage-2 decisions:")
print(second_review["label"].value_counts(dropna=False).to_string())

# =============================================================================
# READ GRID, OBJECT-CHIP MAP AND PREDICTIONS
# =============================================================================

review_grid = clean_gdf(gpd.read_file(REVIEW_GRID_PATH))
review_grid["chip_id"] = review_grid["chip_id"].astype(str)

objmap = pd.read_csv(
    OBJECT_CHIP_MAP_PATH,
    dtype={"candidate_id": str, "chip_id": str},
)[["candidate_id", "chip_id"]].drop_duplicates()

pred = clean_gdf(gpd.read_file(PREDICTION_PATH))
if "candidate_id" not in pred.columns:
    print("\nPrediction layer has no candidate_id; assigning row-order IDs exactly as the app does.")
    pred["candidate_id"] = np.arange(1, len(pred) + 1)
pred["candidate_id"] = pred["candidate_id"].astype(str)

# Current Stage-1 label for each review cell.
lookup = dict(zip(annotations["chip_id"], annotations["label"]))
review_grid["review_class"] = review_grid["chip_id"].map(lookup).fillna("unreviewed")

# Reconstruct the Stage-1 object class from ALL cells touched by each object.
obj_status = objmap.merge(
    review_grid[["chip_id", "review_class"]],
    on="chip_id", how="left"
)
obj_status["review_class"] = obj_status["review_class"].fillna("unreviewed")
obj_status = (
    obj_status.groupby("candidate_id", as_index=False)["review_class"]
    .agg(lambda x: classify_stage1_status(x.tolist()))
    .rename(columns={"review_class": "stage1_class"})
)

pred = pred.merge(obj_status, on="candidate_id", how="left")
pred["stage1_class"] = pred["stage1_class"].fillna("pending")

# Current Stage-2 object label.
s2 = second_review[["candidate_id", "label"]].rename(columns={"label": "stage2_label"})
pred = pred.merge(s2, on="candidate_id", how="left")

# =============================================================================
# CONFIRMED NEGATIVE OBJECTS
# =============================================================================

is_s1_negative = pred["stage1_class"].eq("rejected")
is_s2_negative = pred["stage2_label"].eq("not_waste")

pred["negative_source"] = np.where(
    is_s2_negative, "stage2_not_waste",
    np.where(is_s1_negative, "stage1_rejected", "")
)

neg = pred[is_s1_negative | is_s2_negative].copy()

with rasterio.open(ORTHO_PATH) as src:
    raster_crs = src.crs

if neg.crs != raster_crs:
    neg = neg.to_crs(raster_crs)

neg["area_m2"] = neg.geometry.area
n_before = len(neg)
neg = neg[neg["area_m2"] >= MIN_OBJECT_AREA_M2].copy()

print(f"\nConfirmed negative objects before area filter: {n_before:,}")
print(f"Confirmed negative objects >= {MIN_OBJECT_AREA_M2} m2: {len(neg):,}")
print("\nSource breakdown:")
print(neg["negative_source"].value_counts().to_string())

# =============================================================================
# EXCLUDE INDEPENDENT TEST AREA
# =============================================================================

test_aoi = clean_gdf(gpd.read_file(TEST_AOI_PATH))
if test_aoi.crs != raster_crs:
    test_aoi = test_aoi.to_crs(raster_crs)

test_union = test_aoi.geometry.union_all()
in_test = neg.geometry.intersects(test_union)
excluded_test = neg[in_test].copy()
neg = neg[~in_test].copy()

print(f"\nExcluded because they intersect independent test AOI: {len(excluded_test):,}")
print(f"Available for training outside test AOI: {len(neg):,}")

if len(excluded_test):
    excluded_test[["candidate_id", "stage1_class", "stage2_label",
                   "negative_source", "area_m2", "geometry"]].to_file(
        EXCLUDED_TEST_OBJECTS_PATH,
        layer="excluded_test_negatives",
        driver="GPKG"
    )

if len(neg) == 0:
    raise RuntimeError("No confirmed negative objects remain outside the independent test AOI.")

neg[["candidate_id", "stage1_class", "stage2_label",
     "negative_source", "area_m2"]].to_csv(OBJECT_AUDIT_PATH, index=False)

# =============================================================================
# CREATE IMAGE / BINARY NEGATIVE MASK PAIRS
# =============================================================================

neg = neg.reset_index(drop=True)
sindex = neg.sindex
manifest_rows = []
seen_windows = set()
tile_no = 0

print("\n" + "=" * 80)
print("CREATING 256 x 256 COMPLEMENTARY HARD-NEGATIVE PAIRS")
print("=" * 80)

with rasterio.open(ORTHO_PATH) as src:
    profile = src.profile.copy()

    for i, row in neg.iterrows():
        focal = row.geometry
        focal_id = str(row["candidate_id"])

        for w in windows_for_geometry(src, focal):
            key = (int(w.row_off), int(w.col_off))
            if key in seen_windows:
                continue

            footprint = box(*rasterio.windows.bounds(w, src.transform))

            # Strong leakage guard: the entire image chip must stay outside test AOI.
            if footprint.intersects(test_union):
                continue

            possible = list(sindex.intersection(footprint.bounds))
            if not possible:
                continue
            local = neg.iloc[possible]
            local = local[local.geometry.intersects(footprint)]
            if len(local) == 0:
                continue

            transform = rasterio.windows.transform(w, src.transform)
            shapes = [(g, 1) for g in local.geometry if g is not None and not g.is_empty]
            mask = rasterize(
                shapes=shapes,
                out_shape=(TILE_SIZE, TILE_SIZE),
                transform=transform,
                fill=0,
                default_value=1,
                dtype="uint8",
                all_touched=False,
            )

            n_neg_px = int(mask.sum())
            if n_neg_px < MIN_NEGATIVE_PIXELS:
                continue

            image = src.read([1, 2, 3], window=w)
            if image.shape[1:] != (TILE_SIZE, TILE_SIZE):
                continue

            tile_no += 1
            tile_id = f"hn_{tile_no:06d}"
            image_file = f"{tile_id}.tif"
            mask_file = f"{tile_id}.tif"

            write_rgb(IMAGE_DIR / image_file, image, profile, transform)
            write_mask(NEGATIVE_MASK_DIR / mask_file, mask, profile, transform)

            manifest_rows.append({
                "tile_id": tile_id,
                "sample_id": focal_id,
                "focal_candidate_id": focal_id,
                "focal_source": row["negative_source"],
                "image_file": image_file,
                "negative_mask_file": mask_file,
                "negative_pixels": n_neg_px,
                "n_negative_objects_in_chip": int(len(local)),
                "row_off": int(w.row_off),
                "col_off": int(w.col_off),
                "tile_size": TILE_SIZE,
            })

            seen_windows.add(key)

        if (i + 1) % 250 == 0 or (i + 1) == len(neg):
            print(
                f"Processed negative objects: {i+1:,} / {len(neg):,} | "
                f"saved chips: {tile_no:,}"
            )

manifest = pd.DataFrame(manifest_rows)
if len(manifest) == 0:
    raise RuntimeError(
        "No complementary hard-negative pairs were created. "
        "Check CRS, test-area exclusion and review IDs."
    )
manifest.to_csv(MANIFEST_PATH, index=False)

# =============================================================================
# OPTIONAL QA PREVIEWS
# =============================================================================

try:
    import matplotlib.pyplot as plt

    q = manifest.sample(n=min(N_QA, len(manifest)), random_state=SEED)
    for _, r in q.iterrows():
        with rasterio.open(IMAGE_DIR / r["image_file"]) as src:
            rgb = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
        with rasterio.open(NEGATIVE_MASK_DIR / r["negative_mask_file"]) as src:
            m = src.read(1)

        lo, hi = np.percentile(rgb, [2, 98])
        if hi > lo:
            show = np.clip((rgb - lo) / (hi - lo), 0, 1)
        else:
            show = np.clip(rgb / 255.0, 0, 1)

        plt.figure(figsize=(5, 5))
        plt.imshow(show)
        plt.contour(m, levels=[0.5], linewidths=1.5)
        plt.title(f"{r['tile_id']} | {r['focal_source']} | neg px={r['negative_pixels']}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(QA_DIR / f"{r['tile_id']}_qa.png", dpi=180, bbox_inches="tight")
        plt.close()
except Exception as e:
    print("\nWARNING: QA PNGs were not created, but TIFF pairs are valid.")
    print(str(e))

# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n" + "=" * 80)
print("COMPLEMENTARY HARD-NEGATIVE DATASET COMPLETE")
print("=" * 80)
print("\nObjects used by source:")
print(neg["negative_source"].value_counts().to_string())
print(f"\nIndependent-test negative objects excluded: {len(excluded_test):,}")
print(f"Image/mask pairs created: {len(manifest):,}")
print(f"Total confirmed NOT-waste pixels: {manifest['negative_pixels'].sum():,}")
print(f"\nImages:\n{IMAGE_DIR}")
print(f"\nBinary negative masks:\n{NEGATIVE_MASK_DIR}")
print(f"\nManifest:\n{MANIFEST_PATH}")
print(f"\nObject audit:\n{OBJECT_AUDIT_PATH}")
print(f"\nQA previews:\n{QA_DIR}")
print("\nMask meaning: 1 = confirmed NOT waste; 0 = unknown / ignore")
print("No generated training chip intersects the independent test AOI.")
print("\nDone.")
