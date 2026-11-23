"""
March 2024 training-sample generator -- STREAMING VERSION
==========================================================

This version is designed for a very large prediction shapefile (~466,000
polygons). It DOES NOT load the whole prediction file into GeoPandas.
Instead, it streams one feature at a time with Fiona and keeps only small
summary/reservoir objects in memory.

Outputs
-------
1. hard_negative_iter_XX.gpkg / .shp
2. random_background_iter_XX.gpkg / .shp
3. annotation_boxes_iter_XX.gpkg
4. march_sampling_registry.csv
5. sampling_summary_iter_XX.csv
6. heldout_test_grid_37.gpkg

Design
------
- Fishnet layer: grid_cells
- Fishnet ID field: id
- Grid 37: fixed independent test area
- Chip size: 256 x 256 raster pixels
- Hard negatives: model predictions that do NOT intersect verified waste
- Random background: random chip boxes from non-test AOI
- Sampling without replacement across iterations

IMPORTANT
---------
Random-background chips are not automatically zero-waste masks. Inspect and
annotate the COMPLETE chip. If genuine waste occurs in a random chip, digitise
it as waste.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import math
import warnings

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import Point, box, shape
from shapely.ops import transform as shapely_transform
from shapely.prepared import prep
from shapely.strtree import STRtree

try:
    from shapely import make_valid
except ImportError:
    make_valid = None


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

PROJECT_DIR = Path(
    r"C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa"
)

REFERENCE_RASTER = PROJECT_DIR / "data" / "optical.tiff"

FISHNET_FILE = (
    PROJECT_DIR
    / "data"
    / "tiled_data"
    / "spatial_metadata"
    / "spatial_partitions.gpkg"
)
FISHNET_LAYER = "grid_cells"
GRID_ID_FIELD = "id"

VERIFIED_WASTE_FILE = (
    PROJECT_DIR
    / "results"
    / "final_verified_waste_piles_shapefile_updated"
    / "verified_waste_piles.shp"
)

PREDICTIONS_FILE = PROJECT_DIR / "outputs" / "waste_piles.shp"

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "tiled_data"
    / "training_sample_selection"
)

# Fixed independent test region
TEST_GRID_ID = 37

# Exact training chip size
CHIP_SIZE = 256

# Increment after each annotation/retraining round
ITERATION = 1

# Initial incremental batch
N_RANDOM_BACKGROUND = 100
N_HARD_NEGATIVE = 50

# Remove tiny polygon speckles before hard-negative sampling.
# Set 0.0 if you intentionally want every object considered.
MIN_PREDICTION_AREA_M2 = 0.25

# Optional buffer around verified waste during prediction matching.
# 0.0 means exact intersection only.
VERIFIED_MATCH_BUFFER_M = 0.0

# If a shapefile has lost CRS metadata but its coordinates are already in the
# March raster CRS, assign that CRS without changing coordinates.
ASSUME_MISSING_VECTOR_CRS_IS_RASTER_CRS = True

# For random AOI chips, avoid placing the chip centre directly inside known
# verified waste. This does NOT imply the complete chip contains no waste.
AVOID_VERIFIED_WASTE_CENTRES_FOR_RANDOM = True

# Keep the whole 256x256 training chip inside the non-test fishnet AOI.
REQUIRE_FULL_CHIP_INSIDE_TRAINING_AOI = True

BASE_RANDOM_SEED = 20260925
MAX_ATTEMPTS_PER_CHIP = 5000

EXPORT_SHAPEFILES = True

# Print streaming progress after this many prediction features.
PROGRESS_EVERY = 25000


# =============================================================================
# 2. HELPERS
# =============================================================================

REGISTRY_COLUMNS = [
    "chip_id",
    "chip_key",
    "source",
    "iteration",
    "row0",
    "col0",
    "grid_id",
    "source_ids",
    "selected_at",
]


def read_small_vector(path, layer=None, fallback_crs=None):
    """Read a small vector layer with optional missing-CRS recovery."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")

    if layer is None:
        gdf = gpd.read_file(path)
    else:
        gdf = gpd.read_file(path, layer=layer)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError(f"No valid geometries found in:\n{path}")

    if gdf.crs is None:
        if (
            ASSUME_MISSING_VECTOR_CRS_IS_RASTER_CRS
            and fallback_crs is not None
        ):
            print("\nWARNING: CRS metadata missing:")
            print(f"  {path}")
            print(f"Assigning raster CRS: {fallback_crs}")
            print("Coordinates are NOT transformed.")
            gdf = gdf.set_crs(fallback_crs, allow_override=True)
        else:
            raise ValueError(f"Layer has no CRS:\n{path}")

    return gdf


def union_geometry(gdf):
    try:
        return gdf.geometry.union_all()
    except AttributeError:
        return gdf.geometry.unary_union


def safe_geometry(geom):
    """Attempt a lightweight validity repair only when needed."""
    if geom is None or geom.is_empty:
        return None

    if geom.is_valid:
        return geom

    try:
        if make_valid is not None:
            repaired = make_valid(geom)
        else:
            repaired = geom.buffer(0)
    except Exception:
        try:
            repaired = geom.buffer(0)
        except Exception:
            return None

    if repaired is None or repaired.is_empty:
        return None

    return repaired


def chip_key(row0, col0):
    return f"R{int(row0):07d}_C{int(col0):07d}"


def point_to_window(src, point, chip_size):
    row, col = src.index(point.x, point.y)

    half = chip_size // 2
    row0 = int(row - half)
    col0 = int(col - half)

    row0 = max(0, min(row0, src.height - chip_size))
    col0 = max(0, min(col0, src.width - chip_size))

    if row0 < 0 or col0 < 0:
        return None
    if row0 + chip_size > src.height:
        return None
    if col0 + chip_size > src.width:
        return None

    return row0, col0


def make_chip_polygon(src, row0, col0, chip_size):
    window = Window(
        col_off=int(col0),
        row_off=int(row0),
        width=chip_size,
        height=chip_size,
    )

    left, bottom, right, top = window_bounds(window, src.transform)
    return box(left, bottom, right, top)


def load_registry(path):
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)

    df = pd.read_csv(
        path,
        dtype={"chip_id": str, "chip_key": str},
    )

    for col in REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[REGISTRY_COLUMNS]


def rows_to_gdf(rows, crs):
    if rows:
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    return gpd.GeoDataFrame(
        pd.DataFrame(columns=REGISTRY_COLUMNS + ["geometry"]),
        geometry="geometry",
        crs=crs,
    )


def proportional_quotas(weights, n_total):
    weights = np.asarray(weights, dtype=float)

    if n_total <= 0:
        return np.zeros(len(weights), dtype=int)

    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

    if weights.sum() <= 0:
        return np.zeros(len(weights), dtype=int)

    raw = n_total * weights / weights.sum()
    quotas = np.floor(raw).astype(int)
    remainder = int(n_total - quotas.sum())

    if remainder > 0:
        order = np.argsort(-(raw - quotas))
        quotas[order[:remainder]] += 1

    return quotas


def shapefile_safe(gdf):
    return gdf.rename(
        columns={
            "iteration": "iter",
            "source_ids": "src_ids",
            "selected_at": "sel_time",
        }
    )


def source_crs_and_transformer(fiona_collection, target_crs):
    """
    Determine whether streamed prediction geometries need transformation.
    Returns a callable transformer or None.
    """
    src_crs = None

    try:
        if fiona_collection.crs_wkt:
            src_crs = CRS.from_wkt(fiona_collection.crs_wkt)
        elif fiona_collection.crs:
            src_crs = CRS.from_user_input(fiona_collection.crs)
    except Exception:
        src_crs = None

    target = CRS.from_user_input(target_crs)

    if src_crs is None:
        if ASSUME_MISSING_VECTOR_CRS_IS_RASTER_CRS:
            print("\nWARNING: prediction shapefile has no usable CRS metadata.")
            print(f"Assuming prediction coordinates are already {target}.")
            return None

        raise ValueError(
            "Prediction shapefile has no CRS metadata and CRS fallback is disabled."
        )

    if src_crs == target:
        return None

    print(f"Prediction CRS: {src_crs}")
    print(f"Transforming streamed predictions to: {target}")

    transformer = Transformer.from_crs(
        src_crs,
        target,
        always_xy=True,
    )

    return transformer.transform


def grid_id_for_point(point, grid_tree, grid_geoms, grid_ids):
    """Return containing non-test grid ID or None."""
    try:
        possible = grid_tree.query(point)
    except Exception:
        return None

    for idx in np.asarray(possible, dtype=int):
        geom = grid_geoms[idx]
        try:
            if geom.covers(point):
                return grid_ids[idx]
        except Exception:
            continue

    return None


def intersects_verified(geom, verified_tree):
    """Fast boolean query; no intersection geometry is constructed."""
    try:
        hits = verified_tree.query(geom, predicate="intersects")
        return len(hits) > 0
    except Exception:
        # If predicate query fails for a rare bad geometry, reject the object
        # conservatively rather than risk training on true waste.
        return True


def reservoir_add(reservoir, count_seen, item, capacity, rng):
    """Standard reservoir sampling for one stratum."""
    if capacity <= 0:
        return

    if len(reservoir) < capacity:
        reservoir.append(item)
        return

    j = int(rng.integers(0, count_seen))

    if j < capacity:
        reservoir[j] = item


# =============================================================================
# 3. INITIALISE
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_CSV = OUTPUT_DIR / "march_sampling_registry.csv"

registry = load_registry(REGISTRY_CSV)

if not registry.empty:
    old_iterations = pd.to_numeric(registry["iteration"], errors="coerce")
    if (old_iterations == ITERATION).any():
        raise RuntimeError(
            f"ITERATION={ITERATION} already exists in the registry.\n"
            "Increase ITERATION before drawing another batch."
        )

used_keys = set(registry["chip_key"].dropna().astype(str))

rng = np.random.default_rng(BASE_RANDOM_SEED + ITERATION)


# =============================================================================
# 4. OPEN RASTER AND SMALL VECTOR DATA
# =============================================================================

with rasterio.open(REFERENCE_RASTER) as raster:

    raster_crs = raster.crs

    if raster_crs is None:
        raise ValueError("Reference raster has no CRS.")

    print("\n" + "=" * 78)
    print("MARCH TRAINING-SAMPLE GENERATOR -- STREAMING VERSION")
    print("=" * 78)
    print(f"Iteration       : {ITERATION}")
    print(f"Test grid       : {TEST_GRID_ID}")
    print(f"Raster CRS      : {raster_crs}")
    print(f"Raster size     : {raster.width:,} x {raster.height:,}")
    print(f"Previously used : {len(used_keys):,} chip windows")

    fishnet = read_small_vector(
        FISHNET_FILE,
        layer=FISHNET_LAYER,
        fallback_crs=raster_crs,
    ).to_crs(raster_crs)

    if GRID_ID_FIELD not in fishnet.columns:
        raise ValueError(
            f"Fishnet field '{GRID_ID_FIELD}' was not found.\n"
            f"Fields: {list(fishnet.columns)}"
        )

    test_mask = (
        fishnet[GRID_ID_FIELD].astype(str).str.strip()
        == str(TEST_GRID_ID)
    )

    test_grid = fishnet.loc[test_mask].copy()

    if test_grid.empty:
        raise ValueError(f"Fishnet grid {TEST_GRID_ID} was not found.")

    train_fishnet = fishnet.loc[~test_mask].copy().reset_index(drop=True)

    test_union = union_geometry(test_grid)
    training_aoi = union_geometry(train_fishnet)

    prep_test = prep(test_union)
    prep_training = prep(training_aoi)

    test_grid.to_file(
        OUTPUT_DIR / f"heldout_test_grid_{TEST_GRID_ID}.gpkg",
        layer="test_grid",
        driver="GPKG",
        index=False,
    )

    verified = read_small_vector(
        VERIFIED_WASTE_FILE,
        fallback_crs=raster_crs,
    ).to_crs(raster_crs)

    print(f"Verified waste  : {len(verified):,} polygons")

    raster_extent = box(*raster.bounds)

    if not verified.geometry.intersects(raster_extent).any():
        raise ValueError(
            "Verified waste does not overlap the raster. Check its CRS."
        )

    verified_match_geoms = []

    for geom in verified.geometry:
        geom = safe_geometry(geom)
        if geom is None:
            continue
        if VERIFIED_MATCH_BUFFER_M > 0:
            geom = geom.buffer(VERIFIED_MATCH_BUFFER_M)
        verified_match_geoms.append(geom)

    if not verified_match_geoms:
        raise ValueError("No usable verified-waste geometry remains.")

    verified_tree = STRtree(verified_match_geoms)

    # Tree for assigning candidate centres to non-test fishnet cells.
    grid_geoms = list(train_fishnet.geometry)
    grid_ids = train_fishnet[GRID_ID_FIELD].astype(str).tolist()
    grid_tree = STRtree(grid_geoms)


    # =========================================================================
    # 5. STREAM PREDICTIONS AND BUILD A UNIFORM RESERVOIR OF UNIQUE CHIP WINDOWS
    # =========================================================================

    print("\nStreaming model predictions...")
    print("The full prediction shapefile is NOT loaded into memory.")

    # One reservoir per fishnet grid. Capacity N_HARD_NEGATIVE is enough because
    # no final stratum quota can exceed the total requested number.
    reservoirs = defaultdict(list)
    eligible_counts_by_grid = defaultdict(int)

    # Needed only for deduplicating candidate CHIP WINDOWS, not polygon geometry.
    seen_candidate_chip_keys = set()

    stats = defaultdict(int)

    with fiona.open(PREDICTIONS_FILE) as pred_src:

        transform_func = source_crs_and_transformer(pred_src, raster_crs)

        try:
            total_features = len(pred_src)
        except Exception:
            total_features = None

        if total_features is not None:
            print(f"Prediction features reported by file: {total_features:,}")

        for feature_no, feature in enumerate(pred_src, start=1):

            stats["raw_predictions"] += 1

            geometry_mapping = feature.get("geometry")
            if geometry_mapping is None:
                stats["missing_geometry"] += 1
                continue

            try:
                geom = shape(geometry_mapping)
            except Exception:
                stats["unreadable_geometry"] += 1
                continue

            if transform_func is not None:
                try:
                    geom = shapely_transform(transform_func, geom)
                except Exception:
                    stats["transform_failed"] += 1
                    continue

            geom = safe_geometry(geom)

            if geom is None:
                stats["invalid_geometry"] += 1
                continue

            # Fast bounding / area filters before spatial-index queries.
            try:
                area = float(geom.area)
            except Exception:
                stats["area_failed"] += 1
                continue

            if area < MIN_PREDICTION_AREA_M2:
                stats["below_min_area"] += 1
                continue

            stats["after_area_filter"] += 1

            # Predictions touching verified waste are not hard negatives.
            if intersects_verified(geom, verified_tree):
                stats["matched_verified"] += 1
                continue

            stats["candidate_fp_objects"] += 1

            # We only need a location to construct the full training chip.
            try:
                point = geom.representative_point()
            except Exception:
                stats["representative_point_failed"] += 1
                continue

            # Candidate must lie in a non-test fishnet cell.
            grid_id = grid_id_for_point(
                point,
                grid_tree,
                grid_geoms,
                grid_ids,
            )

            if grid_id is None:
                stats["outside_training_fishnet"] += 1
                continue

            rc = point_to_window(raster, point, CHIP_SIZE)

            if rc is None:
                stats["outside_raster"] += 1
                continue

            row0, col0 = rc
            key = chip_key(row0, col0)

            # Never return a previously annotated raster window.
            if key in used_keys:
                stats["already_used"] += 1
                continue

            # Deduplicate many tiny false-positive objects that fall within the
            # same 256x256 raster chip.
            if key in seen_candidate_chip_keys:
                stats["duplicate_candidate_chip"] += 1
                continue

            chip_geom = make_chip_polygon(
                raster,
                row0,
                col0,
                CHIP_SIZE,
            )

            if prep_test.intersects(chip_geom):
                stats["chip_touches_test_grid"] += 1
                continue

            if REQUIRE_FULL_CHIP_INSIDE_TRAINING_AOI:
                if not training_aoi.covers(chip_geom):
                    stats["chip_not_fully_inside_aoi"] += 1
                    continue
            else:
                if not prep_training.intersects(chip_geom):
                    stats["chip_outside_aoi"] += 1
                    continue

            seen_candidate_chip_keys.add(key)
            stats["eligible_unique_hn_chips"] += 1

            grid_id = str(grid_id)
            eligible_counts_by_grid[grid_id] += 1

            item = {
                "chip_key": key,
                "row0": int(row0),
                "col0": int(col0),
                "grid_id": grid_id,
                "source_id": str(feature.get("id", feature_no)),
                "geometry": chip_geom,
            }

            reservoir_add(
                reservoirs[grid_id],
                eligible_counts_by_grid[grid_id],
                item,
                N_HARD_NEGATIVE,
                rng,
            )

            if PROGRESS_EVERY > 0 and feature_no % PROGRESS_EVERY == 0:
                if total_features:
                    pct = 100.0 * feature_no / total_features
                    print(
                        f"  {feature_no:,}/{total_features:,} "
                        f"({pct:5.1f}%) | "
                        f"eligible unique HN chips: "
                        f"{stats['eligible_unique_hn_chips']:,}"
                    )
                else:
                    print(
                        f"  {feature_no:,} predictions processed | "
                        f"eligible unique HN chips: "
                        f"{stats['eligible_unique_hn_chips']:,}"
                    )

    print("\nPrediction streaming complete.")
    print(f"Raw predictions                  : {stats['raw_predictions']:,}")
    print(f"Below {MIN_PREDICTION_AREA_M2:g} m²               : {stats['below_min_area']:,}")
    print(f"After area filter                : {stats['after_area_filter']:,}")
    print(f"Intersect verified waste         : {stats['matched_verified']:,}")
    print(f"Candidate false-positive objects : {stats['candidate_fp_objects']:,}")
    print(f"Duplicate candidate chip windows : {stats['duplicate_candidate_chip']:,}")
    print(f"Touch test grid {TEST_GRID_ID}               : {stats['chip_touches_test_grid']:,}")
    print(f"Eligible unique HN chip windows  : {stats['eligible_unique_hn_chips']:,}")


    # =========================================================================
    # 6. ALLOCATE AND DRAW THE HARD-NEGATIVE BATCH
    # =========================================================================

    hard_rows = []

    if N_HARD_NEGATIVE > 0 and eligible_counts_by_grid:

        grid_names = list(eligible_counts_by_grid.keys())
        grid_counts = np.asarray(
            [eligible_counts_by_grid[g] for g in grid_names],
            dtype=float,
        )

        target_hn = min(
            N_HARD_NEGATIVE,
            int(grid_counts.sum()),
        )

        quotas = proportional_quotas(grid_counts, target_hn)

        selected_items = []

        for grid_name, quota in zip(grid_names, quotas):
            quota = int(quota)
            if quota <= 0:
                continue

            pool = reservoirs[grid_name]
            if not pool:
                continue

            take = min(quota, len(pool))
            idx = rng.choice(len(pool), size=take, replace=False)
            selected_items.extend([pool[int(i)] for i in np.atleast_1d(idx)])

        # Top-up from all reservoir items if integer allocation or edge cases
        # leave a small shortfall.
        if len(selected_items) < target_hn:
            chosen_keys = {x["chip_key"] for x in selected_items}
            remaining_pool = []

            for pool in reservoirs.values():
                for item in pool:
                    if item["chip_key"] not in chosen_keys:
                        remaining_pool.append(item)

            if remaining_pool:
                rng.shuffle(remaining_pool)
                need = target_hn - len(selected_items)
                selected_items.extend(remaining_pool[:need])

        for i, item in enumerate(selected_items, start=1):
            hard_rows.append(
                {
                    "chip_id": f"HN_I{ITERATION:02d}_{i:04d}",
                    "chip_key": item["chip_key"],
                    "source": "hard_negative",
                    "iteration": ITERATION,
                    "row0": item["row0"],
                    "col0": item["col0"],
                    "grid_id": item["grid_id"],
                    "source_ids": item["source_id"],
                    "selected_at": datetime.now().isoformat(timespec="seconds"),
                    "geometry": item["geometry"],
                }
            )
            used_keys.add(item["chip_key"])

    hard_gdf = rows_to_gdf(hard_rows, raster_crs)

    print(f"Hard-negative chips selected      : {len(hard_gdf):,}")


    # =========================================================================
    # 7. RANDOM AOI BACKGROUND CHIPS
    # =========================================================================

    random_rows = []

    verified_union = union_geometry(verified)
    prep_verified = prep(verified_union)

    strata = train_fishnet[[GRID_ID_FIELD, "geometry"]].copy()
    strata["_area"] = strata.geometry.area

    quotas = proportional_quotas(
        strata["_area"].to_numpy(),
        N_RANDOM_BACKGROUND,
    )

    for (_, stratum), quota in zip(strata.iterrows(), quotas):

        quota = int(quota)
        if quota <= 0:
            continue

        stratum_geom = stratum.geometry
        prepared_stratum = prep(stratum_geom)
        minx, miny, maxx, maxy = stratum_geom.bounds

        added = 0
        attempts = 0
        max_attempts = max(1, quota * MAX_ATTEMPTS_PER_CHIP)

        while added < quota and attempts < max_attempts:
            attempts += 1

            p = Point(
                rng.uniform(minx, maxx),
                rng.uniform(miny, maxy),
            )

            if not prepared_stratum.covers(p):
                continue

            if (
                AVOID_VERIFIED_WASTE_CENTRES_FOR_RANDOM
                and prep_verified.covers(p)
            ):
                continue

            rc = point_to_window(raster, p, CHIP_SIZE)
            if rc is None:
                continue

            row0, col0 = rc
            key = chip_key(row0, col0)

            if key in used_keys:
                continue

            chip_geom = make_chip_polygon(
                raster,
                row0,
                col0,
                CHIP_SIZE,
            )

            if prep_test.intersects(chip_geom):
                continue

            if REQUIRE_FULL_CHIP_INSIDE_TRAINING_AOI:
                if not training_aoi.covers(chip_geom):
                    continue

            random_rows.append(
                {
                    "chip_id": f"BG_I{ITERATION:02d}_{len(random_rows)+1:04d}",
                    "chip_key": key,
                    "source": "random_background",
                    "iteration": ITERATION,
                    "row0": int(row0),
                    "col0": int(col0),
                    "grid_id": str(stratum[GRID_ID_FIELD]),
                    "source_ids": "",
                    "selected_at": datetime.now().isoformat(timespec="seconds"),
                    "geometry": chip_geom,
                }
            )

            used_keys.add(key)
            added += 1

        if added < quota:
            warnings.warn(
                f"Grid {stratum[GRID_ID_FIELD]} requested {quota} "
                f"background chips, but only {added} could be selected."
            )

    # Global top-up if edge/boundary restrictions caused a shortfall.
    shortfall = N_RANDOM_BACKGROUND - len(random_rows)

    if shortfall > 0:
        minx, miny, maxx, maxy = training_aoi.bounds
        attempts = 0
        max_attempts = shortfall * MAX_ATTEMPTS_PER_CHIP

        while shortfall > 0 and attempts < max_attempts:
            attempts += 1

            p = Point(
                rng.uniform(minx, maxx),
                rng.uniform(miny, maxy),
            )

            if not prep_training.covers(p):
                continue

            if (
                AVOID_VERIFIED_WASTE_CENTRES_FOR_RANDOM
                and prep_verified.covers(p)
            ):
                continue

            rc = point_to_window(raster, p, CHIP_SIZE)
            if rc is None:
                continue

            row0, col0 = rc
            key = chip_key(row0, col0)

            if key in used_keys:
                continue

            chip_geom = make_chip_polygon(raster, row0, col0, CHIP_SIZE)

            if prep_test.intersects(chip_geom):
                continue

            if (
                REQUIRE_FULL_CHIP_INSIDE_TRAINING_AOI
                and not training_aoi.covers(chip_geom)
            ):
                continue

            grid_id = grid_id_for_point(
                chip_geom.centroid,
                grid_tree,
                grid_geoms,
                grid_ids,
            )

            if grid_id is None:
                continue

            random_rows.append(
                {
                    "chip_id": f"BG_I{ITERATION:02d}_{len(random_rows)+1:04d}",
                    "chip_key": key,
                    "source": "random_background",
                    "iteration": ITERATION,
                    "row0": int(row0),
                    "col0": int(col0),
                    "grid_id": str(grid_id),
                    "source_ids": "",
                    "selected_at": datetime.now().isoformat(timespec="seconds"),
                    "geometry": chip_geom,
                }
            )

            used_keys.add(key)
            shortfall -= 1

    random_gdf = rows_to_gdf(random_rows, raster_crs)

    print(f"Random-background chips selected  : {len(random_gdf):,}")


    # =========================================================================
    # 8. SAVE OUTPUTS
    # =========================================================================

    parts = []
    if not random_gdf.empty:
        parts.append(random_gdf)
    if not hard_gdf.empty:
        parts.append(hard_gdf)

    if not parts:
        raise RuntimeError("No new annotation chips were selected.")

    current = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        geometry="geometry",
        crs=raster_crs,
    )

    bg_path = OUTPUT_DIR / f"random_background_iter_{ITERATION:02d}.gpkg"
    hn_path = OUTPUT_DIR / f"hard_negative_iter_{ITERATION:02d}.gpkg"
    combined_path = OUTPUT_DIR / f"annotation_boxes_iter_{ITERATION:02d}.gpkg"

    if not random_gdf.empty:
        random_gdf.to_file(
            bg_path,
            layer="random_background",
            driver="GPKG",
            index=False,
        )

    if not hard_gdf.empty:
        hard_gdf.to_file(
            hn_path,
            layer="hard_negative",
            driver="GPKG",
            index=False,
        )

    current.to_file(
        combined_path,
        layer="annotation_boxes",
        driver="GPKG",
        index=False,
    )

    if EXPORT_SHAPEFILES:
        if not random_gdf.empty:
            shapefile_safe(random_gdf).to_file(
                OUTPUT_DIR / f"random_background_iter_{ITERATION:02d}.shp",
                driver="ESRI Shapefile",
                index=False,
            )

        if not hard_gdf.empty:
            shapefile_safe(hard_gdf).to_file(
                OUTPUT_DIR / f"hard_negative_iter_{ITERATION:02d}.shp",
                driver="ESRI Shapefile",
                index=False,
            )


# =============================================================================
# 9. UPDATE REGISTRY
# =============================================================================

current_registry = pd.DataFrame(
    current.drop(columns="geometry")[REGISTRY_COLUMNS]
)

updated_registry = pd.concat(
    [registry, current_registry],
    ignore_index=True,
)

if updated_registry["chip_key"].duplicated().any():
    dup = updated_registry.loc[
        updated_registry["chip_key"].duplicated(keep=False),
        "chip_key",
    ].drop_duplicates().head(20).tolist()

    raise RuntimeError(
        "Duplicate chip windows detected before registry save:\n"
        + "\n".join(dup)
    )

updated_registry.to_csv(REGISTRY_CSV, index=False)


# =============================================================================
# 10. SUMMARY
# =============================================================================

summary_items = {
    "iteration": ITERATION,
    "test_grid_id": TEST_GRID_ID,
    "min_prediction_area_m2": MIN_PREDICTION_AREA_M2,
    "raw_predictions": stats["raw_predictions"],
    "below_min_area": stats["below_min_area"],
    "after_area_filter": stats["after_area_filter"],
    "matched_verified": stats["matched_verified"],
    "candidate_false_positive_objects": stats["candidate_fp_objects"],
    "duplicate_candidate_chip_windows": stats["duplicate_candidate_chip"],
    "eligible_unique_hn_chips": stats["eligible_unique_hn_chips"],
    "hard_negative_chips_selected": len(hard_gdf),
    "random_background_chips_selected": len(random_gdf),
    "total_unique_chips_in_registry": len(updated_registry),
}

summary_df = pd.DataFrame(
    list(summary_items.items()),
    columns=["item", "value"],
)

summary_path = OUTPUT_DIR / f"sampling_summary_iter_{ITERATION:02d}.csv"
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 78)
print("DONE")
print("=" * 78)

if not hard_gdf.empty:
    print(f"Hard-negative boxes : {hn_path}")
if not random_gdf.empty:
    print(f"Random background   : {bg_path}")

print(f"Combined boxes      : {combined_path}")
print(f"Registry            : {REGISTRY_CSV}")
print(f"Summary             : {summary_path}")

print(
    "\nNext: open the two chip-box layers in QGIS and annotate the complete "
    "256x256 footprints. Grid 37 remains untouched for evaluation."
)
