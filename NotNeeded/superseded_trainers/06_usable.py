#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio

from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box


# ============================================================
# 1. PATHS
# ============================================================

ANNOTATIONS = (
    r"C:\Users\Peter\Documents\GitHub\waste_spatio_temporal"
    r"\annotations\march.shp"
)

IMAGE_RASTER = (
    r"C:\Users\Peter\Documents\GitHub\waste_spatio_temporal"
    r"\intermediaries\subset_raw\subset_March_2024.tif"
)

REJECTED_GRIDS = (
    r"C:\Users\Peter\Documents\GitHub\waste_spatio_temporal"
    r"\intermediaries\march_training_chips"
    r"\annotation_diagnostics"
    r"\rejected_256_tiles_incomplete_annotation.shp"
)

OUTPUT_DIR = (
    r"C:\Users\Peter\Documents\GitHub\waste_spatio_temporal"
    r"\intermediaries\march_training_chips"
)


# ============================================================
# 2. SETTINGS
# ============================================================

PATCH_SIZE = 256
CLASS_FIELD = "class"

# A tile is accepted only if every pixel is annotated.
REQUIRED_COVERAGE = 1.0

# If True, replace an existing chip with the same tile_id.
# Normally rejected tiles did not previously produce chips,
# so False is safest.
OVERWRITE_EXISTING = False


# ============================================================
# 3. OUTPUT FOLDERS
# ============================================================

out_dir = Path(OUTPUT_DIR)

image_dir = out_dir / "images"
mask_dir = out_dir / "masks"
diagnostic_dir = out_dir / "annotation_diagnostics"

image_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)
diagnostic_dir.mkdir(parents=True, exist_ok=True)

newly_accepted_shp = (
    diagnostic_dir /
    "rejected_tiles_now_complete.shp"
)

still_incomplete_shp = (
    diagnostic_dir /
    "rejected_tiles_still_incomplete.shp"
)

summary_csv = (
    diagnostic_dir /
    "rejected_tiles_recheck_summary.csv"
)


# ============================================================
# 4. READ UPDATED ANNOTATIONS
# ============================================================

print("=" * 70)
print("READING UPDATED MARCH ANNOTATIONS")
print("=" * 70)

gdf = gpd.read_file(ANNOTATIONS)

if CLASS_FIELD not in gdf.columns:
    raise ValueError(
        f"Field '{CLASS_FIELD}' not found in march.shp.\n"
        f"Available fields: {gdf.columns.tolist()}"
    )

# Remove null/empty geometries
gdf = gdf[
    gdf.geometry.notnull() &
    (~gdf.geometry.is_empty)
].copy()

# Repair invalid geometries
invalid = ~gdf.geometry.is_valid

if invalid.any():
    print(f"Repairing {invalid.sum()} invalid annotation geometries...")
    gdf.loc[invalid, "geometry"] = (
        gdf.loc[invalid, "geometry"].buffer(0)
    )

gdf = gdf[
    gdf.geometry.notnull() &
    (~gdf.geometry.is_empty)
].copy()

# Check class values
gdf[CLASS_FIELD] = gdf[CLASS_FIELD].astype(int)

classes_found = sorted(gdf[CLASS_FIELD].unique())

print("Classes found:", classes_found)

bad_classes = [
    x for x in classes_found
    if x < 1 or x > 10
]

if bad_classes:
    raise ValueError(
        f"Unexpected class values: {bad_classes}. "
        "Expected classes 1-10."
    )


# ============================================================
# 5. READ PREVIOUSLY REJECTED GRID CELLS
# ============================================================

print("\n")
print("=" * 70)
print("READING PREVIOUSLY REJECTED 256 × 256 GRID CELLS")
print("=" * 70)

rejected_gdf = gpd.read_file(REJECTED_GRIDS)

print("Rejected records found:", len(rejected_gdf))
print("Fields:", rejected_gdf.columns.tolist())

required_fields = {"tile_id", "row_id", "col_id"}

missing_fields = (
    required_fields -
    set(rejected_gdf.columns)
)

if missing_fields:
    raise ValueError(
        "Rejected-grid layer is missing required fields: "
        f"{sorted(missing_fields)}"
    )

# In case the layer contains more than incomplete annotation
# records, retain only the ones rejected for annotation gaps.
if "reason" in rejected_gdf.columns:

    before = len(rejected_gdf)

    rejected_gdf = rejected_gdf[
        rejected_gdf["reason"] ==
        "incomplete_annotation"
    ].copy()

    print(
        f"Incomplete-annotation grids retained: "
        f"{len(rejected_gdf)} of {before}"
    )

if len(rejected_gdf) == 0:
    raise ValueError(
        "No incomplete-annotation grids were found."
    )


# ============================================================
# 6. OPEN SOURCE RASTER
# ============================================================

with rasterio.open(IMAGE_RASTER) as src:

    print("\n")
    print("=" * 70)
    print("SOURCE MARCH IMAGE")
    print("=" * 70)

    print("CRS       :", src.crs)
    print("Resolution:", src.res)
    print("Size      :", src.width, "x", src.height)
    print("Bands     :", src.count)

    if src.crs is None:
        raise ValueError(
            "Source raster has no defined CRS."
        )

    if gdf.crs is None:
        raise ValueError(
            "march.shp has no defined CRS."
        )

    if rejected_gdf.crs is None:
        raise ValueError(
            "Rejected-grid shapefile has no defined CRS."
        )

    # Match both vector datasets to raster CRS
    if gdf.crs != src.crs:
        print("Reprojecting annotations to raster CRS...")
        gdf = gdf.to_crs(src.crs)

    if rejected_gdf.crs != src.crs:
        print("Reprojecting rejected grids to raster CRS...")
        rejected_gdf = rejected_gdf.to_crs(src.crs)


    # ========================================================
    # 7. PREPARE ANNOTATION SHAPES
    # ========================================================

    class_shapes = [
        (
            geom,
            int(class_value)
        )
        for geom, class_value in zip(
            gdf.geometry,
            gdf[CLASS_FIELD]
        )
    ]

    coverage_shapes = [
        (geom, 1)
        for geom in gdf.geometry
    ]


    # ========================================================
    # 8. PROCESS ONLY PREVIOUSLY REJECTED CELLS
    # ========================================================

    accepted_records = []
    incomplete_records = []
    summary_records = []

    saved = 0
    still_incomplete = 0
    skipped_existing = 0
    skipped_nodata = 0
    invalid_window = 0

    print("\n")
    print("=" * 70)
    print("RECHECKING PREVIOUSLY REJECTED GRID CELLS")
    print("=" * 70)

    for i, row in rejected_gdf.iterrows():

        tile_id = str(row["tile_id"])

        row_id = int(row["row_id"])
        col_id = int(row["col_id"])

        # ----------------------------------------------------
        # Reconstruct EXACT original raster window
        # ----------------------------------------------------

        row_off = row_id * PATCH_SIZE
        col_off = col_id * PATCH_SIZE

        window = Window(
            col_off=col_off,
            row_off=row_off,
            width=PATCH_SIZE,
            height=PATCH_SIZE
        )

        # Ensure window lies within raster
        if (
            col_off < 0 or
            row_off < 0 or
            col_off + PATCH_SIZE > src.width or
            row_off + PATCH_SIZE > src.height
        ):

            invalid_window += 1

            print(
                f"Skipping {tile_id}: "
                "window falls outside raster."
            )

            continue

        window_transform = (
            src.window_transform(window)
        )

        left, bottom, right, top = (
            rasterio.windows.bounds(
                window,
                src.transform
            )
        )

        tile_geom = box(
            left,
            bottom,
            right,
            top
        )


        # ----------------------------------------------------
        # Check updated annotation coverage
        # ----------------------------------------------------

        coverage = rasterize(
            coverage_shapes,
            out_shape=(
                PATCH_SIZE,
                PATCH_SIZE
            ),
            transform=window_transform,
            fill=0,
            default_value=1,
            dtype="uint8",
            all_touched=False
        )

        total_pixels = (
            PATCH_SIZE * PATCH_SIZE
        )

        annotated_pixels = (
            np.count_nonzero(
                coverage == 1
            )
        )

        coverage_fraction = (
            annotated_pixels /
            total_pixels
        )

        coverage_percent = (
            coverage_fraction * 100
        )


        # ----------------------------------------------------
        # Still incomplete? Do NOT make image-mask pair
        # ----------------------------------------------------

        if coverage_fraction < REQUIRED_COVERAGE:

            still_incomplete += 1

            incomplete_records.append(
                {
                    "tile_id": tile_id,
                    "row_id": row_id,
                    "col_id": col_id,
                    "cov_pct": coverage_percent,
                    "miss_pct":
                        100 - coverage_percent,
                    "geometry": tile_geom
                }
            )

            summary_records.append(
                {
                    "tile_id": tile_id,
                    "row_id": row_id,
                    "col_id": col_id,
                    "coverage_pct":
                        coverage_percent,
                    "status":
                        "still_incomplete"
                }
            )

            continue


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = src.read(
            window=window
        )

        if (
            image.shape[1] != PATCH_SIZE or
            image.shape[2] != PATCH_SIZE
        ):
            invalid_window += 1
            continue


        # ----------------------------------------------------
        # Check source-image nodata
        # ----------------------------------------------------

        if src.nodata is not None:

            nodata_mask = np.all(
                image == src.nodata,
                axis=0
            )

            if np.any(nodata_mask):

                skipped_nodata += 1

                summary_records.append(
                    {
                        "tile_id": tile_id,
                        "row_id": row_id,
                        "col_id": col_id,
                        "coverage_pct":
                            coverage_percent,
                        "status":
                            "image_nodata"
                    }
                )

                continue


        # ----------------------------------------------------
        # Rasterize updated land-cover classes
        # ----------------------------------------------------

        mask = rasterize(
            class_shapes,
            out_shape=(
                PATCH_SIZE,
                PATCH_SIZE
            ),
            transform=window_transform,
            fill=0,
            dtype="uint8",
            all_touched=False
        )


        # ----------------------------------------------------
        # Final safety check:
        # NO unannotated pixels permitted
        # ----------------------------------------------------

        zero_pixels = np.count_nonzero(
            mask == 0
        )

        if zero_pixels > 0:

            zero_fraction = (
                zero_pixels /
                total_pixels
            )

            still_incomplete += 1

            incomplete_records.append(
                {
                    "tile_id": tile_id,
                    "row_id": row_id,
                    "col_id": col_id,
                    "cov_pct":
                        (1 - zero_fraction) * 100,
                    "miss_pct":
                        zero_fraction * 100,
                    "geometry": tile_geom
                }
            )

            summary_records.append(
                {
                    "tile_id": tile_id,
                    "row_id": row_id,
                    "col_id": col_id,
                    "coverage_pct":
                        (1 - zero_fraction) * 100,
                    "status":
                        "class_gap"
                }
            )

            continue


        # ----------------------------------------------------
        # Ensure only Classes 1-10 are present
        # ----------------------------------------------------

        unique_values = np.unique(mask)

        invalid_values = [
            int(value)
            for value in unique_values
            if value < 1 or value > 10
        ]

        if invalid_values:

            print(
                f"WARNING: {tile_id} contains "
                f"invalid mask values: {invalid_values}"
            )

            continue


        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        image_path = (
            image_dir /
            f"{tile_id}.tif"
        )

        mask_path = (
            mask_dir /
            f"{tile_id}.tif"
        )


        # ----------------------------------------------------
        # Avoid accidental replacement
        # ----------------------------------------------------

        if (
            not OVERWRITE_EXISTING and
            (
                image_path.exists() or
                mask_path.exists()
            )
        ):

            skipped_existing += 1

            print(
                f"Skipping existing pair: {tile_id}"
            )

            continue


        # ----------------------------------------------------
        # Save image chip
        # ----------------------------------------------------

        image_profile = (
            src.profile.copy()
        )

        image_profile.update(
            {
                "driver": "GTiff",
                "height": PATCH_SIZE,
                "width": PATCH_SIZE,
                "transform":
                    window_transform,
                "compress": "lzw"
            }
        )

        with rasterio.open(
            image_path,
            "w",
            **image_profile
        ) as dst:

            dst.write(image)


        # ----------------------------------------------------
        # Save mask chip
        # ----------------------------------------------------

        mask_profile = {
            "driver": "GTiff",
            "height": PATCH_SIZE,
            "width": PATCH_SIZE,
            "count": 1,
            "dtype": "uint8",
            "crs": src.crs,
            "transform":
                window_transform,
            "nodata": 0,
            "compress": "lzw"
        }

        with rasterio.open(
            mask_path,
            "w",
            **mask_profile
        ) as dst:

            dst.write(
                mask,
                1
            )


        # ----------------------------------------------------
        # Record newly accepted grid
        # ----------------------------------------------------

        accepted_records.append(
            {
                "tile_id": tile_id,
                "row_id": row_id,
                "col_id": col_id,
                "cov_pct": 100.0,
                "geometry": tile_geom
            }
        )

        summary_records.append(
            {
                "tile_id": tile_id,
                "row_id": row_id,
                "col_id": col_id,
                "coverage_pct": 100.0,
                "status": "new_pair_created"
            }
        )

        saved += 1

        if saved % 25 == 0:
            print(
                f"Created {saved} new "
                "image-mask pairs..."
            )


# ============================================================
# 9. SAVE DIAGNOSTIC LAYERS
# ============================================================

if accepted_records:

    accepted_gdf = gpd.GeoDataFrame(
        accepted_records,
        geometry="geometry",
        crs=rejected_gdf.crs
    )

    accepted_gdf.to_file(
        newly_accepted_shp
    )

    print("\nNewly completed grids saved:")
    print(newly_accepted_shp)


if incomplete_records:

    incomplete_gdf = gpd.GeoDataFrame(
        incomplete_records,
        geometry="geometry",
        crs=rejected_gdf.crs
    )

    incomplete_gdf.to_file(
        still_incomplete_shp
    )

    print("\nStill-incomplete grids saved:")
    print(still_incomplete_shp)


# ============================================================
# 10. SAVE CSV SUMMARY
# ============================================================

if summary_records:

    import pandas as pd

    summary_df = pd.DataFrame(
        summary_records
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("REJECTED GRID RECHECK COMPLETE")
print("=" * 70)

print(
    f"Previously rejected grids checked : "
    f"{len(rejected_gdf)}"
)

print(
    f"New image-mask pairs created      : "
    f"{saved}"
)

print(
    f"Still incompletely annotated      : "
    f"{still_incomplete}"
)

print(
    f"Skipped because pair exists       : "
    f"{skipped_existing}"
)

print(
    f"Skipped because of image nodata   : "
    f"{skipped_nodata}"
)

print(
    f"Invalid raster windows            : "
    f"{invalid_window}"
)

print("\nImages:")
print(image_dir)

print("\nMasks:")
print(mask_dir)

print("\nNewly complete grids:")
print(newly_accepted_shp)

print("\nStill incomplete grids:")
print(still_incomplete_shp)

print("\nSummary:")
print(summary_csv)

print("\n")
print("Mask classes:")
print("1  - Waste")
print("2  - Soil")
print("3  - Vegetation")
print("4  - Built-up surface")
print("5  - Built-up surface (paved area)")
print("6  - Water Body")
print("7  - Natural Surface - rocks")
print("8  - Non-persistent object")
print("9  - Ancillary class")
print("10 - Unidentified")
print("0  - Unannotated / nodata")

print("=" * 70)