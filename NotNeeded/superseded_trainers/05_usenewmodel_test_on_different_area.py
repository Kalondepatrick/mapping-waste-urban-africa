from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio

from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box
from shapely.ops import unary_union


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

OUTPUT_DIR = (
    r"C:\Users\Peter\Documents\GitHub\waste_spatio_temporal"
    r"\intermediaries\march_training_chips"
)


# ============================================================
# 2. SETTINGS
# ============================================================

PATCH_SIZE = 256

# Field in march.shp containing values 1-10
CLASS_FIELD = "class"

# Every pixel in a chip must be annotated
REQUIRED_COVERAGE = 1.0


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

missing_shp = (
    diagnostic_dir /
    "missing_annotation_extent.shp"
)

rejected_shp = (
    diagnostic_dir /
    "rejected_256_tiles_incomplete_annotation.shp"
)

accepted_shp = (
    diagnostic_dir /
    "accepted_256_tiles.shp"
)


# ============================================================
# 4. READ ANNOTATIONS
# ============================================================

print("=" * 60)
print("READING ANNOTATIONS")
print("=" * 60)

gdf = gpd.read_file(ANNOTATIONS)

print("\nAvailable fields:")
print(gdf.columns.tolist())

if CLASS_FIELD not in gdf.columns:
    raise ValueError(
        f"\nField '{CLASS_FIELD}' does not exist.\n"
        f"Available fields:\n{gdf.columns.tolist()}\n\n"
        "Change CLASS_FIELD near the top of the script."
    )


# ============================================================
# 5. REMOVE EMPTY GEOMETRIES
# ============================================================

gdf = gdf[
    gdf.geometry.notnull() &
    (~gdf.geometry.is_empty)
].copy()

print(f"\nAnnotation polygons: {len(gdf)}")


# ============================================================
# 6. REPAIR INVALID GEOMETRIES
# ============================================================

invalid = ~gdf.geometry.is_valid

if invalid.any():

    print(
        f"Repairing {invalid.sum()} invalid geometries..."
    )

    gdf.loc[invalid, "geometry"] = (
        gdf.loc[invalid, "geometry"].buffer(0)
    )

# Remove anything that became empty after repair
gdf = gdf[
    gdf.geometry.notnull() &
    (~gdf.geometry.is_empty)
].copy()


# ============================================================
# 7. CHECK CLASS FIELD
# ============================================================

# Convert class values to numeric
gdf[CLASS_FIELD] = gdf[CLASS_FIELD].astype(int)

classes_found = sorted(
    gdf[CLASS_FIELD].unique()
)

print("\nClasses found:")
print(classes_found)

bad_classes = [
    x
    for x in classes_found
    if x < 1 or x > 10
]

if bad_classes:

    raise ValueError(
        f"\nUnexpected classes found: {bad_classes}\n"
        "Expected class values between 1 and 10."
    )


# ============================================================
# 8. OPEN MARCH IMAGE
# ============================================================

with rasterio.open(IMAGE_RASTER) as src:

    print("\n")
    print("=" * 60)
    print("MARCH IMAGE")
    print("=" * 60)

    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)
    print("Data type:", src.dtypes)


    # ========================================================
    # 9. CHECK CRS
    # ========================================================

    if gdf.crs is None:

        raise ValueError(
            "march.shp does not have a defined CRS."
        )

    if src.crs is None:

        raise ValueError(
            "The March raster does not have a defined CRS."
        )

    if gdf.crs != src.crs:

        print(
            "\nReprojecting annotations to match "
            "the March raster..."
        )

        gdf = gdf.to_crs(src.crs)


    # ========================================================
    # 10. CALCULATE COMPLETE 256 x 256 GRID
    # ========================================================

    n_rows = src.height // PATCH_SIZE
    n_cols = src.width // PATCH_SIZE

    usable_height = n_rows * PATCH_SIZE
    usable_width = n_cols * PATCH_SIZE

    total_possible = n_rows * n_cols

    print("\n")
    print("=" * 60)
    print("CHIP GRID")
    print("=" * 60)

    print(f"Patch size             : {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Complete rows of chips : {n_rows}")
    print(f"Complete cols of chips : {n_cols}")
    print(f"Potential chips        : {total_possible}")

    # We deliberately ignore any remainder smaller than 256 pixels
    # along the right or bottom edge of the raster.


    # ========================================================
    # 11. DEFINE CHIPPABLE IMAGE EXTENT
    # ========================================================

    full_grid_window = Window(
        col_off=0,
        row_off=0,
        width=usable_width,
        height=usable_height
    )

    grid_left, grid_bottom, grid_right, grid_top = (
        rasterio.windows.bounds(
            full_grid_window,
            src.transform
        )
    )

    grid_extent = box(
        grid_left,
        grid_bottom,
        grid_right,
        grid_top
    )


    # ========================================================
    # 12. CLIP ANNOTATIONS TO CHIPPABLE EXTENT
    # ========================================================

    gdf = gdf[
        gdf.geometry.intersects(grid_extent)
    ].copy()

    gdf["geometry"] = (
        gdf.geometry.intersection(grid_extent)
    )

    gdf = gdf[
        gdf.geometry.notnull() &
        (~gdf.geometry.is_empty)
    ].copy()

    if len(gdf) == 0:

        raise ValueError(
            "\nNo annotation polygons overlap the March raster."
        )

    print(
        f"\nAnnotation polygons overlapping image: {len(gdf)}"
    )


    # ========================================================
    # 13. BUILD ANNOTATION COVERAGE UNION
    # ========================================================

    print("\nBuilding annotation coverage...")

    annotation_union = unary_union(
        gdf.geometry.tolist()
    )


    # ========================================================
    # 14. FIND EXACT AREAS WHERE ANNOTATION IS MISSING
    # ========================================================

    print(
        "Calculating geographical areas where "
        "annotation is missing..."
    )

    missing_geometry = (
        grid_extent.difference(annotation_union)
    )


    # ========================================================
    # 15. FUNCTION TO EXTRACT POLYGONS
    # ========================================================

    def extract_polygons(geometry):
        """
        Convert Polygon, MultiPolygon or GeometryCollection
        into a flat list of Polygon geometries.
        """

        polygons = []

        if geometry.is_empty:
            return polygons

        if geometry.geom_type == "Polygon":

            polygons.append(geometry)

        elif geometry.geom_type == "MultiPolygon":

            polygons.extend(
                list(geometry.geoms)
            )

        elif geometry.geom_type == "GeometryCollection":

            for geom in geometry.geoms:

                if geom.is_empty:
                    continue

                if geom.geom_type == "Polygon":

                    polygons.append(geom)

                elif geom.geom_type == "MultiPolygon":

                    polygons.extend(
                        list(geom.geoms)
                    )

        return polygons


    # ========================================================
    # 16. SAVE EXACT MISSING ANNOTATION EXTENT
    # ========================================================

    missing_geoms = extract_polygons(
        missing_geometry
    )

    if missing_geoms:

        missing_gdf = gpd.GeoDataFrame(
            {
                "gap_id": range(
                    1,
                    len(missing_geoms) + 1
                )
            },
            geometry=missing_geoms,
            crs=src.crs
        )

        # Calculate area if CRS is projected
        if src.crs.is_projected:

            missing_gdf["area_m2"] = (
                missing_gdf.geometry.area
            )

        else:

            missing_gdf["area_m2"] = np.nan

        missing_gdf.to_file(
            missing_shp
        )

        print("\nMissing annotation layer saved:")
        print(missing_shp)

        print(
            f"Number of missing areas: "
            f"{len(missing_gdf)}"
        )

        if src.crs.is_projected:

            print(
                "Total missing area: "
                f"{missing_gdf['area_m2'].sum():,.2f} m²"
            )

    else:

        print(
            "\nThere are no missing annotation areas."
        )


    # ========================================================
    # 17. PREPARE RASTERIZATION SHAPES
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
        (
            geom,
            1
        )
        for geom in gdf.geometry
    ]


    # ========================================================
    # 18. SET UP PROCESSING COUNTERS
    # ========================================================

    saved = 0
    rejected = 0
    skipped_image_nodata = 0

    rejected_records = []
    accepted_records = []


    # ========================================================
    # 19. START CHIPPING
    # ========================================================

    print("\n")
    print("=" * 60)
    print("CREATING 256 x 256 IMAGE-MASK PAIRS")
    print("=" * 60)


    for row_index in range(n_rows):

        row_off = (
            row_index * PATCH_SIZE
        )

        for col_index in range(n_cols):

            col_off = (
                col_index * PATCH_SIZE
            )


            # =================================================
            # DEFINE WINDOW
            # =================================================

            window = Window(
                col_off=col_off,
                row_off=row_off,
                width=PATCH_SIZE,
                height=PATCH_SIZE
            )

            window_transform = (
                src.window_transform(window)
            )


            # =================================================
            # CHIP NAME
            # =================================================

            tile_id = (
                f"march_r{row_index:04d}"
                f"_c{col_index:04d}"
            )


            # =================================================
            # GEOGRAPHICAL FOOTPRINT OF CHIP
            # =================================================

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


            # =================================================
            # RASTERIZE ANNOTATION COVERAGE
            # =================================================

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


            # =================================================
            # REJECT IF CHIP IS NOT 100% ANNOTATED
            # =================================================

            if coverage_fraction < REQUIRED_COVERAGE:

                rejected += 1

                rejected_records.append(
                    {
                        "tile_id": tile_id,
                        "row_id": row_index,
                        "col_id": col_index,
                        "cov_pct":
                            coverage_fraction * 100,
                        "miss_pct":
                            (
                                1 -
                                coverage_fraction
                            ) * 100,
                        "reason":
                            "incomplete_annotation",
                        "geometry": tile_geom
                    }
                )

                continue


            # =================================================
            # READ IMAGE CHIP
            # =================================================

            image = src.read(
                window=window
            )

            if (
                image.shape[1] != PATCH_SIZE
                or
                image.shape[2] != PATCH_SIZE
            ):

                continue


            # =================================================
            # CHECK IMAGE NODATA
            # =================================================

            if src.nodata is not None:

                nodata_mask = np.all(
                    image == src.nodata,
                    axis=0
                )

                if np.any(nodata_mask):

                    skipped_image_nodata += 1

                    rejected_records.append(
                        {
                            "tile_id": tile_id,
                            "row_id": row_index,
                            "col_id": col_index,
                            "cov_pct":
                                coverage_fraction * 100,
                            "miss_pct": 0.0,
                            "reason":
                                "image_nodata",
                            "geometry": tile_geom
                        }
                    )

                    continue


            # =================================================
            # RASTERIZE LAND-COVER CLASSES
            # =================================================

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


            # =================================================
            # FINAL ANNOTATION SAFETY CHECK
            # =================================================

            zero_pixels = np.count_nonzero(
                mask == 0
            )

            if zero_pixels > 0:

                rejected += 1

                zero_fraction = (
                    zero_pixels /
                    total_pixels
                )

                rejected_records.append(
                    {
                        "tile_id": tile_id,
                        "row_id": row_index,
                        "col_id": col_index,
                        "cov_pct":
                            (1 - zero_fraction) * 100,
                        "miss_pct":
                            zero_fraction * 100,
                        "reason":
                            "class_gap",
                        "geometry": tile_geom
                    }
                )

                continue


            # =================================================
            # CHECK MASK VALUES
            # =================================================

            unique_values = np.unique(mask)

            invalid_mask_values = [
                value
                for value in unique_values
                if value < 1 or value > 10
            ]

            if invalid_mask_values:

                print(
                    f"\nWARNING: {tile_id} contains "
                    f"unexpected mask values: "
                    f"{invalid_mask_values}"
                )

                continue


            # =================================================
            # OUTPUT PATHS
            # =================================================

            image_path = (
                image_dir /
                f"{tile_id}.tif"
            )

            mask_path = (
                mask_dir /
                f"{tile_id}.tif"
            )


            # =================================================
            # SAVE IMAGE CHIP
            # =================================================

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


            # =================================================
            # SAVE MASK CHIP
            # =================================================

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


            # =================================================
            # RECORD ACCEPTED TILE
            # =================================================

            accepted_records.append(
                {
                    "tile_id": tile_id,
                    "row_id": row_index,
                    "col_id": col_index,
                    "geometry": tile_geom
                }
            )

            saved += 1


            # =================================================
            # PROGRESS REPORT
            # =================================================

            if saved % 50 == 0:

                print(
                    f"Saved {saved} "
                    f"image-mask pairs..."
                )


    # ========================================================
    # 20. SAVE REJECTED TILE FOOTPRINTS
    # ========================================================

    if rejected_records:

        rejected_gdf = (
            gpd.GeoDataFrame(
                rejected_records,
                geometry="geometry",
                crs=src.crs
            )
        )

        rejected_gdf.to_file(
            rejected_shp
        )

        print(
            "\nRejected tile footprints saved:"
        )
        print(rejected_shp)


    # ========================================================
    # 21. SAVE ACCEPTED TILE FOOTPRINTS
    # ========================================================

    if accepted_records:

        accepted_gdf = (
            gpd.GeoDataFrame(
                accepted_records,
                geometry="geometry",
                crs=src.crs
            )
        )

        accepted_gdf.to_file(
            accepted_shp
        )

        print(
            "\nAccepted tile footprints saved:"
        )
        print(accepted_shp)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("MARCH TRAINING DATA CREATION COMPLETE")
print("=" * 60)

print(
    f"Potential complete 256x256 chips : "
    f"{total_possible}"
)

print(
    f"Image-mask pairs saved           : "
    f"{saved}"
)

print(
    f"Incomplete annotation rejected   : "
    f"{rejected}"
)

print(
    f"Image nodata rejected            : "
    f"{skipped_image_nodata}"
)

print("\n")
print("IMAGE CHIPS:")
print(image_dir)

print("\nMASK CHIPS:")
print(mask_dir)

print("\nEXACT MISSING ANNOTATION:")
print(missing_shp)

print("\nREJECTED 256x256 TILE FOOTPRINTS:")
print(rejected_shp)

print("\nACCEPTED 256x256 TILE FOOTPRINTS:")
print(accepted_shp)


print("\n")
print("=" * 60)
print("MASK CLASSES")
print("=" * 60)

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

print("\n0 = unannotated / nodata")

print(
    "\nIMPORTANT: Any 256x256 chip containing "
    "even one unannotated pixel is excluded."
)

print("=" * 60)