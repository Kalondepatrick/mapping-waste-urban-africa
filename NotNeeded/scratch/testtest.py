from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
import rasterio


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa"
)

PARTITIONED_DIR = (
    PROJECT_DIR
    / "data"
    / "tiled_data"
    / "partitioned"
)

IMAGE_DIRS = [
    PARTITIONED_DIR / "modelDevelopment" / "images",
    PARTITIONED_DIR / "Validation" / "images",
]

OUTPUT_FILE = (
    PARTITIONED_DIR
    / "modelDevelopment_Validation_extent.gpkg"
)


# ============================================================
# FIND IMAGE TILES
# ============================================================

image_files = []

for image_dir in IMAGE_DIRS:

    files = sorted(image_dir.glob("*.tif"))

    print(image_dir)
    print(f"  Found {len(files):,} TIFFs")

    image_files.extend(files)


print()
print(f"Total image tiles: {len(image_files):,}")


if len(image_files) == 0:
    raise FileNotFoundError(
        "No TIFF image tiles were found."
    )


# ============================================================
# CREATE FOOTPRINT OF EACH IMAGE TILE
# ============================================================

tile_footprints = []

reference_crs = None


for i, image_file in enumerate(image_files, start=1):

    with rasterio.open(image_file) as src:

        # ----------------------------------------------------
        # Establish/check CRS
        # ----------------------------------------------------

        if reference_crs is None:
            reference_crs = src.crs

        elif src.crs != reference_crs:
            raise ValueError(
                f"CRS mismatch:\n"
                f"{image_file}\n"
                f"Found: {src.crs}\n"
                f"Expected: {reference_crs}"
            )

        # ----------------------------------------------------
        # Raster footprint from its spatial bounds
        # ----------------------------------------------------

        bounds = src.bounds

        footprint = box(
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )

        tile_footprints.append(footprint)


print()
print("Dissolving tile extents...")


# ============================================================
# DISSOLVE ALL 1,026 TILE FOOTPRINTS
# ============================================================

combined_extent = unary_union(tile_footprints)


# ============================================================
# CREATE OUTPUT
# ============================================================

extent_gdf = gpd.GeoDataFrame(
    {
        "name": [
            "modelDevelopment_Validation_image_extent"
        ],
        "n_tiles": [
            len(image_files)
        ],
    },
    geometry=[combined_extent],
    crs=reference_crs,
)


# Calculate actual combined footprint area
extent_gdf["area_m2"] = (
    extent_gdf.geometry.area
)

extent_gdf["area_km2"] = (
    extent_gdf["area_m2"] / 1_000_000
)


# ============================================================
# SAVE ONE GPKG
# ============================================================

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()


extent_gdf.to_file(
    OUTPUT_FILE,
    layer="image_extent",
    driver="GPKG",
)


# ============================================================
# SUMMARY
# ============================================================

area_m2 = extent_gdf["area_m2"].iloc[0]
area_km2 = extent_gdf["area_km2"].iloc[0]


print()
print("=" * 60)
print("COMPLETE")
print("=" * 60)

print(
    f"Tiles included: "
    f"{len(image_files):,}"
)

print(
    f"Extent area:    "
    f"{area_m2:,.2f} m²"
)

print(
    f"Extent area:    "
    f"{area_km2:.4f} km²"
)

print()
print("Saved to:")
print(OUTPUT_FILE)