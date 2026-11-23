from pathlib import Path
import geopandas as gpd
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_PATH = (
    Path.home()
    / "Documents"
    / "GitHub"
    / "mapping-waste-urban-africa"
)

INPUT_PATH = (
    BASE_PATH
    / "results"
    / "hard_example_object_sampling"
    / "selected_inspection_areas.gpkg"
)

OUTPUT_PATH = (
    BASE_PATH
    / "results"
    / "hard_example_object_sampling"
    / "confirmed_false_positives.gpkg"
)


# =============================================================================
# READ INSPECTION AREAS
# =============================================================================

gdf = gpd.read_file(
    INPUT_PATH
)

print(
    f"Total inspection areas: {len(gdf)}"
)

print(
    f"Fields: {list(gdf.columns)}"
)


# =============================================================================
# CHECK FP FIELD
# =============================================================================

if "fp" not in gdf.columns:
    raise ValueError(
        "The layer does not contain a field called 'fp'."
    )


# Convert safely to numeric in case QGIS stored it as text
gdf["fp"] = pd.to_numeric(
    gdf["fp"],
    errors="coerce"
)


# =============================================================================
# KEEP CONFIRMED FALSE POSITIVES
# =============================================================================

false_positives = (
    gdf[
        gdf["fp"] == 1
    ]
    .copy()
)


print(
    f"Confirmed false positives: {len(false_positives)}"
)


if false_positives.empty:
    raise ValueError(
        "No features with fp = 1 were found."
    )


# =============================================================================
# SAVE
# =============================================================================

false_positives.to_file(
    OUTPUT_PATH,
    layer="confirmed_false_positives",
    driver="GPKG"
)


print("\nSaved:")
print(
    OUTPUT_PATH
)