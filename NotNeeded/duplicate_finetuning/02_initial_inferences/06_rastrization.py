# =============================================================================
# CREATE 256 x 256 IMAGE-MASK PAIRS FROM THE 18 FULLY ANNOTATED
# FALSE-POSITIVE INSPECTION AREAS
#
# INPUTS
# -------
# Full orthomosaic:
#   data/optical.tiff
#
# Complete manual multiclass digitisation:
#   data/aoi/digitization_finetuning.shp
#
# Confirmed false-positive inspection areas:
#   results/hard_example_object_sampling/confirmed_false_positives.gpkg
#
#
# OUTPUTS
# -------
# data/tiled_data/finetuning_hard_negatives_complete/
#
#   images/
#       fp_002_0001.tif
#       ...
#
#   masks/
#       fp_002_0001.tif
#       ...
#
#   qa/
#       accepted_training_tiles.gpkg
#       rejected_training_tiles.gpkg
#       annotation_coverage_by_area.csv
#       repaired_manual_labels.gpkg
#       repaired_inspection_areas.gpkg
#
#   finetuning_manifest.csv
#   class_summary.csv
#   class_lookup.csv
#
#
# IMPORTANT
# ---------
# 1. The 18 inspection areas have been completely manually classified.
#
# 2. Class 0 means UNANNOTATED / NODATA.
#
# 3. A training tile is accepted only if:
#       - it is 256 x 256 pixels;
#       - it lies completely inside an inspection area;
#       - every mask pixel has a class value 1-10;
#       - the corresponding RGB image contains valid imagery.
#
# 4. Invalid manually digitised geometries are repaired before rasterisation.
#
# =============================================================================


from pathlib import Path
import math
import re
import shutil

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.windows import Window
from rasterio.features import (
    rasterize,
    geometry_mask
)

from shapely.geometry import (
    box,
    mapping,
    Polygon,
    MultiPolygon,
    GeometryCollection
)

from shapely.ops import unary_union


# =============================================================================
# 1. SHAPELY MAKE_VALID
# =============================================================================

try:

    from shapely import make_valid

except ImportError:

    make_valid = None


# =============================================================================
# 2. PATHS
# =============================================================================

BASE_PATH = (
    Path.home()
    / "Documents"
    / "GitHub"
    / "mapping-waste-urban-africa"
)


# Full RGB orthomosaic
IMAGE_PATH = (
    BASE_PATH
    / "data"
    / "optical.tiff"
)


# Complete manual multiclass digitisation
DIGITIZATION_PATH = (
    BASE_PATH
    / "data"
    / "aoi"
    / "digitization_finetuning.shp"
)


# The 18 confirmed false-positive inspection areas
INSPECTION_AREAS_PATH = (
    BASE_PATH
    / "results"
    / "hard_example_object_sampling"
    / "confirmed_false_positives.gpkg"
)


# New clean output directory
OUTPUT_BASE = (
    BASE_PATH
    / "data"
    / "tiled_data"
    / "finetuning_hard_negatives_complete"
)


IMAGE_OUTPUT_DIR = (
    OUTPUT_BASE
    / "images"
)


MASK_OUTPUT_DIR = (
    OUTPUT_BASE
    / "masks"
)


QA_DIR = (
    OUTPUT_BASE
    / "qa"
)


# =============================================================================
# 3. CLASS DEFINITIONS
# =============================================================================

CLASS_NAMES = {

    1:
        "Waste",

    2:
        "Soil",

    3:
        "Vegetation",

    4:
        "Built-up surface",

    5:
        "Built-up surface (paved area)",

    6:
        "Water Body",

    7:
        "Natural Surface - rocks",

    8:
        "Non-persistent object",

    9:
        "Ancillary class",

    10:
        "Unidentified"

}


VALID_LABELS = set(
    CLASS_NAMES.keys()
)


# =============================================================================
# 4. CLASS ATTRIBUTE
# =============================================================================

# Leave as None for automatic detection.
#
# Your current digitisation fields are:
#
#   id
#   class
#
# so this script should automatically select "class".

CLASS_FIELD = None


POSSIBLE_CLASS_FIELDS = [

    "class",
    "label",
    "class_id",
    "classid",
    "value",
    "code",
    "manual_lbl",
    "manual_cls",
    "gridcode",
    "dn"

]


# =============================================================================
# 5. TILE SETTINGS
# =============================================================================

TILE_SIZE = 256


# 50% overlap for fine-tuning.
#
# These are deliberately selected TRAINING examples, not independent
# validation observations, so overlap is acceptable here.
STRIDE = 128


# Every pixel in an accepted mask must be labelled 1-10.
REQUIRE_COMPLETE_MASK = True


# Pixel-centre rasterisation.
ALL_TOUCHED = False


# =============================================================================
# 6. RESET OUTPUT DIRECTORY
# =============================================================================

print("\n")
print("=" * 80)
print("PREPARING OUTPUT DIRECTORY")
print("=" * 80)


if OUTPUT_BASE.exists():

    print(
        f"Removing previous output:\n"
        f"{OUTPUT_BASE}"
    )

    shutil.rmtree(
        OUTPUT_BASE
    )


for directory in [

    IMAGE_OUTPUT_DIR,
    MASK_OUTPUT_DIR,
    QA_DIR

]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# =============================================================================
# 7. CHECK INPUT FILES
# =============================================================================

print("\n")
print("=" * 80)
print("CHECKING INPUTS")
print("=" * 80)


for path in [

    IMAGE_PATH,
    DIGITIZATION_PATH,
    INSPECTION_AREAS_PATH

]:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired input not found:\n{path}"
        )


print(
    f"\nOrthomosaic:\n"
    f"{IMAGE_PATH}"
)


print(
    f"\nManual labels:\n"
    f"{DIGITIZATION_PATH}"
)


print(
    f"\nConfirmed false-positive areas:\n"
    f"{INSPECTION_AREAS_PATH}"
)


# =============================================================================
# 8. GEOMETRY REPAIR FUNCTION
# =============================================================================

def repair_polygon_geometry(geom):

    """
    Repair an invalid geometry and retain only polygonal components.

    make_valid() may return:
        Polygon
        MultiPolygon
        GeometryCollection

    GeometryCollections can contain lines/points generated while repairing
    self-intersections. Only polygonal components are retained.
    """

    if geom is None:

        return None


    if geom.is_empty:

        return None


    # -------------------------------------------------------------------------
    # Repair if invalid
    # -------------------------------------------------------------------------

    if not geom.is_valid:

        repaired = None


        if make_valid is not None:

            try:

                repaired = make_valid(
                    geom
                )

            except Exception:

                repaired = None


        # Standard fallback
        if (
            repaired is None
            or repaired.is_empty
        ):

            try:

                repaired = geom.buffer(
                    0
                )

            except Exception:

                return None


        geom = repaired


    if (
        geom is None
        or geom.is_empty
    ):

        return None


    # -------------------------------------------------------------------------
    # Already polygonal
    # -------------------------------------------------------------------------

    if isinstance(
        geom,
        Polygon
    ):

        return geom


    if isinstance(
        geom,
        MultiPolygon
    ):

        polygon_parts = [

            part

            for part in geom.geoms

            if (
                part is not None
                and not part.is_empty
                and part.area > 0
            )

        ]


        if len(
            polygon_parts
        ) == 0:

            return None


        # unary_union is safer than manually constructing a MultiPolygon
        # if repaired pieces touch or overlap.
        return unary_union(
            polygon_parts
        )


    # -------------------------------------------------------------------------
    # GeometryCollection
    # -------------------------------------------------------------------------

    if isinstance(
        geom,
        GeometryCollection
    ):

        polygon_parts = []


        for part in geom.geoms:

            if isinstance(
                part,
                Polygon
            ):

                if (
                    not part.is_empty
                    and part.area > 0
                ):

                    polygon_parts.append(
                        part
                    )


            elif isinstance(
                part,
                MultiPolygon
            ):

                for polygon_part in (
                    part.geoms
                ):

                    if (
                        not polygon_part.is_empty
                        and polygon_part.area > 0
                    ):

                        polygon_parts.append(
                            polygon_part
                        )


        if len(
            polygon_parts
        ) == 0:

            return None


        return unary_union(
            polygon_parts
        )


    # Anything else is not useful for mask rasterisation
    return None


# =============================================================================
# 9. SAFE INTERSECTION FUNCTION
# =============================================================================

def safe_intersection(
    geom1,
    geom2
):

    """
    Intersect two polygons robustly.

    Used because even apparently valid manually digitised polygons may
    occasionally trigger a GEOS topology exception during overlay.
    """

    if (
        geom1 is None
        or geom2 is None
    ):

        return None


    if (
        geom1.is_empty
        or geom2.is_empty
    ):

        return None


    try:

        result = geom1.intersection(
            geom2
        )


    except Exception:

        geom1 = repair_polygon_geometry(
            geom1
        )

        geom2 = repair_polygon_geometry(
            geom2
        )


        if (
            geom1 is None
            or geom2 is None
        ):

            return None


        try:

            result = geom1.intersection(
                geom2
            )

        except Exception:

            # Last-resort zero-width cleanup
            try:

                result = (
                    geom1.buffer(0)
                    .intersection(
                        geom2.buffer(0)
                    )
                )

            except Exception:

                return None


    return repair_polygon_geometry(
        result
    )


# =============================================================================
# 10. LOAD MANUAL DIGITISATION
# =============================================================================

print("\n")
print("=" * 80)
print("LOADING MANUAL DIGITISATION")
print("=" * 80)


labels = gpd.read_file(
    DIGITIZATION_PATH
)


labels = labels[
    labels.geometry.notna()
    & (~labels.geometry.is_empty)
].copy()


if labels.empty:

    raise ValueError(
        "digitization_finetuning.shp contains no valid features."
    )


print(
    f"Manual polygons loaded: "
    f"{len(labels):,}"
)


print(
    "\nAvailable digitisation fields:"
)


for field in labels.columns:

    if field != "geometry":

        print(
            f"  {field}"
        )


# =============================================================================
# 11. LOAD CONFIRMED FALSE-POSITIVE AREAS
# =============================================================================

areas = gpd.read_file(
    INSPECTION_AREAS_PATH
)


areas = areas[
    areas.geometry.notna()
    & (~areas.geometry.is_empty)
].copy()


if areas.empty:

    raise ValueError(
        "confirmed_false_positives.gpkg contains no valid features."
    )


# Safety: retain fp == 1 if field is still present
if "fp" in areas.columns:

    areas["fp"] = pd.to_numeric(
        areas["fp"],
        errors="coerce"
    )


    areas = areas[
        areas["fp"] == 1
    ].copy()


print(
    f"\nConfirmed FP inspection areas: "
    f"{len(areas)}"
)


if len(
    areas
) != 18:

    print(
        "\nWARNING:"
        f" Expected 18 confirmed false-positive areas "
        f"but found {len(areas)}."
    )


# =============================================================================
# 12. IDENTIFY CLASS FIELD
# =============================================================================

if CLASS_FIELD is None:

    field_lookup = {

        str(
            column
        ).lower():
            column

        for column in labels.columns

        if column != "geometry"

    }


    detected_field = None


    for candidate in POSSIBLE_CLASS_FIELDS:

        if candidate.lower() in field_lookup:

            detected_field = (
                field_lookup[
                    candidate.lower()
                ]
            )

            break


    if detected_field is None:

        raise ValueError(
            "\nCould not identify the class attribute automatically.\n\n"
            f"Fields found:\n{list(labels.columns)}\n\n"
            "Set CLASS_FIELD manually near the top of the script."
        )


    CLASS_FIELD = (
        detected_field
    )


print(
    f"\nUsing class field: "
    f"{CLASS_FIELD}"
)


# =============================================================================
# 13. CONVERT CLASS VALUES TO RAW LABELS 1-10
# =============================================================================

def normalize_text(value):

    text = str(
        value
    ).strip().lower()


    text = re.sub(
        r"[_\-]+",
        " ",
        text
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text


TEXT_TO_LABEL = {

    "waste":
        1,

    "soil":
        2,

    "vegetation":
        3,

    "built up surface":
        4,

    "builtup surface":
        4,

    "built up":
        4,

    "builtup":
        4,

    "built up surface paved area":
        5,

    "built up surface (paved area)":
        5,

    "paved area":
        5,

    "paved":
        5,

    "water body":
        6,

    "water":
        6,

    "natural surface rocks":
        7,

    "natural surface rock":
        7,

    "rocks":
        7,

    "rock":
        7,

    "non persistent object":
        8,

    "nonpersistent object":
        8,

    "ancillary class":
        9,

    "ancillary":
        9,

    "unidentified":
        10

}


def convert_class(
    value
):

    # -------------------------------------------------------------------------
    # Numeric input
    # -------------------------------------------------------------------------

    try:

        number = float(
            value
        )


        if number.is_integer():

            raw_label = int(
                number
            )


            if raw_label in VALID_LABELS:

                return raw_label


    except (
        ValueError,
        TypeError
    ):

        pass


    # -------------------------------------------------------------------------
    # Character class name
    # -------------------------------------------------------------------------

    text = normalize_text(
        value
    )


    return TEXT_TO_LABEL.get(
        text,
        None
    )


labels[
    "_raw_label"
] = labels[
    CLASS_FIELD
].apply(
    convert_class
)


# =============================================================================
# 14. CHECK FOR UNKNOWN CLASS VALUES
# =============================================================================

unknown = labels[
    labels[
        "_raw_label"
    ].isna()
]


if not unknown.empty:

    print("\n")
    print("=" * 80)
    print("UNKNOWN CLASS VALUES")
    print("=" * 80)


    unknown_values = sorted(

        unknown[
            CLASS_FIELD
        ]
        .astype(str)
        .unique()
        .tolist()

    )


    for value in unknown_values:

        print(
            f"  {value}"
        )


    raise ValueError(
        "\nSome manual polygons could not be mapped to the "
        "existing raw labels 1-10."
    )


labels[
    "_raw_label"
] = labels[
    "_raw_label"
].astype(
    np.uint8
)


# =============================================================================
# 15. PRINT MANUAL CLASS CONTENT
# =============================================================================

print("\n")
print("=" * 80)
print("MANUAL CLASS CONTENT")
print("=" * 80)


polygon_counts = (
    labels[
        "_raw_label"
    ]
    .value_counts()
    .sort_index()
)


for raw_label, count in (
    polygon_counts.items()
):

    print(
        f"{raw_label:2d}  "
        f"{CLASS_NAMES[raw_label]:35s} "
        f"{count:6,d} polygons"
    )


# =============================================================================
# 16. OPEN ORTHOMOSAIC AND REPROJECT VECTORS
# =============================================================================

with rasterio.open(
    IMAGE_PATH
) as src:


    if src.crs is None:

        raise ValueError(
            "Orthomosaic has no CRS."
        )


    labels = labels.to_crs(
        src.crs
    )


    areas = areas.to_crs(
        src.crs
    )


    print("\n")
    print("=" * 80)
    print("ORTHOMOSAIC")
    print("=" * 80)


    print(
        f"CRS: "
        f"{src.crs}"
    )


    print(
        f"Raster size: "
        f"{src.width:,} x "
        f"{src.height:,}"
    )


    print(
        f"Pixel size: "
        f"{abs(src.transform.a):.4f} x "
        f"{abs(src.transform.e):.4f}"
    )


# =============================================================================
# 17. CHECK AND REPAIR GEOMETRY
# =============================================================================

print("\n")
print("=" * 80)
print("CHECKING GEOMETRY VALIDITY")
print("=" * 80)


n_invalid_labels_before = int(
    (
        ~labels.geometry.is_valid
    ).sum()
)


n_invalid_areas_before = int(
    (
        ~areas.geometry.is_valid
    ).sum()
)


print(
    f"Invalid manual class polygons before repair: "
    f"{n_invalid_labels_before}"
)


print(
    f"Invalid inspection-area polygons before repair: "
    f"{n_invalid_areas_before}"
)


# -------------------------------------------------------------------------
# Repair all geometries, not just the ones currently flagged invalid.
#
# This makes the subsequent intersection step more robust.
# -------------------------------------------------------------------------

labels[
    "geometry"
] = labels.geometry.apply(
    repair_polygon_geometry
)


areas[
    "geometry"
] = areas.geometry.apply(
    repair_polygon_geometry
)


# Remove anything that could not be repaired
labels = labels[
    labels.geometry.notna()
    & (~labels.geometry.is_empty)
].copy()


areas = areas[
    areas.geometry.notna()
    & (~areas.geometry.is_empty)
].copy()


n_invalid_labels_after = int(
    (
        ~labels.geometry.is_valid
    ).sum()
)


n_invalid_areas_after = int(
    (
        ~areas.geometry.is_valid
    ).sum()
)


print(
    f"Invalid manual class polygons after repair: "
    f"{n_invalid_labels_after}"
)


print(
    f"Invalid inspection-area polygons after repair: "
    f"{n_invalid_areas_after}"
)


if (
    n_invalid_labels_after > 0
    or n_invalid_areas_after > 0
):

    raise RuntimeError(
        "Some vector geometries remain invalid after automatic repair."
    )


# =============================================================================
# 18. SAVE REPAIRED VECTORS FOR QA
# =============================================================================

labels.to_file(

    QA_DIR
    / "repaired_manual_labels.gpkg",

    layer="manual_labels",

    driver="GPKG"

)


areas.to_file(

    QA_DIR
    / "repaired_inspection_areas.gpkg",

    layer="inspection_areas",

    driver="GPKG"

)


# =============================================================================
# 19. TILE POSITION FUNCTION
# =============================================================================

def tile_positions(
    length,
    tile_size,
    stride
):

    """
    Return possible local tile starts.

    The final tile is explicitly positioned against the far edge so useful
    imagery is not lost merely because the dimensions are not divisible
    by the stride.
    """

    if length < tile_size:

        return []


    positions = list(

        range(
            0,
            length - tile_size + 1,
            stride
        )

    )


    final_position = (
        length
        - tile_size
    )


    if final_position not in positions:

        positions.append(
            final_position
        )


    return sorted(
        set(
            positions
        )
    )


# =============================================================================
# 20. STORAGE
# =============================================================================

accepted_records = []

rejected_records = []

manifest_records = []

coverage_records = []


accepted = 0

rejected_outside_area = 0

rejected_incomplete_mask = 0

rejected_image_nodata = 0

areas_too_small = 0


# Avoid duplicate global windows if two inspection areas overlap
used_global_windows = set()


# =============================================================================
# 21. GENERATE IMAGE-MASK PAIRS
# =============================================================================

print("\n")
print("=" * 80)
print("GENERATING LOCAL IMAGE-MASK PAIRS")
print("=" * 80)


with rasterio.open(
    IMAGE_PATH
) as src:


    # Build spatial index AFTER geometry repair
    label_sindex = (
        labels.sindex
    )


    for area_counter, (
        area_index,
        area_row
    ) in enumerate(

        areas.iterrows(),
        start=1

    ):


        area_geometry = (
            area_row.geometry
        )


        if (
            "sample_id"
            in areas.columns
        ):

            sample_id = int(
                area_row[
                    "sample_id"
                ]
            )


        else:

            sample_id = (
                area_counter
            )


        print("\n")
        print("-" * 80)


        print(
            f"Inspection area "
            f"{area_counter}/{len(areas)} "
            f"(sample_id={sample_id})"
        )


        # =====================================================================
        # 21.1 WINDOW COVERING THE INSPECTION AREA
        # =====================================================================

        minx, miny, maxx, maxy = (
            area_geometry.bounds
        )


        area_window = (
            rasterio.windows.from_bounds(

                minx,
                miny,
                maxx,
                maxy,

                transform=src.transform

            )
        )


        area_window = (
            area_window
            .round_offsets()
            .round_lengths()
        )


        # Ensure window remains inside full orthomosaic
        area_window = (
            area_window.intersection(

                Window(
                    0,
                    0,
                    src.width,
                    src.height
                )

            )
        )


        area_height = int(
            area_window.height
        )


        area_width = int(
            area_window.width
        )


        area_transform = (
            src.window_transform(
                area_window
            )
        )


        print(
            f"  Local raster dimensions: "
            f"{area_width} x "
            f"{area_height} pixels"
        )


        # =====================================================================
        # 21.2 EXACT PIXELS INSIDE THIS INSPECTION AREA
        # =====================================================================

        inside_area = geometry_mask(

            [
                mapping(
                    area_geometry
                )
            ],

            out_shape=(
                area_height,
                area_width
            ),

            transform=area_transform,

            invert=True

        )


        # =====================================================================
        # 21.3 FIND MANUAL LABELS INTERSECTING THE AREA
        # =====================================================================

        label_indices = list(

            label_sindex.query(

                area_geometry,

                predicate="intersects"

            )

        )


        if len(
            label_indices
        ) == 0:

            raise RuntimeError(
                f"No manual class polygons intersect "
                f"inspection area sample_id={sample_id}."
            )


        local_labels = (
            labels.iloc[
                label_indices
            ]
            .copy()
        )


        # =====================================================================
        # 21.4 ROBUSTLY CLIP LABEL POLYGONS TO INSPECTION AREA
        # =====================================================================

        clipped_geometries = []


        for label_geometry in (
            local_labels.geometry
        ):

            clipped = safe_intersection(

                label_geometry,
                area_geometry

            )


            clipped_geometries.append(
                clipped
            )


        local_labels[
            "geometry"
        ] = clipped_geometries


        local_labels = local_labels[
            local_labels.geometry.notna()
            & (~local_labels.geometry.is_empty)
        ].copy()


        if local_labels.empty:

            raise RuntimeError(
                f"All intersecting label geometries disappeared "
                f"after clipping for sample_id={sample_id}."
            )


        # =====================================================================
        # 21.5 RASTERIZE MANUAL CLASSES
        # =====================================================================

        raster_shapes = [

            (
                geom,
                int(
                    raw_label
                )
            )

            for geom, raw_label in zip(

                local_labels.geometry,

                local_labels[
                    "_raw_label"
                ]

            )

        ]


        local_mask = rasterize(

            raster_shapes,

            out_shape=(
                area_height,
                area_width
            ),

            transform=area_transform,

            fill=0,

            all_touched=ALL_TOUCHED,

            dtype=np.uint8

        )


        # ---------------------------------------------------------------------
        # Outside inspection boundary MUST remain zero.
        # -------------------------------------------------------------------------

        local_mask[
            ~inside_area
        ] = 0


        # =====================================================================
        # 21.6 CHECK ANNOTATION COVERAGE
        # =====================================================================

        n_inside = int(
            inside_area.sum()
        )


        n_labelled = int(
            np.sum(

                inside_area
                & (
                    local_mask > 0
                )

            )
        )


        if n_inside > 0:

            coverage = (
                n_labelled
                / n_inside
            )

        else:

            coverage = 0.0


        missing_pixels = (
            n_inside
            - n_labelled
        )


        coverage_records.append(
            {

                "sample_id":
                    sample_id,

                "inside_pixels":
                    n_inside,

                "labelled_pixels":
                    n_labelled,

                "missing_pixels":
                    missing_pixels,

                "coverage":
                    coverage,

                "coverage_pct":
                    (
                        coverage
                        * 100.0
                    )

            }
        )


        print(
            f"  Annotation coverage: "
            f"{coverage * 100:.4f}%"
        )


        if missing_pixels > 0:

            print(
                f"  WARNING: "
                f"{missing_pixels:,} pixels inside "
                f"the inspection area remain unlabelled."
            )


        # =====================================================================
        # 21.7 CANDIDATE TILE POSITIONS
        # =====================================================================

        row_positions = tile_positions(

            area_height,
            TILE_SIZE,
            STRIDE

        )


        col_positions = tile_positions(

            area_width,
            TILE_SIZE,
            STRIDE

        )


        if (
            len(row_positions) == 0
            or len(col_positions) == 0
        ):

            areas_too_small += 1


            print(
                "  Area is smaller than one "
                "complete 256 x 256 tile."
            )


            continue


        possible_here = (

            len(row_positions)
            * len(col_positions)

        )


        print(
            f"  Potential local tiles: "
            f"{possible_here}"
        )


        accepted_here = 0


        # =====================================================================
        # 21.8 TEST EACH CANDIDATE TILE
        # =====================================================================

        for local_row in row_positions:

            for local_col in col_positions:


                row_end = (
                    local_row
                    + TILE_SIZE
                )


                col_end = (
                    local_col
                    + TILE_SIZE
                )


                area_chip = inside_area[

                    local_row:
                    row_end,

                    local_col:
                    col_end

                ]


                mask_chip = local_mask[

                    local_row:
                    row_end,

                    local_col:
                    col_end

                ]


                # -------------------------------------------------------------
                # Defensive dimensions check
                # -------------------------------------------------------------

                if area_chip.shape != (
                    TILE_SIZE,
                    TILE_SIZE
                ):

                    continue


                if mask_chip.shape != (
                    TILE_SIZE,
                    TILE_SIZE
                ):

                    continue


                # -------------------------------------------------------------
                # Tile must lie wholly inside inspection area
                # -------------------------------------------------------------

                if not np.all(
                    area_chip
                ):

                    rejected_outside_area += 1


                    global_row_tmp = (
                        int(
                            area_window.row_off
                        )
                        + local_row
                    )


                    global_col_tmp = (
                        int(
                            area_window.col_off
                        )
                        + local_col
                    )


                    temp_window = Window(

                        global_col_tmp,
                        global_row_tmp,
                        TILE_SIZE,
                        TILE_SIZE

                    )


                    temp_bounds = (
                        rasterio.windows.bounds(

                            temp_window,
                            src.transform

                        )
                    )


                    rejected_records.append(
                        {

                            "sample_id":
                                sample_id,

                            "reason":
                                "crosses_inspection_boundary",

                            "geometry":
                                box(
                                    *temp_bounds
                                )

                        }
                    )


                    continue


                # -------------------------------------------------------------
                # Tile must contain class labels 1-10 everywhere
                # -------------------------------------------------------------

                if (
                    REQUIRE_COMPLETE_MASK
                    and np.any(
                        mask_chip == 0
                    )
                ):

                    rejected_incomplete_mask += 1


                    global_row_tmp = (
                        int(
                            area_window.row_off
                        )
                        + local_row
                    )


                    global_col_tmp = (
                        int(
                            area_window.col_off
                        )
                        + local_col
                    )


                    temp_window = Window(

                        global_col_tmp,
                        global_row_tmp,
                        TILE_SIZE,
                        TILE_SIZE

                    )


                    temp_bounds = (
                        rasterio.windows.bounds(

                            temp_window,
                            src.transform

                        )
                    )


                    rejected_records.append(
                        {

                            "sample_id":
                                sample_id,

                            "reason":
                                "incomplete_mask",

                            "geometry":
                                box(
                                    *temp_bounds
                                )

                        }
                    )


                    continue


                # -------------------------------------------------------------
                # Convert local window to full orthomosaic coordinates
                # -------------------------------------------------------------

                global_row = (

                    int(
                        area_window.row_off
                    )
                    + local_row

                )


                global_col = (

                    int(
                        area_window.col_off
                    )
                    + local_col

                )


                global_key = (
                    global_row,
                    global_col
                )


                # Skip accidental duplicate windows
                if global_key in used_global_windows:

                    continue


                global_window = Window(

                    global_col,
                    global_row,
                    TILE_SIZE,
                    TILE_SIZE

                )


                # -------------------------------------------------------------
                # Read RGB chip directly from original orthomosaic
                # -------------------------------------------------------------

                image_chip = src.read(

                    [1, 2, 3],

                    window=global_window

                )


                if image_chip.shape != (
                    3,
                    TILE_SIZE,
                    TILE_SIZE
                ):

                    rejected_image_nodata += 1

                    continue


                # -------------------------------------------------------------
                # Nodata test
                # -------------------------------------------------------------

                if src.nodata is not None:

                    invalid_pixels = np.all(

                        image_chip
                        == src.nodata,

                        axis=0

                    )


                else:

                    # Orthomosaic background convention
                    invalid_pixels = np.all(

                        image_chip
                        == 0,

                        axis=0

                    )


                if np.any(
                    invalid_pixels
                ):

                    rejected_image_nodata += 1


                    tile_bounds = (
                        rasterio.windows.bounds(

                            global_window,
                            src.transform

                        )
                    )


                    rejected_records.append(
                        {

                            "sample_id":
                                sample_id,

                            "reason":
                                "image_nodata",

                            "geometry":
                                box(
                                    *tile_bounds
                                )

                        }
                    )


                    continue


                # =============================================================
                # ACCEPT IMAGE-MASK PAIR
                # =============================================================

                used_global_windows.add(
                    global_key
                )


                accepted += 1

                accepted_here += 1


                tile_name = (

                    f"fp_"
                    f"{sample_id:03d}_"
                    f"{accepted:04d}"

                )


                image_output_path = (

                    IMAGE_OUTPUT_DIR
                    / f"{tile_name}.tif"

                )


                mask_output_path = (

                    MASK_OUTPUT_DIR
                    / f"{tile_name}.tif"

                )


                tile_transform = (
                    src.window_transform(
                        global_window
                    )
                )


                # =============================================================
                # SAVE IMAGE
                # =============================================================

                image_profile = (
                    src.profile.copy()
                )


                image_profile.update(
                    {

                        "height":
                            TILE_SIZE,

                        "width":
                            TILE_SIZE,

                        "count":
                            3,

                        "transform":
                            tile_transform,

                        "compress":
                            "lzw",

                        "BIGTIFF":
                            "IF_SAFER"

                    }
                )


                with rasterio.open(

                    image_output_path,
                    "w",
                    **image_profile

                ) as dst:

                    dst.write(
                        image_chip
                    )


                # =============================================================
                # SAVE MASK
                # =============================================================

                mask_profile = (
                    image_profile.copy()
                )


                mask_profile.update(
                    {

                        "count":
                            1,

                        "dtype":
                            "uint8",

                        "nodata":
                            0

                    }
                )


                with rasterio.open(

                    mask_output_path,
                    "w",
                    **mask_profile

                ) as dst:

                    dst.write(

                        mask_chip.astype(
                            np.uint8
                        ),

                        1

                    )


                # =============================================================
                # TILE FOOTPRINT
                # =============================================================

                tile_bounds = (
                    rasterio.windows.bounds(

                        global_window,
                        src.transform

                    )
                )


                footprint = box(
                    *tile_bounds
                )


                accepted_records.append(
                    {

                        "tile_id":
                            tile_name,

                        "sample_id":
                            sample_id,

                        "row_off":
                            global_row,

                        "col_off":
                            global_col,

                        "geometry":
                            footprint

                    }
                )


                # =============================================================
                # MANIFEST
                # =============================================================

                manifest_record = {

                    "tile_id":
                        tile_name,

                    "sample_id":
                        sample_id,

                    "image_file":
                        image_output_path.name,

                    "mask_file":
                        mask_output_path.name,

                    "row_off":
                        global_row,

                    "col_off":
                        global_col

                }


                for raw_label in range(
                    1,
                    11
                ):

                    manifest_record[
                        f"class_{raw_label}_pixels"
                    ] = int(

                        np.sum(
                            mask_chip
                            == raw_label
                        )

                    )


                manifest_records.append(
                    manifest_record
                )


        print(
            f"  Accepted tiles: "
            f"{accepted_here}"
        )


# =============================================================================
# 22. SAVE ACCEPTED TILE FOOTPRINTS
# =============================================================================

if accepted_records:

    accepted_gdf = gpd.GeoDataFrame(

        accepted_records,

        geometry="geometry",

        crs=areas.crs

    )


    accepted_gdf.to_file(

        QA_DIR
        / "accepted_training_tiles.gpkg",

        layer="accepted_tiles",

        driver="GPKG"

    )


# =============================================================================
# 23. SAVE REJECTED TILE FOOTPRINTS
# =============================================================================

if rejected_records:

    rejected_gdf = gpd.GeoDataFrame(

        rejected_records,

        geometry="geometry",

        crs=areas.crs

    )


    rejected_gdf.to_file(

        QA_DIR
        / "rejected_training_tiles.gpkg",

        layer="rejected_tiles",

        driver="GPKG"

    )


# =============================================================================
# 24. SAVE ANNOTATION COVERAGE
# =============================================================================

coverage_df = pd.DataFrame(
    coverage_records
)


coverage_path = (

    QA_DIR
    / "annotation_coverage_by_area.csv"

)


coverage_df.to_csv(
    coverage_path,
    index=False
)


# =============================================================================
# 25. SAVE MANIFEST
# =============================================================================

manifest = pd.DataFrame(
    manifest_records
)


manifest_path = (

    OUTPUT_BASE
    / "finetuning_manifest.csv"

)


manifest.to_csv(
    manifest_path,
    index=False
)


# =============================================================================
# 26. CLASS SUMMARY
# =============================================================================

class_summary_records = []


if not manifest.empty:

    total_training_pixels = (

        len(
            manifest
        )
        * TILE_SIZE
        * TILE_SIZE

    )


    for raw_label in range(
        1,
        11
    ):

        column_name = (
            f"class_{raw_label}_pixels"
        )


        pixel_count = int(

            manifest[
                column_name
            ].sum()

        )


        percentage = (

            100
            * pixel_count
            / total_training_pixels

        )


        class_summary_records.append(
            {

                "raw_label":
                    raw_label,

                "model_channel":
                    raw_label - 1,

                "class_name":
                    CLASS_NAMES[
                        raw_label
                    ],

                "pixel_count":
                    pixel_count,

                "percentage":
                    percentage

            }
        )


class_summary_df = pd.DataFrame(
    class_summary_records
)


class_summary_path = (

    OUTPUT_BASE
    / "class_summary.csv"

)


class_summary_df.to_csv(
    class_summary_path,
    index=False
)


# =============================================================================
# 27. CLASS LOOKUP
# =============================================================================

class_lookup = pd.DataFrame(

    [

        {

            "raw_label":
                raw_label,

            "model_channel":
                raw_label - 1,

            "class_name":
                class_name

        }

        for (
            raw_label,
            class_name
        ) in CLASS_NAMES.items()

    ]

)


class_lookup.to_csv(

    OUTPUT_BASE
    / "class_lookup.csv",

    index=False

)


# =============================================================================
# 28. FINAL REPORT
# =============================================================================

print("\n")
print("=" * 80)
print("FINETUNING IMAGE-MASK CREATION COMPLETE")
print("=" * 80)


print(
    f"\nConfirmed false-positive areas: "
    f"{len(areas)}"
)


print(
    f"Accepted 256 x 256 image-mask pairs: "
    f"{accepted}"
)


print(
    f"Rejected because tile crossed "
    f"inspection-area edge: "
    f"{rejected_outside_area}"
)


print(
    f"Rejected because mask contained "
    f"unlabelled pixels: "
    f"{rejected_incomplete_mask}"
)


print(
    f"Rejected because image contained nodata: "
    f"{rejected_image_nodata}"
)


print(
    f"Areas smaller than a complete "
    f"256 x 256 tile: "
    f"{areas_too_small}"
)


print("\n")
print("=" * 80)
print("ANNOTATION COVERAGE")
print("=" * 80)


if not coverage_df.empty:

    print(

        coverage_df[
            [
                "sample_id",
                "inside_pixels",
                "labelled_pixels",
                "missing_pixels",
                "coverage_pct"
            ]
        ]
        .to_string(
            index=False
        )

    )


print("\n")
print("=" * 80)
print("TRAINING CLASS COMPOSITION")
print("=" * 80)


if not class_summary_df.empty:

    for _, row in (
        class_summary_df.iterrows()
    ):

        print(

            f"{int(row['raw_label']):2d}  "
            f"{row['class_name']:35s} "
            f"{int(row['pixel_count']):12,d} pixels "
            f"({row['percentage']:6.2f}%)"

        )


print("\n")
print("=" * 80)
print("OUTPUTS")
print("=" * 80)


print(
    "\nIMAGE CHIPS:"
)

print(
    IMAGE_OUTPUT_DIR
)


print(
    "\nMASK CHIPS:"
)

print(
    MASK_OUTPUT_DIR
)


print(
    "\nMANIFEST:"
)

print(
    manifest_path
)


print(
    "\nCLASS SUMMARY:"
)

print(
    class_summary_path
)


print(
    "\nANNOTATION COVERAGE:"
)

print(
    coverage_path
)


print(
    "\nACCEPTED/REJECTED TILE FOOTPRINTS:"
)

print(
    QA_DIR
)


print("\n")
print("=" * 80)
print("IMPORTANT")
print("=" * 80)


print(
    "\n0 remains unannotated/nodata."
)


print(
    "Only completely labelled 256 x 256 masks "
    "were accepted."
)


print(
    "\nThe generated pairs are additional fine-tuning "
    "examples and should later be combined with the "
    "existing model-development training data rather "
    "than replacing it."
)


print("\nDone.")