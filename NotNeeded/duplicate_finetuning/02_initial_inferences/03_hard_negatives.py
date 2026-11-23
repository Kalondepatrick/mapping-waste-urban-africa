# =============================================================================
# OBJECTIVE SAMPLING OF HIGH-PROBABILITY U-NET OBJECTS
# FOR HARD-EXAMPLE / FALSE-POSITIVE CHARACTERISATION
#
# PURPOSE
# -------
# 1. Identify contiguous high-probability U-Net objects.
#
# 2. Exclude:
#       a) existing model test area
#       b) additional AOI that must not be targeted
#
# 3. Treat predicted waste OBJECTS as the sampling population.
#
# 4. Calculate the required random sample size:
#
#       n0 = z² p(1-p) / d²
#
#    followed by finite population correction:
#
#       n = N*n0 / (N + n0 - 1)
#
# 5. Randomly sample candidate objects.
#
# 6. Define the manual inspection/digitisation area as:
#
#       predicted object extent
#                 +
#       model-derived contextual imagery
#
#    There is therefore NO fixed 100 x 100 m digitisation area.
#
# 7. Export candidate objects and inspection areas for QGIS.
#
# =============================================================================


from pathlib import Path
import math

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.windows import Window
from rasterio.features import (
    geometry_mask,
    shapes
)

from scipy import ndimage
from scipy.stats import norm

from shapely.geometry import (
    box,
    shape,
    mapping
)
from shapely.ops import unary_union


# =============================================================================
# 1. PATHS
# =============================================================================

BASE_PATH = (
    Path.home()
    / "Documents"
    / "GitHub"
    / "mapping-waste-urban-africa"
)


# -----------------------------------------------------------------------------
# Current probability raster
#
# Once the seam-free inference has FINISHED, change this to:
#
# waste_probability_full_seamfree.tif
#
# Do not point this script at a raster that is still being written.
# -----------------------------------------------------------------------------

PROBABILITY_PATH = (
    BASE_PATH
    / "outputs"
    / "waste_probability_full.tif"
)


# -----------------------------------------------------------------------------
# Existing test area
# -----------------------------------------------------------------------------

TEST_AREA_PATH = (
    BASE_PATH
    / "data"
    / "aoi"
    / "testarea.shp"
)


# -----------------------------------------------------------------------------
# Additional area that must not be targeted
# -----------------------------------------------------------------------------

EXCLUSION_AOI_PATH = (
    Path.home()
    / "Documents"
    / "GitHub"
    / "waste_spatio_temporal"
    / "inputs"
    / "aoi"
    / "aoi.geojson"
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

OUTPUT_DIR = (
    BASE_PATH
    / "results"
    / "hard_example_object_sampling"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# High-confidence waste probability
#
# 0.90 is currently based on the threshold analysis.
#
# This is a diagnostic threshold for finding locations where the model is
# highly confident that waste is present.
# -----------------------------------------------------------------------------

HIGH_PROBABILITY_THRESHOLD = 0.90


# -----------------------------------------------------------------------------
# Minimum candidate object area
# -----------------------------------------------------------------------------

MIN_OBJECT_AREA_M2 = 0.25


# -----------------------------------------------------------------------------
# Sample-size settings
#
# We are estimating the proportion of high-confidence predictions that are
# genuine waste versus false-positive / hard-negative cases.
#
# p = 0.50 gives the largest sample size when the actual proportion is unknown.
# -----------------------------------------------------------------------------

CONFIDENCE_LEVEL = 0.95

MARGIN_OF_ERROR = 0.15

EXPECTED_PROPORTION = 0.50


# -----------------------------------------------------------------------------
# Reproducible random sampling
# -----------------------------------------------------------------------------

RANDOM_SEED = 20260905


# -----------------------------------------------------------------------------
# Model input size
#
# The U-Net was trained on 256 x 256 image tiles.
#
# We therefore provide half a model tile of contextual imagery around each
# selected predicted object:
#
#       256 / 2 = 128 pixels on each side
#
# This replaces an arbitrary 100 m x 100 m inspection window.
# -----------------------------------------------------------------------------

MODEL_INPUT_SIZE = 256

CONTEXT_PIXELS = (
    MODEL_INPUT_SIZE // 2
)


# -----------------------------------------------------------------------------
# Raster processing strip height
#
# Keeps memory requirements manageable for a large probability raster.
# -----------------------------------------------------------------------------

STRIP_ROWS = 512


# -----------------------------------------------------------------------------
# 8-connected predicted objects
# -----------------------------------------------------------------------------

STRUCTURE = np.ones(
    (3, 3),
    dtype=np.uint8
)


# =============================================================================
# 3. CHECK FILES
# =============================================================================

print("\n" + "=" * 78)
print("CHECKING INPUT FILES")
print("=" * 78)


if not PROBABILITY_PATH.exists():

    raise FileNotFoundError(
        "\nProbability raster not found:\n"
        f"{PROBABILITY_PATH}"
    )


if not TEST_AREA_PATH.exists():

    raise FileNotFoundError(
        "\nTest-area polygon not found:\n"
        f"{TEST_AREA_PATH}"
    )


if not EXCLUSION_AOI_PATH.exists():

    raise FileNotFoundError(
        "\nAdditional exclusion AOI not found:\n"
        f"{EXCLUSION_AOI_PATH}"
    )


print(
    f"Probability raster:\n"
    f"  {PROBABILITY_PATH}"
)

print(
    f"\nTest area:\n"
    f"  {TEST_AREA_PATH}"
)

print(
    f"\nAdditional exclusion AOI:\n"
    f"  {EXCLUSION_AOI_PATH}"
)


# =============================================================================
# 4. UNION-FIND FUNCTIONS
#
# Predicted objects can cross raster-processing strip boundaries.
#
# Union-find makes sure that such objects are treated as ONE contiguous
# object rather than as separate objects.
# =============================================================================

parent = {}


def make_set(x):

    parent[x] = x


def find(x):

    root = x

    while parent[root] != root:
        root = parent[root]

    while parent[x] != x:

        nxt = parent[x]

        parent[x] = root

        x = nxt

    return root


def union(a, b):

    ra = find(a)
    rb = find(b)

    if ra == rb:
        return

    if ra < rb:

        parent[rb] = ra

    else:

        parent[ra] = rb


# =============================================================================
# 5. RASTER INFORMATION
# =============================================================================

print("\n" + "=" * 78)
print("READING PROBABILITY RASTER")
print("=" * 78)


with rasterio.open(
    PROBABILITY_PATH
) as src:

    raster_crs = src.crs

    if raster_crs is None:

        raise ValueError(
            "Probability raster does not have a CRS."
        )

    if raster_crs.is_geographic:

        raise ValueError(
            "Probability raster uses a geographic CRS. "
            "A projected metre-based CRS is required for "
            "area calculations."
        )

    transform = src.transform

    raster_width = src.width
    raster_height = src.height

    nodata = src.nodata


pixel_width_m = abs(
    transform.a
)

pixel_height_m = abs(
    transform.e
)

pixel_area_m2 = (
    pixel_width_m
    * pixel_height_m
)


print(
    f"Raster dimensions: "
    f"{raster_width:,} x "
    f"{raster_height:,} pixels"
)

print(
    f"Pixel dimensions: "
    f"{pixel_width_m:.4f} x "
    f"{pixel_height_m:.4f} m"
)

print(
    f"Pixel area: "
    f"{pixel_area_m2:.6f} m²"
)


# =============================================================================
# 6. LOAD TEST AREA
# =============================================================================

print("\n" + "=" * 78)
print("LOADING AREAS TO EXCLUDE")
print("=" * 78)


test_area = gpd.read_file(
    TEST_AREA_PATH
)

test_area = test_area[
    test_area.geometry.notna()
    & (~test_area.geometry.is_empty)
].copy()

if test_area.empty:

    raise ValueError(
        "testarea.shp contains no valid geometries."
    )


test_area = test_area.to_crs(
    raster_crs
)


if hasattr(
    test_area.geometry,
    "union_all"
):

    test_geom = (
        test_area.geometry.union_all()
    )

else:

    test_geom = unary_union(
        list(
            test_area.geometry
        )
    )


print(
    "Test area loaded and will be excluded."
)


# =============================================================================
# 7. LOAD ADDITIONAL EXCLUSION AOI
# =============================================================================

excluded_aoi = gpd.read_file(
    EXCLUSION_AOI_PATH
)

excluded_aoi = excluded_aoi[
    excluded_aoi.geometry.notna()
    & (~excluded_aoi.geometry.is_empty)
].copy()


if excluded_aoi.empty:

    raise ValueError(
        "aoi.geojson contains no valid geometries."
    )


excluded_aoi = excluded_aoi.to_crs(
    raster_crs
)


if hasattr(
    excluded_aoi.geometry,
    "union_all"
):

    excluded_aoi_geom = (
        excluded_aoi.geometry.union_all()
    )

else:

    excluded_aoi_geom = unary_union(
        list(
            excluded_aoi.geometry
        )
    )


print(
    "Additional AOI loaded and will be excluded."
)


# =============================================================================
# 8. COMBINE EXCLUSIONS
# =============================================================================

exclusion_geom = unary_union(
    [
        test_geom,
        excluded_aoi_geom
    ]
)


print(
    "\nSampling will therefore occur ONLY outside:"
)

print(
    "  1. data/aoi/testarea.shp"
)

print(
    "  2. waste_spatio_temporal/inputs/aoi/aoi.geojson"
)


# =============================================================================
# 9. MODEL-DERIVED CONTEXT
# =============================================================================

context_width_m = (
    CONTEXT_PIXELS
    * pixel_width_m
)

context_height_m = (
    CONTEXT_PIXELS
    * pixel_height_m
)


print("\n" + "=" * 78)
print("INSPECTION WINDOW DEFINITION")
print("=" * 78)


print(
    f"Model input size: "
    f"{MODEL_INPUT_SIZE} x "
    f"{MODEL_INPUT_SIZE} pixels"
)

print(
    f"Context around each selected object: "
    f"{CONTEXT_PIXELS} pixels per side"
)

print(
    f"This corresponds to approximately "
    f"{context_width_m:.2f} m horizontally "
    f"and {context_height_m:.2f} m vertically."
)


# =============================================================================
# 10. FIND HIGH-PROBABILITY OBJECTS
# =============================================================================

print("\n" + "=" * 78)
print("IDENTIFYING HIGH-PROBABILITY OBJECTS")
print("=" * 78)


print(
    f"Probability threshold: "
    f"{HIGH_PROBABILITY_THRESHOLD:.2f}"
)

print(
    f"Minimum object area: "
    f"{MIN_OBJECT_AREA_M2:.2f} m²"
)


component_stats = []

next_component_id = 0

previous_last_row = None


with rasterio.open(
    PROBABILITY_PATH
) as src:

    for row_start in range(
        0,
        raster_height,
        STRIP_ROWS
    ):

        height = min(
            STRIP_ROWS,
            raster_height - row_start
        )


        window = Window(
            0,
            row_start,
            raster_width,
            height
        )


        probability = src.read(
            1,
            window=window
        ).astype(
            np.float32
        )


        local_transform = (
            src.window_transform(
                window
            )
        )


        # ---------------------------------------------------------------------
        # Valid pixels
        # ---------------------------------------------------------------------

        valid = np.isfinite(
            probability
        )


        if nodata is not None:

            valid &= (
                probability
                != nodata
            )


        # ---------------------------------------------------------------------
        # High-confidence pixels
        # ---------------------------------------------------------------------

        high = (
            valid
            & (
                probability
                >= HIGH_PROBABILITY_THRESHOLD
            )
        )


        # ---------------------------------------------------------------------
        # Exclude test area and additional AOI
        # ---------------------------------------------------------------------

        exclusion_mask = geometry_mask(
            [
                mapping(
                    exclusion_geom
                )
            ],
            out_shape=high.shape,
            transform=local_transform,
            invert=True
        )


        high[
            exclusion_mask
        ] = False


        # ---------------------------------------------------------------------
        # Connected components
        # ---------------------------------------------------------------------

        labels, n_local = (
            ndimage.label(
                high,
                structure=STRUCTURE
            )
        )


        if n_local == 0:

            previous_last_row = np.zeros(
                raster_width,
                dtype=np.int64
            )

            print(
                f"\rRows processed: "
                f"{min(row_start + height, raster_height):,}"
                f" / {raster_height:,}",
                end="",
                flush=True
            )

            continue


        # ---------------------------------------------------------------------
        # Assign provisional global IDs
        # ---------------------------------------------------------------------

        base_id = next_component_id


        for local_id in range(
            1,
            n_local + 1
        ):

            global_id = (
                base_id
                + local_id
            )

            make_set(
                global_id
            )


        next_component_id += (
            n_local
        )


        # ---------------------------------------------------------------------
        # Connect components across strip boundaries
        # ---------------------------------------------------------------------

        first_local = (
            labels[
                0,
                :
            ]
        )


        first_global = np.where(
            first_local > 0,
            first_local + base_id,
            0
        ).astype(
            np.int64
        )


        if previous_last_row is not None:

            # Same-column neighbours
            m = (
                (first_global > 0)
                & (
                    previous_last_row
                    > 0
                )
            )

            pairs = np.column_stack(
                (
                    first_global[m],
                    previous_last_row[m]
                )
            )


            if len(
                pairs
            ) > 0:

                for a, b in np.unique(
                    pairs,
                    axis=0
                ):

                    union(
                        int(a),
                        int(b)
                    )


            # Upper-left diagonal
            m = (
                (
                    first_global[
                        1:
                    ]
                    > 0
                )
                & (
                    previous_last_row[
                        :-1
                    ]
                    > 0
                )
            )


            pairs = np.column_stack(
                (
                    first_global[
                        1:
                    ][m],

                    previous_last_row[
                        :-1
                    ][m]
                )
            )


            if len(
                pairs
            ) > 0:

                for a, b in np.unique(
                    pairs,
                    axis=0
                ):

                    union(
                        int(a),
                        int(b)
                    )


            # Upper-right diagonal
            m = (
                (
                    first_global[
                        :-1
                    ]
                    > 0
                )
                & (
                    previous_last_row[
                        1:
                    ]
                    > 0
                )
            )


            pairs = np.column_stack(
                (
                    first_global[
                        :-1
                    ][m],

                    previous_last_row[
                        1:
                    ][m]
                )
            )


            if len(
                pairs
            ) > 0:

                for a, b in np.unique(
                    pairs,
                    axis=0
                ):

                    union(
                        int(a),
                        int(b)
                    )


        # ---------------------------------------------------------------------
        # Component statistics
        # ---------------------------------------------------------------------

        flat_labels = (
            labels.ravel()
        )


        counts = np.bincount(
            flat_labels,
            minlength=n_local + 1
        )


        probability_sums = np.bincount(
            flat_labels,
            weights=np.where(
                labels > 0,
                probability,
                0.0
            ).ravel(),
            minlength=n_local + 1
        )


        maximum_probabilities = (
            ndimage.maximum(
                probability,
                labels=labels,
                index=np.arange(
                    1,
                    n_local + 1
                )
            )
        )


        object_slices = (
            ndimage.find_objects(
                labels
            )
        )


        # One real pixel from each component
        unique_labels, first_indices = (
            np.unique(
                flat_labels,
                return_index=True
            )
        )


        first_lookup = dict(
            zip(
                unique_labels.tolist(),
                first_indices.tolist()
            )
        )


        for local_id in range(
            1,
            n_local + 1
        ):

            object_slice = (
                object_slices[
                    local_id - 1
                ]
            )


            if object_slice is None:
                continue


            rs, cs = (
                object_slice
            )


            global_id = (
                base_id
                + local_id
            )


            first_flat = (
                first_lookup[
                    local_id
                ]
            )


            seed_local_row = (
                first_flat
                // raster_width
            )


            seed_col = (
                first_flat
                % raster_width
            )


            seed_global_row = (
                row_start
                + seed_local_row
            )


            component_stats.append(
                {
                    "provisional_id":
                        global_id,

                    "pixel_count":
                        int(
                            counts[
                                local_id
                            ]
                        ),

                    "probability_sum":
                        float(
                            probability_sums[
                                local_id
                            ]
                        ),

                    "max_probability":
                        float(
                            maximum_probabilities[
                                local_id - 1
                            ]
                        ),

                    "row_min":
                        int(
                            row_start
                            + rs.start
                        ),

                    "row_max":
                        int(
                            row_start
                            + rs.stop
                            - 1
                        ),

                    "col_min":
                        int(
                            cs.start
                        ),

                    "col_max":
                        int(
                            cs.stop
                            - 1
                        ),

                    "seed_row":
                        int(
                            seed_global_row
                        ),

                    "seed_col":
                        int(
                            seed_col
                        )
                }
            )


        # ---------------------------------------------------------------------
        # Last row for connection with next strip
        # ---------------------------------------------------------------------

        last_local = (
            labels[
                -1,
                :
            ]
        )


        previous_last_row = (
            np.where(
                last_local > 0,
                last_local + base_id,
                0
            )
            .astype(
                np.int64
            )
        )


        print(
            f"\rRows processed: "
            f"{min(row_start + height, raster_height):,}"
            f" / {raster_height:,}",
            end="",
            flush=True
        )


        del probability
        del valid
        del high
        del labels
        del exclusion_mask


print("\n")


# =============================================================================
# 11. MERGE OBJECTS CROSSING STRIPS
# =============================================================================

print(
    "Merging components that cross "
    "processing-strip boundaries..."
)


merged = {}


for record in component_stats:

    root = find(
        record[
            "provisional_id"
        ]
    )


    if root not in merged:

        merged[
            root
        ] = {

            "pixel_count":
                0,

            "probability_sum":
                0.0,

            "max_probability":
                0.0,

            "row_min":
                record[
                    "row_min"
                ],

            "row_max":
                record[
                    "row_max"
                ],

            "col_min":
                record[
                    "col_min"
                ],

            "col_max":
                record[
                    "col_max"
                ],

            "seed_row":
                record[
                    "seed_row"
                ],

            "seed_col":
                record[
                    "seed_col"
                ]
        }


    target = merged[
        root
    ]


    target[
        "pixel_count"
    ] += record[
        "pixel_count"
    ]


    target[
        "probability_sum"
    ] += record[
        "probability_sum"
    ]


    target[
        "max_probability"
    ] = max(
        target[
            "max_probability"
        ],
        record[
            "max_probability"
        ]
    )


    target[
        "row_min"
    ] = min(
        target[
            "row_min"
        ],
        record[
            "row_min"
        ]
    )


    target[
        "row_max"
    ] = max(
        target[
            "row_max"
        ],
        record[
            "row_max"
        ]
    )


    target[
        "col_min"
    ] = min(
        target[
            "col_min"
        ],
        record[
            "col_min"
        ]
    )


    target[
        "col_max"
    ] = max(
        target[
            "col_max"
        ],
        record[
            "col_max"
        ]
    )


del component_stats
del parent


# =============================================================================
# 12. CREATE CANDIDATE POPULATION
# =============================================================================

print(
    "Constructing candidate-object population..."
)


candidate_records = []

inspection_geometry_lookup = {}


candidate_id = 0


for _, record in merged.items():

    pixel_count = (
        record[
            "pixel_count"
        ]
    )


    object_area_m2 = (
        pixel_count
        * pixel_area_m2
    )


    if (
        object_area_m2
        < MIN_OBJECT_AREA_M2
    ):

        continue


    mean_probability = (
        record[
            "probability_sum"
        ]
        / pixel_count
    )


    # -------------------------------------------------------------------------
    # Object bounding extent
    # -------------------------------------------------------------------------

    object_window = Window(
        record[
            "col_min"
        ],
        record[
            "row_min"
        ],
        (
            record[
                "col_max"
            ]
            - record[
                "col_min"
            ]
            + 1
        ),
        (
            record[
                "row_max"
            ]
            - record[
                "row_min"
            ]
            + 1
        )
    )


    left, bottom, right, top = (
        rasterio.windows.bounds(
            object_window,
            transform
        )
    )


    object_extent = box(
        left,
        bottom,
        right,
        top
    )


    # -------------------------------------------------------------------------
    # Variable inspection area
    #
    # Object extent + 128 pixels of context on each side.
    # -------------------------------------------------------------------------

    inspection_row_min = max(
        0,
        record[
            "row_min"
        ]
        - CONTEXT_PIXELS
    )


    inspection_row_max = min(
        raster_height - 1,
        record[
            "row_max"
        ]
        + CONTEXT_PIXELS
    )


    inspection_col_min = max(
        0,
        record[
            "col_min"
        ]
        - CONTEXT_PIXELS
    )


    inspection_col_max = min(
        raster_width - 1,
        record[
            "col_max"
        ]
        + CONTEXT_PIXELS
    )


    inspection_window = Window(
        inspection_col_min,
        inspection_row_min,
        (
            inspection_col_max
            - inspection_col_min
            + 1
        ),
        (
            inspection_row_max
            - inspection_row_min
            + 1
        )
    )


    i_left, i_bottom, i_right, i_top = (
        rasterio.windows.bounds(
            inspection_window,
            transform
        )
    )


    inspection_geom = box(
        i_left,
        i_bottom,
        i_right,
        i_top
    )


    # -------------------------------------------------------------------------
    # Do not select objects whose inspection window would enter either
    # excluded area.
    # -------------------------------------------------------------------------

    if inspection_geom.intersects(
        exclusion_geom
    ):

        continue


    candidate_id += 1


    inspection_area_m2 = (
        inspection_geom.area
    )


    candidate_records.append(
        {
            "object_id":
                candidate_id,

            "pixel_count":
                pixel_count,

            "object_area_m2":
                object_area_m2,

            "mean_probability":
                mean_probability,

            "max_probability":
                record[
                    "max_probability"
                ],

            "row_min":
                record[
                    "row_min"
                ],

            "row_max":
                record[
                    "row_max"
                ],

            "col_min":
                record[
                    "col_min"
                ],

            "col_max":
                record[
                    "col_max"
                ],

            "seed_row":
                record[
                    "seed_row"
                ],

            "seed_col":
                record[
                    "seed_col"
                ],

            "inspection_width_m":
                i_right - i_left,

            "inspection_height_m":
                i_top - i_bottom,

            "inspection_area_m2":
                inspection_area_m2,

            "geometry":
                object_extent
        }
    )


    inspection_geometry_lookup[
        candidate_id
    ] = inspection_geom


if len(
    candidate_records
) == 0:

    raise RuntimeError(
        "No candidate objects remained after "
        "thresholding and applying exclusions."
    )


candidates = gpd.GeoDataFrame(
    candidate_records,
    geometry="geometry",
    crs=raster_crs
)


N = len(
    candidates
)


# =============================================================================
# 13. SAMPLE-SIZE CALCULATION
# =============================================================================

print("\n" + "=" * 78)
print("CANDIDATE POPULATION")
print("=" * 78)


print(
    f"Candidate high-probability objects: "
    f"{N:,}"
)


alpha = (
    1.0
    - CONFIDENCE_LEVEL
)


z = norm.ppf(
    1.0
    - alpha / 2.0
)


p = EXPECTED_PROPORTION

d = MARGIN_OF_ERROR


n0 = (
    z ** 2
    * p
    * (
        1.0 - p
    )
    / d ** 2
)


n_corrected = (
    N
    * n0
    / (
        N
        + n0
        - 1
    )
)


n_required = min(
    N,
    int(
        math.ceil(
            n_corrected
        )
    )
)


print("\n" + "=" * 78)
print("SAMPLE-SIZE CALCULATION")
print("=" * 78)


print(
    f"Population N: "
    f"{N}"
)


print(
    f"Confidence level: "
    f"{CONFIDENCE_LEVEL:.0%}"
)


print(
    f"Margin of error: "
    f"±{MARGIN_OF_ERROR:.0%}"
)


print(
    f"Conservative assumed proportion: "
    f"{EXPECTED_PROPORTION:.2f}"
)


print(
    f"Initial sample size n0: "
    f"{n0:.2f}"
)


print(
    f"Finite-population corrected sample: "
    f"{n_required}"
)


# -----------------------------------------------------------------------------
# Also report alternative precision levels.
#
# This helps you decide whether ±15% is an acceptable workload.
# -----------------------------------------------------------------------------

print(
    "\nSample-size sensitivity:"
)


for alternative_d in [
    0.10,
    0.15,
    0.20
]:

    alternative_n0 = (
        z ** 2
        * p
        * (
            1 - p
        )
        / alternative_d ** 2
    )


    alternative_n = (
        N
        * alternative_n0
        / (
            N
            + alternative_n0
            - 1
        )
    )


    alternative_n = min(
        N,
        int(
            math.ceil(
                alternative_n
            )
        )
    )


    print(
        f"  ±{alternative_d:.0%}: "
        f"{alternative_n} objects"
    )


# =============================================================================
# 14. RANDOM SAMPLE
# =============================================================================

selected = (
    candidates.sample(
        n=n_required,
        replace=False,
        random_state=RANDOM_SEED
    )
    .copy()
    .reset_index(
        drop=True
    )
)


selected[
    "sample_id"
] = np.arange(
    1,
    len(selected) + 1
)


# =============================================================================
# 15. BUILD INSPECTION AREAS
# =============================================================================

inspection_records = []


for _, row in selected.iterrows():

    object_id = int(
        row[
            "object_id"
        ]
    )


    geom = (
        inspection_geometry_lookup[
            object_id
        ]
    )


    inspection_records.append(
        {
            "sample_id":
                int(
                    row[
                        "sample_id"
                    ]
                ),

            "object_id":
                object_id,

            "object_area_m2":
                float(
                    row[
                        "object_area_m2"
                    ]
                ),

            "mean_prob":
                float(
                    row[
                        "mean_probability"
                    ]
                ),

            "max_prob":
                float(
                    row[
                        "max_probability"
                    ]
                ),

            "width_m":
                float(
                    row[
                        "inspection_width_m"
                    ]
                ),

            "height_m":
                float(
                    row[
                        "inspection_height_m"
                    ]
                ),

            "inspection_area_m2":
                float(
                    row[
                        "inspection_area_m2"
                    ]
                ),

            "geometry":
                geom
        }
    )


inspection_areas = gpd.GeoDataFrame(
    inspection_records,
    geometry="geometry",
    crs=raster_crs
)


# =============================================================================
# 16. CREATE NAVIGATION POINTS
# =============================================================================

point_records = []


for _, row in selected.iterrows():

    point_records.append(
        {
            "sample_id":
                int(
                    row[
                        "sample_id"
                    ]
                ),

            "object_id":
                int(
                    row[
                        "object_id"
                    ]
                ),

            "object_area_m2":
                float(
                    row[
                        "object_area_m2"
                    ]
                ),

            "mean_prob":
                float(
                    row[
                        "mean_probability"
                    ]
                ),

            "max_prob":
                float(
                    row[
                        "max_probability"
                    ]
                ),

            "geometry":
                row.geometry.centroid
        }
    )


sample_points = gpd.GeoDataFrame(
    point_records,
    geometry="geometry",
    crs=raster_crs
)


# =============================================================================
# 17. RECONSTRUCT ACTUAL GEOMETRIES OF SELECTED PREDICTED OBJECTS
# =============================================================================

print(
    "\nReconstructing exact geometries "
    "of selected candidate objects..."
)


selected_object_records = []


with rasterio.open(
    PROBABILITY_PATH
) as src:

    for _, record in selected.iterrows():

        sample_id = int(
            record[
                "sample_id"
            ]
        )


        object_id = int(
            record[
                "object_id"
            ]
        )


        row_min = max(
            0,
            int(
                record[
                    "row_min"
                ]
            ) - 1
        )


        row_max = min(
            raster_height - 1,
            int(
                record[
                    "row_max"
                ]
            ) + 1
        )


        col_min = max(
            0,
            int(
                record[
                    "col_min"
                ]
            ) - 1
        )


        col_max = min(
            raster_width - 1,
            int(
                record[
                    "col_max"
                ]
            ) + 1
        )


        local_window = Window(
            col_min,
            row_min,
            col_max - col_min + 1,
            row_max - row_min + 1
        )


        probability = src.read(
            1,
            window=local_window
        ).astype(
            np.float32
        )


        local_transform = (
            src.window_transform(
                local_window
            )
        )


        valid = np.isfinite(
            probability
        )


        if src.nodata is not None:

            valid &= (
                probability
                != src.nodata
            )


        high = (
            valid
            & (
                probability
                >= HIGH_PROBABILITY_THRESHOLD
            )
        )


        # Exclusions again, for safety
        local_exclusion = geometry_mask(
            [
                mapping(
                    exclusion_geom
                )
            ],
            out_shape=high.shape,
            transform=local_transform,
            invert=True
        )


        high[
            local_exclusion
        ] = False


        local_labels, _ = ndimage.label(
            high,
            structure=STRUCTURE
        )


        seed_row_local = (
            int(
                record[
                    "seed_row"
                ]
            )
            - row_min
        )


        seed_col_local = (
            int(
                record[
                    "seed_col"
                ]
            )
            - col_min
        )


        target_label = int(
            local_labels[
                seed_row_local,
                seed_col_local
            ]
        )


        if target_label == 0:

            print(
                f"WARNING: could not reconstruct "
                f"sample {sample_id}"
            )

            continue


        target_mask = (
            local_labels
            == target_label
        )


        polygons = []


        for geom, value in shapes(
            target_mask.astype(
                np.uint8
            ),
            mask=target_mask,
            transform=local_transform,
            connectivity=8
        ):

            if int(
                value
            ) != 1:

                continue


            polygons.append(
                shape(
                    geom
                )
            )


        if len(
            polygons
        ) == 0:

            continue


        actual_geometry = unary_union(
            polygons
        )


        selected_object_records.append(
            {
                "sample_id":
                    sample_id,

                "object_id":
                    object_id,

                "object_area_m2":
                    float(
                        record[
                            "object_area_m2"
                        ]
                    ),

                "mean_prob":
                    float(
                        record[
                            "mean_probability"
                        ]
                    ),

                "max_prob":
                    float(
                        record[
                            "max_probability"
                        ]
                    ),

                "geometry":
                    actual_geometry
            }
        )


selected_objects = gpd.GeoDataFrame(
    selected_object_records,
    geometry="geometry",
    crs=raster_crs
)


# =============================================================================
# 18. MERGE OVERLAPPING INSPECTION AREAS
#
# This reduces unnecessary duplicate digitisation if two sampled objects
# happen to be close together.
# =============================================================================

merged_inspection_geom = unary_union(
    list(
        inspection_areas.geometry
    )
)


if (
    merged_inspection_geom.geom_type
    == "Polygon"
):

    merged_geometries = [
        merged_inspection_geom
    ]

else:

    merged_geometries = list(
        merged_inspection_geom.geoms
    )


cluster_records = []


for cluster_id, geom in enumerate(
    merged_geometries,
    start=1
):

    intersects = (
        inspection_areas[
            inspection_areas
            .geometry
            .intersects(
                geom
            )
        ]
    )


    sample_ids = ",".join(
        str(
            x
        )
        for x in sorted(
            intersects[
                "sample_id"
            ].tolist()
        )
    )


    cluster_records.append(
        {
            "cluster_id":
                cluster_id,

            "n_sample_objects":
                len(
                    intersects
                ),

            "sample_ids":
                sample_ids,

            "area_m2":
                geom.area,

            "geometry":
                geom
        }
    )


inspection_clusters = gpd.GeoDataFrame(
    cluster_records,
    geometry="geometry",
    crs=raster_crs
)


# =============================================================================
# 19. SAVE GEOSPATIAL OUTPUTS
# =============================================================================

print(
    "\nSaving outputs..."
)


# -----------------------------------------------------------------------------
# All eligible high-confidence objects
#
# Geometry here is the object bounding extent, which is sufficient for seeing
# the complete candidate population efficiently.
# -----------------------------------------------------------------------------

candidates.to_file(
    OUTPUT_DIR
    / "candidate_object_extents.gpkg",
    layer="candidate_objects",
    driver="GPKG"
)


# -----------------------------------------------------------------------------
# Actual thresholded polygons for sampled objects
# -----------------------------------------------------------------------------

selected_objects.to_file(
    OUTPUT_DIR
    / "selected_candidate_objects.gpkg",
    layer="selected_objects",
    driver="GPKG"
)


# -----------------------------------------------------------------------------
# Individual variable inspection areas
# -----------------------------------------------------------------------------

inspection_areas.to_file(
    OUTPUT_DIR
    / "selected_inspection_areas.gpkg",
    layer="inspection_areas",
    driver="GPKG"
)


# -----------------------------------------------------------------------------
# Merged inspection areas: use these if you want to avoid digitising
# overlapping windows more than once.
# -----------------------------------------------------------------------------

inspection_clusters.to_file(
    OUTPUT_DIR
    / "merged_inspection_areas.gpkg",
    layer="inspection_clusters",
    driver="GPKG"
)


# -----------------------------------------------------------------------------
# Navigation points
# -----------------------------------------------------------------------------

sample_points.to_file(
    OUTPUT_DIR
    / "selected_candidate_points.gpkg",
    layer="sample_points",
    driver="GPKG"
)


# -----------------------------------------------------------------------------
# Exclusion layers
# -----------------------------------------------------------------------------

exclusion_output = gpd.GeoDataFrame(
    {
        "area_type": [
            "test_area",
            "additional_exclusion"
        ]
    },
    geometry=[
        test_geom,
        excluded_aoi_geom
    ],
    crs=raster_crs
)


exclusion_output.to_file(
    OUTPUT_DIR
    / "sampling_exclusions.gpkg",
    layer="exclusions",
    driver="GPKG"
)


# =============================================================================
# 20. SAVE SAMPLING SUMMARY
# =============================================================================

summary = selected[
    [
        "sample_id",
        "object_id",
        "object_area_m2",
        "mean_probability",
        "max_probability",
        "inspection_width_m",
        "inspection_height_m",
        "inspection_area_m2"
    ]
].copy()


summary[
    "candidate_population_N"
] = N


summary[
    "sample_size_n"
] = n_required


summary[
    "confidence_level"
] = CONFIDENCE_LEVEL


summary[
    "margin_of_error"
] = MARGIN_OF_ERROR


summary[
    "assumed_proportion"
] = EXPECTED_PROPORTION


summary[
    "probability_threshold"
] = HIGH_PROBABILITY_THRESHOLD


summary[
    "minimum_object_area_m2"
] = MIN_OBJECT_AREA_M2


summary[
    "random_seed"
] = RANDOM_SEED


summary.to_csv(
    OUTPUT_DIR
    / "sampling_summary.csv",
    index=False
)


# =============================================================================
# 21. MANUAL INTERPRETATION TEMPLATE
# =============================================================================

interpretation = summary[
    [
        "sample_id",
        "object_id",
        "object_area_m2",
        "mean_probability",
        "max_probability"
    ]
].copy()


interpretation[
    "visual_class"
] = ""


interpretation[
    "waste_present"
] = ""


interpretation[
    "classification"
] = ""


interpretation[
    "digitise_for_finetuning"
] = ""


interpretation[
    "notes"
] = ""


interpretation.to_csv(
    OUTPUT_DIR
    / "manual_interpretation_template.csv",
    index=False
)


# =============================================================================
# 22. FINAL REPORT
# =============================================================================

print("\n" + "=" * 78)
print("SAMPLING COMPLETE")
print("=" * 78)


print(
    f"\nCandidate high-confidence objects: "
    f"{N:,}"
)


print(
    f"Objects randomly selected: "
    f"{n_required:,}"
)


print(
    f"Confidence level: "
    f"{CONFIDENCE_LEVEL:.0%}"
)


print(
    f"Margin of error: "
    f"±{MARGIN_OF_ERROR:.0%}"
)


print(
    f"Probability threshold: "
    f"{HIGH_PROBABILITY_THRESHOLD:.2f}"
)


print(
    "\nVariable inspection-area dimensions:"
)


print(
    f"  Minimum area: "
    f"{inspection_areas['inspection_area_m2'].min():,.1f} m²"
)


print(
    f"  Median area: "
    f"{inspection_areas['inspection_area_m2'].median():,.1f} m²"
)


print(
    f"  Maximum area: "
    f"{inspection_areas['inspection_area_m2'].max():,.1f} m²"
)


print(
    f"\nIndividual inspection windows: "
    f"{len(inspection_areas)}"
)


print(
    f"Merged digitisation areas: "
    f"{len(inspection_clusters)}"
)


print(
    "\nOutputs saved to:"
)

print(
    OUTPUT_DIR
)


print(
    "\nLoad these in QGIS:"
)

print(
    "  selected_candidate_objects.gpkg"
)

print(
    "  selected_inspection_areas.gpkg"
)

print(
    "  merged_inspection_areas.gpkg"
)

print(
    "  selected_candidate_points.gpkg"
)

print(
    "  sampling_exclusions.gpkg"
)


print(
    "\nUse this to record what each prediction actually represents:"
)

print(
    "  manual_interpretation_template.csv"
)