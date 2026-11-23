"""Paired spatial waste-map evaluation. Python 3.12.

Uses existing probability GeoTIFFs; it does not run TensorFlow or train a model.
The same test polygon, corrected reference, valid pixels, thresholds and object
matching rules are used for both models. Large rasters are processed in strips.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from array import array
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import Window, from_bounds
from scipy import ndimage as ndi
from shapely.geometry import box, mapping
from shapely.ops import unary_union

# =============================================================================
# 1. CONFIGURATION — check these paths before running
# =============================================================================
BASE = Path.home() / "Documents" / "GitHub" / "mapping-waste-urban-africa"

TEST_AREA = BASE / "data" / "aoi" / "testarea.shp"

# This is the original inference raster used for hard-example discovery.
# Do not replace it with a different model's probability raster.
ORIGINAL_PROBABILITY = BASE / "outputs" / "waste_probability_full.tif"

# Use the actual fine-tuned spatial probability file from the completed run.
REFINED_PROBABILITY = (
    BASE / "results" / "test_area_finetuned"
    / "testarea_waste_probability11.tif"
)

# Earlier object-analysis code used this path. Confirm that it is the UPDATED,
# corrected reference, including the genuine piles found during error review.
# If the corrected copy is elsewhere, change this single path.
REFERENCE_WASTE = BASE / "05_finetuning" / "step0" / "waste.shp"

OUTPUT_ROOT = BASE / "results" / "spatial_test_comparison"

# Do not assert completeness merely because a file is named waste.shp.
# Set True ONLY after confirming every waste pile within TEST_AREA has been
# checked and the corrected reference is exhaustive. Otherwise, precision and
# false-positive counts are reported as provisional/unmatched candidates.
REFERENCE_COMPLETE = True

# Optional exhaustive, georeferenced land-cover reference raster. If supplied,
# raw label 1 is waste; labels 2–10 are non-waste; 0, 99 and other values are
# ignored. This is NOT the predicted land-cover raster and NOT a training mask.
# It must cover the intended evaluation domain on the exact orthomosaic grid.
PIXEL_REFERENCE_RASTER = None

# Previous object-evaluation conventions. Do not optimise these on the test area.
THRESHOLDS = (0.50, 0.90)  # 0.50 primary; 0.90 high-confidence diagnostic
MIN_OBJECT_AREA_M2 = 0.25
MIN_REFERENCE_COVERAGE = 0.01
CONNECTIVITY = 8
STRIP_ROWS = 512

# Refuse hidden resampling. Inputs must share CRS, resolution and pixel alignment.
# A full-area raster and a smaller test-area raster may have different extents.
GRID_TOLERANCE_PIXELS = 1e-5

# If true, the script stops when either probability surface has missing pixels
# within the test polygon. This prevents silently changing the evaluation area.
REQUIRE_COMPLETE_PROBABILITY_COVERAGE = True

# Optional: write a GeoPackage of connected-component polygon fragments. The
# label raster is the authoritative object map; fragment IDs can be dissolved
# in QGIS if polygon geometries are needed. Disabled to avoid large exports.
EXPORT_POLYGON_FRAGMENTS = False

# =============================================================================
# 2. HELPERS
# =============================================================================
def require(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_info(path: Path) -> dict:
    p = Path(path)
    result = {"path": str(p.resolve()), "bytes": p.stat().st_size,
              "modified_ns": p.stat().st_mtime_ns}
    if p.suffix.lower() == ".shp":
        result["sidecar_sha256"] = {
            s: sha256_file(p.with_suffix(s))
            for s in (".shp", ".shx", ".dbf", ".prj", ".cpg")
            if p.with_suffix(s).exists()
        }
    else:
        with rasterio.open(p) as src:
            result.update(width=src.width, height=src.height,
                          crs=str(src.crs), transform=list(src.transform),
                          nodata=src.nodata, tags=src.tags())
    return result


def polygonal(g):
    if g is None or g.is_empty:
        return None
    if not g.is_valid:
        try:
            from shapely import make_valid
            g = make_valid(g)
        except ImportError:
            g = g.buffer(0)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g if not g.is_empty else None
    if hasattr(g, "geoms"):
        parts = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
        return unary_union(parts) if parts else None
    return None


def read_polygons(path, crs):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Vector CRS is missing: {path}")
    gdf = gdf.to_crs(crs)
    gdf["geometry"] = gdf.geometry.map(polygonal)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy().reset_index(drop=True)


def grid_offset(src, target_transform):
    if src.crs != GRID_CRS:
        raise ValueError(f"CRS mismatch: {src.name}. Reproject explicitly before evaluation.")
    a, b = src.transform, target_transform
    if any(abs(x-y) > 1e-10 for x, y in ((a.a,b.a),(a.b,b.b),(a.d,b.d),(a.e,b.e))):
        raise ValueError(f"Resolution or rotation mismatch: {src.name}. No automatic resampling is permitted.")
    col, row = (~a) * (b.c, b.f)
    if max(abs(col-round(col)), abs(row-round(row))) > GRID_TOLERANCE_PIXELS:
        raise ValueError(f"Pixel grids are not aligned: {src.name}. No automatic resampling is permitted.")
    return int(round(row)), int(round(col))


def checked_read(src, window, dtype=None):
    if (window.row_off < 0 or window.col_off < 0 or
        window.row_off + window.height > src.height or
        window.col_off + window.width > src.width):
        raise ValueError(f"Raster does not cover the evaluation window: {src.name}")
    data = src.read(1, window=window, masked=True)
    valid = ~np.ma.getmaskarray(data)
    values = np.asarray(data.data, dtype=dtype) if dtype else np.asarray(data.data)
    return values, valid


def profile(dtype, nodata, transform, height, width):
    return dict(driver="GTiff", height=height, width=width, count=1,
                dtype=dtype, crs=GRID_CRS, transform=transform, nodata=nodata,
                tiled=True, blockxsize=256, blockysize=256,
                compress="deflate", BIGTIFF="IF_SAFER")


def strips(height, width):
    for row in range(0, height, STRIP_ROWS):
        yield Window(0, row, width, min(STRIP_ROWS, height-row))


def read_on_grid(src, offsets, window, dtype=None):
    return checked_read(src, Window(window.col_off+offsets[1],
                                    window.row_off+offsets[0],
                                    window.width, window.height), dtype)


def safe_ratio(a, b):
    return float(a/b) if b else float("nan")


def scores(tp, fp, fn, tn=None):
    p = safe_ratio(tp, tp+fp)
    r = safe_ratio(tp, tp+fn)
    f = safe_ratio(2*tp, 2*tp+fp+fn)
    j = safe_ratio(tp, tp+fp+fn)
    return dict(precision=p, recall=r, f1=f, iou=j,
                accuracy=safe_ratio(tp+tn, tp+fp+fn+tn) if tn is not None else float("nan"))


def write_csv(path, records, columns=None):
    pd.DataFrame(records, columns=columns).to_csv(path, index=False)


class UnionFind:
    """Compact union-find for components crossing strip boundaries."""
    def __init__(self):
        self.parent = array("I", [0])
        self.size = array("Q", [0])

    def add(self, counts):
        start = len(self.parent)
        for i, n in enumerate(counts):
            self.parent.append(start+i)
            self.size.append(int(n))
        return start

    def find(self, x):
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != x:
            nxt = p[x]
            p[x] = root
            x = nxt
        return root

    def union(self, a, b):
        a, b = self.find(int(a)), self.find(int(b))
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        self.size[b] = 0


def merge_seam(uf, upper, lower):
    # Same-column and diagonal contact constitutes 8-connectivity.
    for shift in (-1, 0, 1):
        if shift == -1:
            a, b = upper[1:], lower[:-1]
        elif shift == 1:
            a, b = upper[:-1], lower[1:]
        else:
            a, b = upper, lower
        keep = (a != 0) & (b != 0) & (a != b)
        if not np.any(keep):
            continue
        pairs = np.unique(np.column_stack((a[keep], b[keep])), axis=0)
        for x, y in pairs:
            uf.union(x, y)


# =============================================================================
# 3. PREPARE THE SHARED SPATIAL DOMAIN AND CORRECTED REFERENCE
# =============================================================================
def prepare_domain(original, refined, reference_path, out):
    global GRID_CRS
    GRID_CRS = refined.crs
    if GRID_CRS is None or GRID_CRS.is_geographic:
        raise ValueError("Use a projected, metre-based raster CRS.")
    if refined.count != 1 or original.count != 1:
        raise ValueError("Both probability rasters must contain exactly one band.")
    from pyproj import CRS as PyprojCRS
    if not all(ax.unit_name.lower() in ("metre", "meter", "metres", "meters")
               for ax in PyprojCRS.from_user_input(GRID_CRS).axis_info[:2]):
        raise ValueError("Raster horizontal CRS units must be metres.")
    aoi = read_polygons(require(TEST_AREA, "Test-area polygon"), GRID_CRS)
    test_geom = polygonal(unary_union(aoi.geometry.tolist()))
    if test_geom is None:
        raise ValueError("Empty test-area polygon.")
    raw = from_bounds(*test_geom.bounds, transform=refined.transform)
    r0, c0 = math.floor(raw.row_off), math.floor(raw.col_off)
    r1 = math.ceil(raw.row_off + raw.height)
    c1 = math.ceil(raw.col_off + raw.width)
    if r0 < 0 or c0 < 0 or r1 > refined.height or c1 > refined.width:
        raise ValueError("Refined probability raster does not cover the complete test-area bounds.")
    height, width = r1-r0, c1-c0
    transform = refined.window_transform(Window(c0, r0, width, height))
    old_offset = grid_offset(original, transform)
    new_offset = grid_offset(refined, transform)
    # Check coverage before creating any large scratch files.
    for src, offset in ((original,old_offset),(refined,new_offset)):
        if offset[0] < 0 or offset[1] < 0 or offset[0]+height > src.height or offset[1]+width > src.width:
            raise ValueError(f"{src.name} does not cover the entire test-area bounding window.")
    cell_area = abs(transform.a*transform.e-transform.b*transform.d)
    if cell_area <= 0:
        raise ValueError("Invalid raster pixel area.")
    valid_path = out / "evaluation_domain.tif"
    ref_path = out / "reference_ids.tif"
    ref_gdf = read_polygons(reference_path, GRID_CRS)
    if ref_gdf.empty:
        raise ValueError("Reference contains no polygon features.")
    # Preserve each source feature as a separate pile; clip to the study domain.
    ref_gdf["source_row"] = np.arange(1, len(ref_gdf)+1)
    ref_gdf["geometry"] = ref_gdf.geometry.map(lambda g: polygonal(g.intersection(test_geom)))
    ref_gdf = ref_gdf[ref_gdf.geometry.notna() & ~ref_gdf.geometry.is_empty].copy().reset_index(drop=True)
    if ref_gdf.empty:
        raise ValueError("No reference piles intersect the test area.")
    ref_gdf["pile_id"] = np.arange(1, len(ref_gdf)+1)
    ref_gdf["reference_area_m2"] = ref_gdf.geometry.area
    # Distinct reference objects may share boundaries, but should not overlap in area.
    sidx = ref_gdf.sindex
    for i, g in enumerate(ref_gdf.geometry):
        for j in sidx.query(g, predicate="intersects"):
            if j <= i:
                continue
            if g.intersection(ref_gdf.geometry.iloc[j]).area > cell_area*1e-4:
                raise ValueError(f"Reference polygons {i+1} and {j+1} overlap in area. Resolve duplicate/overlapping pile identities first.")
    # Snapshot the exact source-derived reference used in this run.
    ref_gdf.to_file(out / "reference_snapshot.gpkg", layer="reference_piles", driver="GPKG")
    reference_counts = np.zeros(len(ref_gdf)+1, dtype=np.int64)
    coverage_rows = []
    with rasterio.open(valid_path,"w",**profile("uint8",0,transform,height,width)) as valid_dst, \
         rasterio.open(ref_path,"w",**profile("uint32",0,transform,height,width)) as ref_dst:
        for w in strips(height,width):
            tr = rasterio.windows.transform(w,transform)
            inside = geometry_mask([mapping(test_geom)],out_shape=(int(w.height),int(w.width)),transform=tr,invert=True,all_touched=False)
            old, ov = read_on_grid(original,old_offset,w,"float32")
            new, nv = read_on_grid(refined,new_offset,w,"float32")
            old_ok = ov & np.isfinite(old) & (old >= 0) & (old <= 1)
            new_ok = nv & np.isfinite(new) & (new >= 0) & (new <= 1)
            common = inside & old_ok & new_ok
            coverage_rows.append(dict(inside=int(inside.sum()),valid=int(common.sum()),
                                      original_missing=int((inside & ~old_ok).sum()),
                                      refined_missing=int((inside & ~new_ok).sum())))
            valid_dst.write(common.astype("uint8"),1,window=w)
            indices = sidx.query(box(*rasterio.windows.bounds(w,transform)),predicate="intersects")
            shapes = [(mapping(ref_gdf.geometry.iloc[int(i)]),int(ref_gdf.pile_id.iloc[int(i)])) for i in indices]
            ids = rasterize(shapes,out_shape=inside.shape,transform=tr,fill=0,dtype="uint32",all_touched=False)
            ids[~common] = 0
            ref_dst.write(ids,1,window=w)
            reference_counts += np.bincount(ids.ravel(),minlength=len(reference_counts))
    coverage = pd.DataFrame(coverage_rows).sum().to_dict()
    if coverage["valid"] == 0:
        raise ValueError("No common valid probability pixels in the test area.")
    missing = coverage["inside"]-coverage["valid"]
    if missing and REQUIRE_COMPLETE_PROBABILITY_COVERAGE:
        raise ValueError(f"The two probability rasters have {missing:,} missing/invalid test-area pixels. No full-area accuracy is reported. Inspect evaluation_domain.tif and the source raster footprints. Set REQUIRE_COMPLETE_PROBABILITY_COVERAGE=False only if a restricted common-coverage analysis is intended.")
    missing_refs = np.flatnonzero(reference_counts[1:]==0)+1
    if len(missing_refs):
        raise ValueError(f"Reference piles have no valid raster-centre pixels: {missing_refs.tolist()}. Check their geometry, coverage and resolution before evaluation.")
    ref_gdf["reference_pixels"] = reference_counts[1:]
    ref_gdf["raster_reference_area_m2"] = reference_counts[1:]*cell_area
    ref_gdf.drop(columns="geometry").to_csv(out/"reference_inventory.csv",index=False)
    (out/"coverage.json").write_text(json.dumps(coverage,indent=2),encoding="utf-8")
    return dict(transform=transform,height=height,width=width,old_offset=old_offset,
                new_offset=new_offset,cell_area=cell_area,valid_path=valid_path,
                ref_path=ref_path,ref_gdf=ref_gdf,coverage=coverage,
                reference_counts=reference_counts)


# =============================================================================
# 4. LABEL CONTIGUOUS OBJECTS WITHOUT LOADING THE WHOLE RASTER
# =============================================================================
def label_model(name, src, offset, threshold, domain, out):
    height,width = domain["height"],domain["width"]
    transform = domain["transform"]
    area = domain["cell_area"]
    minimum_cells = max(1,math.ceil(MIN_OBJECT_AREA_M2/area-1e-10))
    folder = out / name / f"threshold_{threshold:.2f}"
    folder.mkdir(parents=True,exist_ok=True)
    provisional = folder/"_provisional_labels.tif"
    final_path = folder/"connected_objects.tif"
    uf = UnionFind()
    previous_row = None
    structure = np.ones((3,3),dtype=np.uint8)
    print(f"\n{name}, threshold {threshold:.2f}: first connected-component pass",flush=True)
    with rasterio.open(domain["valid_path"]) as valid_src, \
         rasterio.open(provisional,"w",**profile("uint32",0,transform,height,width)) as dst:
        for w in strips(height,width):
            valid = valid_src.read(1,window=w).astype(bool)
            prob, pmask = read_on_grid(src,offset,w,"float32")
            binary = valid & pmask & (prob >= threshold)
            local,n = ndi.label(binary,structure=structure)
            if n:
                counts = np.bincount(local.ravel(),minlength=n+1)[1:]
                start = uf.add(counts)
                labels = local.astype(np.uint32,copy=False)
                labels[labels>0] += start-1
            else:
                labels = local.astype(np.uint32,copy=False)
            if previous_row is not None:
                merge_seam(uf,previous_row,labels[0])
            previous_row = labels[-1].copy()
            dst.write(labels,1,window=w)
    # Resolve all cross-strip equivalences and remove tiny objects only AFTER
    # joining them. A component crossing a strip boundary is counted once.
    n_provisional = len(uf.parent)-1
    root_to_id = {}
    lookup = np.zeros(n_provisional+1,dtype=np.uint32)
    final_counts = [0]
    for i in range(1,n_provisional+1):
        root = uf.find(i)
        if uf.size[root] < minimum_cells:
            continue
        if root not in root_to_id:
            root_to_id[root] = len(final_counts)
            final_counts.append(int(uf.size[root]))
        lookup[i] = root_to_id[root]
    n_objects = len(final_counts)-1
    print(f"  {n_provisional:,} provisional components; {n_objects:,} retained objects",flush=True)
    if n_objects >= 2**32:
        raise ValueError("Too many objects for uint32 labels.")
    # Second pass: compact IDs, probability/pixel statistics, and intersections
    # with the SAME corrected reference raster for both models.
    intersections = defaultdict(int)
    sums_row = np.zeros(n_objects+1,dtype=np.float64)
    sums_col = np.zeros(n_objects+1,dtype=np.float64)
    pixel_counts = dict(tp=0,fp=0,fn=0,tn=0)
    ref_pixel_counts = domain["reference_counts"]
    with ExitStack() as stack:
        prov = stack.enter_context(rasterio.open(provisional))
        valid_src = stack.enter_context(rasterio.open(domain["valid_path"]))
        ref_src = stack.enter_context(rasterio.open(domain["ref_path"]))
        dst = stack.enter_context(rasterio.open(final_path,"w",**profile("uint32",0,transform,height,width)))
        for w in strips(height,width):
            labels = lookup[prov.read(1,window=w)]
            dst.write(labels,1,window=w)
            valid = valid_src.read(1,window=w).astype(bool)
            ref = ref_src.read(1,window=w)
            pred = labels>0
            truth = ref>0
            # Pixel counts are conditional on a complete polygon reference.
            pixel_counts["tp"] += int((pred & truth & valid).sum())
            pixel_counts["fp"] += int((pred & ~truth & valid).sum())
            pixel_counts["fn"] += int((~pred & truth & valid).sum())
            pixel_counts["tn"] += int((~pred & ~truth & valid).sum())
            rr,cc = np.nonzero(pred)
            if len(rr):
                ids = labels[rr,cc]
                sums_row += np.bincount(ids,weights=rr.astype(np.float64)+w.row_off+0.5,minlength=n_objects+1)
                sums_col += np.bincount(ids,weights=cc.astype(np.float64)+0.5,minlength=n_objects+1)
            overlap = pred & truth
            if np.any(overlap):
                codes = ref[overlap].astype(np.uint64)*(n_objects+1)+labels[overlap].astype(np.uint64)
                unique,counts = np.unique(codes,return_counts=True)
                for code,count in zip(unique,counts):
                    intersections[(int(code//(n_objects+1)),int(code%(n_objects+1)))] += int(count)
    # The reference-area denominator and IoU convention follow the earlier R
    # implementation: vector reference area, raster intersection, capped by both.
    refs = domain["ref_gdf"]
    reference_areas = np.r_[0.0,refs.reference_area_m2.to_numpy(dtype=float)]
    predicted_areas = np.asarray(final_counts,dtype=np.float64)*area
    candidates = []
    for (rid,pid),n in intersections.items():
        overlap_area = n*area
        intersection = min(overlap_area,reference_areas[rid],predicted_areas[pid])
        coverage = intersection/reference_areas[rid]
        if coverage < MIN_REFERENCE_COVERAGE:
            continue
        union_area = reference_areas[rid]+predicted_areas[pid]-intersection
        candidates.append(dict(pile_id=rid,object_id=pid,overlap_pixels=n,
            overlap_area_m2=overlap_area,intersection_area_m2=intersection,
            reference_area_m2=reference_areas[rid],predicted_area_m2=predicted_areas[pid],
            reference_coverage=coverage,predicted_coverage=intersection/predicted_areas[pid],
            iou=intersection/union_area if union_area>0 else 0.0))
    candidates.sort(key=lambda x:(-x["iou"],-x["reference_coverage"],-x["overlap_area_m2"],x["pile_id"],x["object_id"]))
    used_ref,used_pred = set(),set()
    matches = []
    for c in candidates:
        if c["pile_id"] not in used_ref and c["object_id"] not in used_pred:
            matches.append(c)
            used_ref.add(c["pile_id"])
            used_pred.add(c["object_id"])
    tp = len(matches)
    apparent_fp = n_objects-tp
    fn = len(refs)-tp
    # "Unmatched" is a factual status; "false positive" requires exhaustive
    # reference coverage. Keep both fields distinct in output tables.
    object_scores = scores(tp,apparent_fp,fn)
    if not REFERENCE_COMPLETE:
        for k in ("precision","f1","iou","accuracy"):
            object_scores[k] = float("nan")
    summary = dict(model=name,threshold=threshold,reference_complete=REFERENCE_COMPLETE,
        reference_objects=len(refs),predicted_objects=n_objects,true_positives=tp,
        unmatched_predictions=apparent_fp,
        false_positives=apparent_fp if REFERENCE_COMPLETE else float("nan"),
        false_negatives=fn,**{f"object_{k}":v for k,v in object_scores.items()},
        apparent_precision=safe_ratio(tp,n_objects),
        apparent_f1=safe_ratio(2*tp,2*tp+apparent_fp+fn),
        mapped_area_m2=float(predicted_areas.sum()),
        reference_area_m2=float(refs.reference_area_m2.sum()),
        mapped_area_error_m2=float(predicted_areas.sum()-refs.reference_area_m2.sum()),
        mapped_area_ratio=safe_ratio(predicted_areas.sum(),refs.reference_area_m2.sum()),
        evaluation_area_km2=domain["coverage"]["valid"]*area/1e6,
        predicted_objects_per_km2=safe_ratio(n_objects,domain["coverage"]["valid"]*area/1e6),
        unmatched_predictions_per_km2=safe_ratio(apparent_fp,domain["coverage"]["valid"]*area/1e6),
        false_positives_per_km2=safe_ratio(apparent_fp,domain["coverage"]["valid"]*area/1e6) if REFERENCE_COMPLETE else float("nan"),
        minimum_object_area_m2=MIN_OBJECT_AREA_M2,minimum_reference_coverage=MIN_REFERENCE_COVERAGE,
        minimum_object_pixels=minimum_cells,connectivity=CONNECTIVITY)
    # Polygon-derived binary pixel metrics: valid only if the positive reference
    # is exhaustive. Otherwise only reference-pixel recall is interpretable.
    pm = scores(pixel_counts["tp"],pixel_counts["fp"],pixel_counts["fn"],pixel_counts["tn"])
    if not REFERENCE_COMPLETE:
        for k in ("accuracy","precision","f1","iou"):
            pm[k] = float("nan")
    pixel_summary = dict(model=name,threshold=threshold,reference_complete=REFERENCE_COMPLETE,
        **pixel_counts,**pm,prediction_rule="probability_threshold_then_0.25m2_object_filter",
        reference_rule="corrected_waste_polygons; other pixels non-waste only if exhaustive")
    summary["pixel_recall"] = pm["recall"]
    summary["pixel_precision"] = pm["precision"]
    summary["pixel_f1"] = pm["f1"]
    summary["pixel_iou"] = pm["iou"]
    summary["pixel_accuracy"] = pm["accuracy"]
    summary["pixel_tp"] = pixel_counts["tp"]
    summary["pixel_fp"] = pixel_counts["fp"] if REFERENCE_COMPLETE else float("nan")
    summary["pixel_fn"] = pixel_counts["fn"]
    summary["pixel_tn"] = pixel_counts["tn"] if REFERENCE_COMPLETE else float("nan")
    # Per-pile matched status, and separate all-predicted-object inventory.
    match_by_ref = {m["pile_id"]:m for m in matches}
    match_by_pred = {m["object_id"]:m for m in matches}
    ref_records = []
    for _,r in refs.iterrows():
        rid = int(r.pile_id)
        m = match_by_ref.get(rid)
        ref_records.append(dict(pile_id=rid,source_row=int(r.source_row),
            reference_area_m2=float(r.reference_area_m2),reference_pixels=int(ref_pixel_counts[rid]),
            detected=m is not None,matched_object_id=m["object_id"] if m else pd.NA,
            reference_coverage=m["reference_coverage"] if m else 0.0,
            overlap_area_m2=m["overlap_area_m2"] if m else 0.0,
            match_iou=m["iou"] if m else 0.0))
    object_records=[]
    for pid in range(1,n_objects+1):
        m=match_by_pred.get(pid)
        col=sums_col[pid]/final_counts[pid]
        row=sums_row[pid]/final_counts[pid]
        x,y=transform*(col,row)
        object_records.append(dict(object_id=pid,pixels=final_counts[pid],
            area_m2=predicted_areas[pid],centroid_x=x,centroid_y=y,
            status="matched" if m else "unmatched_candidate",
            matched_pile_id=m["pile_id"] if m else pd.NA,
            reference_coverage=m["reference_coverage"] if m else 0.0,
            match_iou=m["iou"] if m else 0.0))
    write_csv(folder/"object_metrics.csv",[summary])
    write_csv(folder/"pixel_metrics.csv",[pixel_summary])
    write_csv(folder/"candidate_matches.csv",candidates)
    write_csv(folder/"accepted_matches.csv",matches)
    write_csv(folder/"reference_detection.csv",ref_records)
    write_csv(folder/"predicted_objects.csv",object_records)
    if object_records:
        from shapely.geometry import Point
        points=gpd.GeoDataFrame(object_records,geometry=[Point(r["centroid_x"],r["centroid_y"]) for r in object_records],crs=GRID_CRS)
        points.to_file(folder/"predicted_object_centroids.gpkg",layer="objects",driver="GPKG")
    if EXPORT_POLYGON_FRAGMENTS and n_objects:
        import fiona
        from rasterio.features import shapes as raster_shapes
        schema={"geometry":"Polygon","properties":{"object_id":"int","status":"str"}}
        with rasterio.open(final_path) as final_src, fiona.open(folder/"object_fragments.gpkg","w",driver="GPKG",layer="fragments",schema=schema,crs_wkt=GRID_CRS.to_wkt()) as dst:
            for w in strips(height,width):
                lab=final_src.read(1,window=w)
                for geom,value in raster_shapes(lab,mask=lab>0,transform=rasterio.windows.transform(w,transform)):
                    pid=int(value)
                    dst.write({"geometry":geom,"properties":{"object_id":pid,"status":"matched" if pid in used_pred else "unmatched"}})
    # Remove only private scratch data produced during this run, never inputs.
    try:
        provisional.unlink()
    except OSError:
        warnings.warn(f"Could not remove scratch raster: {provisional}")
    print(f"  TP={tp:,}; unmatched={apparent_fp:,}; FN={fn:,}; recall={object_scores['recall']:.4f}; mapped area={summary['mapped_area_m2']:.1f} m2",flush=True)
    return summary,pixel_summary,ref_records


# =============================================================================
# 5. OPTIONAL PIXEL METRICS AGAINST A SEPARATE FULL LAND-COVER REFERENCE
# =============================================================================
def evaluate_full_reference(domain, original, refined, output):
    if PIXEL_REFERENCE_RASTER is None:
        return []
    path=require(Path(PIXEL_REFERENCE_RASTER),"Exhaustive pixel reference")
    results=[]
    with rasterio.open(path) as truth_src:
        offset=grid_offset(truth_src,domain["transform"])
        if offset[0]<0 or offset[1]<0 or offset[0]+domain["height"]>truth_src.height or offset[1]+domain["width"]>truth_src.width:
            raise ValueError("Full-reference raster does not cover the test area.")
        for name,src,src_offset in (("original",original,domain["old_offset"]),("refined",refined,domain["new_offset"])):
            for threshold in THRESHOLDS:
                counts=dict(tp=0,fp=0,fn=0,tn=0)
                excluded=0
                label_path=output/name/f"threshold_{threshold:.2f}"/"connected_objects.tif"
                with rasterio.open(label_path) as pred_src,rasterio.open(domain["valid_path"]) as valid_src:
                    for w in strips(domain["height"],domain["width"]):
                        valid=valid_src.read(1,window=w).astype(bool)
                        truth,tmask=read_on_grid(truth_src,offset,w)
                        labelled=valid&tmask&np.isfinite(truth)&np.isin(truth,list(range(1,11)))
                        excluded+=int((valid&~labelled).sum())
                        pred=pred_src.read(1,window=w)>0
                        pos=truth==1
                        counts["tp"]+=int((pred&pos&labelled).sum())
                        counts["fp"]+=int((pred&~pos&labelled).sum())
                        counts["fn"]+=int((~pred&pos&labelled).sum())
                        counts["tn"]+=int((~pred&~pos&labelled).sum())
                results.append(dict(model=name,threshold=threshold,**counts,**scores(**counts),
                    excluded_pixels=excluded,reference_raster=str(path),
                    prediction_rule="probability threshold with minimum-object-area filter"))
    write_csv(output/"full_reference_pixel_metrics.csv",results)
    return results


# =============================================================================
# 6. PAIRED RESULTS AND PROVENANCE
# =============================================================================
def make_comparison(rows,per_ref,out):
    df=pd.DataFrame(rows)
    df.to_csv(out/"spatial_object_metrics.csv",index=False)
    comparisons=[]
    for threshold in THRESHOLDS:
        old=df[(df.model=="original")&(df.threshold==threshold)].iloc[0]
        new=df[(df.model=="refined")&(df.threshold==threshold)].iloc[0]
        fields=["predicted_objects","true_positives","unmatched_predictions","false_positives","false_negatives",
                "object_precision","object_recall","object_f1","mapped_area_m2","mapped_area_error_m2",
                "pixel_precision","pixel_recall","pixel_f1","pixel_iou","pixel_accuracy"]
        for field in fields:
            a,b=old[field],new[field]
            comparisons.append(dict(threshold=threshold,metric=field,original=a,refined=b,
                                    difference=b-a,relative_change_percent=100*(b-a)/a if pd.notna(a) and a!=0 and pd.notna(b) else float("nan")))
        before=pd.DataFrame(per_ref[("original",threshold)])[ ["pile_id","detected"] ].rename(columns={"detected":"original_detected"})
        after=pd.DataFrame(per_ref[("refined",threshold)])[ ["pile_id","detected"] ].rename(columns={"detected":"refined_detected"})
        paired=before.merge(after,on="pile_id",validate="one_to_one")
        paired["change"]=np.select([paired.original_detected&paired.refined_detected,
            ~paired.original_detected&paired.refined_detected,
            paired.original_detected&~paired.refined_detected],
            ["detected_by_both","newly_detected","lost_detection"],default="missed_by_both")
        paired.to_csv(out/f"paired_reference_detection_{threshold:.2f}.csv",index=False)
    write_csv(out/"paired_changes.csv",comparisons)
    return df


def main():
    require(ORIGINAL_PROBABILITY,"Original probability raster")
    require(REFINED_PROBABILITY,"Refined probability raster")
    if ORIGINAL_PROBABILITY.resolve() == REFINED_PROBABILITY.resolve():
        raise ValueError("Original and refined probability paths point to the same file.")
    require(REFERENCE_WASTE,"Corrected manual waste reference")
    require(TEST_AREA,"Test-area polygon")
    if not (0<MIN_REFERENCE_COVERAGE<=1) or MIN_OBJECT_AREA_M2<=0:
        raise ValueError("Invalid object-matching configuration.")
    if CONNECTIVITY!=8:
        raise ValueError("This implementation uses 8-connectivity, as in the previous evaluation.")
    if any(not 0<t<=1 for t in THRESHOLDS):
        raise ValueError("Thresholds must lie in (0,1].")
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    out=OUTPUT_ROOT/datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out.mkdir(parents=True,exist_ok=False)
    print("\nPAIRED SPATIAL TEST-AREA EVALUATION")
    print("Original:",ORIGINAL_PROBABILITY)
    print("Refined :",REFINED_PROBABILITY)
    print("Reference:",REFERENCE_WASTE)
    print("Completeness confirmed:",REFERENCE_COMPLETE)
    print("Results:",out)
    manifest=dict(original=source_info(ORIGINAL_PROBABILITY),refined=source_info(REFINED_PROBABILITY),
        reference=source_info(REFERENCE_WASTE),test_area=source_info(TEST_AREA),
        reference_complete=REFERENCE_COMPLETE,thresholds=THRESHOLDS,
        minimum_object_area_m2=MIN_OBJECT_AREA_M2,minimum_reference_coverage=MIN_REFERENCE_COVERAGE,
        connectivity=CONNECTIVITY,strip_rows=STRIP_ROWS,pixel_reference=str(PIXEL_REFERENCE_RASTER),
        notes="Paired evaluation of existing map products; prior inference settings may differ. No threshold optimisation or model training. Spatial diagnostic test area previously informed error discovery; not an untouched independent final holdout.")
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    with rasterio.open(ORIGINAL_PROBABILITY) as original,rasterio.open(REFINED_PROBABILITY) as refined:
        domain=prepare_domain(original,refined,REFERENCE_WASTE,out)
        rows,pixel_rows,per_ref=[],[],{}
        for name,src,offset in (("original",original,domain["old_offset"]),("refined",refined,domain["new_offset"])):
            for threshold in THRESHOLDS:
                summary,pixels,refs=label_model(name,src,offset,threshold,domain,out)
                rows.append(summary)
                pixel_rows.append(pixels)
                per_ref[(name,threshold)]=refs
        df=make_comparison(rows,per_ref,out)
        write_csv(out/"spatial_pixel_metrics.csv",pixel_rows)
        evaluate_full_reference(domain,original,refined,out)
    # Report and preserve the audit trail, including uncertainty about reference completeness.
    lines=["PAIRED SPATIAL WASTE-MAP COMPARISON", "="*65,
        "Area: data/aoi/testarea.shp; NOT the 222 waste-centred validation tiles.",
        f"Reference: {REFERENCE_WASTE}",f"Reference completeness confirmed: {REFERENCE_COMPLETE}",
        f"Fixed thresholds: {THRESHOLDS}; minimum object area: {MIN_OBJECT_AREA_M2} m2",
        f"Object matching: 8-connected, >= {MIN_REFERENCE_COVERAGE:.1%} reference coverage, greedy one-to-one by IoU.",
        "The test area was used for error discovery; this is a paired spatial diagnostic evaluation, not an untouched external test.",
        "No thresholds were selected using these results.","",
        "MAIN RESULTS (0.50)"]
    for _,r in df[df.threshold==0.50].iterrows():
        lines += [f"\n{r.model}: predicted={r.predicted_objects:,}, TP={r.true_positives:,}, unmatched={r.unmatched_predictions:,}, FN={r.false_negatives:,}",
            f"Object precision={r.object_precision:.4f}; recall={r.object_recall:.4f}; F1={r.object_f1:.4f}",
            f"Mapped area={r.mapped_area_m2:.2f} m2; reference area={r.reference_area_m2:.2f} m2"]
    main_rows = df[df.threshold==0.50].set_index("model")
    if len(main_rows) == 2:
        old, new = main_rows.loc["original"], main_rows.loc["refined"]
        lines += ["", "REFINED MINUS ORIGINAL (0.50)",
            f"Predicted objects: {new.predicted_objects-old.predicted_objects:+,}",
            f"Matched reference piles: {new.true_positives-old.true_positives:+,}",
            f"Unmatched candidates: {new.unmatched_predictions-old.unmatched_predictions:+,}",
            f"Missed reference piles: {new.false_negatives-old.false_negatives:+,}",
            f"Mapped area: {new.mapped_area_m2-old.mapped_area_m2:+.2f} m2"]
        if REFERENCE_COMPLETE:
            lines += [f"Object precision change: {new.object_precision-old.object_precision:+.4f}",
                      f"Object recall change: {new.object_recall-old.object_recall:+.4f}",
                      f"Object F1 change: {new.object_f1-old.object_f1:+.4f}"]
    if not REFERENCE_COMPLETE:
        lines += ["", "WARNING: Reference completeness has not been confirmed. Unmatched predictions are candidates for review, not verified false positives. Precision/F1 are intentionally undefined. Apparent precision/F1 are saved separately for diagnostic comparison."]
    lines += ["", "The outputs use the same corrected reference and common valid raster pixels. If the source models used different inference mosaicking/halos, the comparison measures map-product changes rather than isolating fine-tuning effects.",
              "",f"Results directory: {out}"]
    report="\n".join(lines)
    (out/"README_results.txt").write_text(report,encoding="utf-8")
    print("\n"+report)
    print("\nCSV tables, component rasters, object centroids, reference snapshot and source manifest saved.")


if __name__=="__main__":
    main()
