"""Object-level (per waste pile) evaluation, kept separate from pixel-level.

Why separate
------------
Pixel prevalence and object prevalence are different quantities and must not
be mixed. A pixel-level precision of 0.72 and an object-level precision of
0.02 are both true of the same map; they answer different questions:

  pixel-level   how much of the predicted waste AREA is really waste
  object-level  how many of the predicted PILES are really piles

The paper reports both, labelled, and never converts one into the other.

Caveat recorded for the methods section: connected components are computed
per tile, so a pile crossing a tile boundary is counted once per tile it
touches. Tiles are 256 px (9.75 m) and piles are typically far smaller, so
this affects few objects, but it is a known upward bias on object counts.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Ndirande orthomosaic ground sample distance
DEFAULT_GSD_M = 0.0381


def _objects(mask2d, min_px):
    lab, n = ndimage.label(mask2d)
    if n == 0:
        return lab, []
    sizes = ndimage.sum(mask2d, lab, index=np.arange(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_px]
    return lab, keep


def object_counts(truth, pred, gsd_m=DEFAULT_GSD_M, min_area_m2=0.25,
                  iou_thresh=0.1, valid=None):
    """Object-level confusion over a stack of tiles.

    truth, pred : (N, H, W) boolean arrays
    valid       : (N, H, W) boolean, pixels that count (excludes ignore label)
    min_area_m2 : objects smaller than this are dropped from BOTH sets
    iou_thresh  : a predicted object matches a reference object when their
                  intersection-over-union is at least this. Boundaries of
                  waste piles are fuzzy, so the default is deliberately
                  lenient; results are also reported at stricter values.

    Returns a dict of counts and derived rates, plus the area assessed.
    """
    px_area = gsd_m ** 2
    min_px = max(int(round(min_area_m2 / px_area)), 1)

    tp = fp = fn = 0
    n_ref = n_pred = 0
    for i in range(len(truth)):
        t = truth[i]
        p = pred[i]
        if valid is not None:
            t = t & valid[i]
            p = p & valid[i]
        tlab, tkeep = _objects(t, min_px)
        plab, pkeep = _objects(p, min_px)
        n_ref += len(tkeep)
        n_pred += len(pkeep)
        if not tkeep and not pkeep:
            continue

        matched_ref, matched_pred = set(), set()
        for pi in pkeep:
            pm = plab == pi
            psum = pm.sum()
            best, best_iou = None, 0.0
            for ti in tkeep:
                if ti in matched_ref:
                    continue
                tm = tlab == ti
                inter = np.logical_and(pm, tm).sum()
                if inter == 0:
                    continue
                iou = inter / (psum + tm.sum() - inter)
                if iou > best_iou:
                    best, best_iou = ti, iou
            if best is not None and best_iou >= iou_thresh:
                matched_ref.add(best)
                matched_pred.add(pi)
        tp += len(matched_pred)
        fp += len(pkeep) - len(matched_pred)
        fn += len(tkeep) - len(matched_ref)

    n_px = int(np.prod(truth.shape))
    if valid is not None:
        n_px = int(valid.sum())
    area_km2 = n_px * px_area / 1e6
    eps = 1e-12
    return {
        "obj_tp": tp, "obj_fp": fp, "obj_fn": fn,
        "obj_n_reference": n_ref, "obj_n_predicted": n_pred,
        "obj_precision": tp / (tp + fp + eps),
        "obj_recall": tp / (tp + fn + eps),
        "obj_f1": 2 * tp / (2 * tp + fp + fn + eps),
        "assessed_area_km2": area_km2,
        "fp_objects_per_km2": fp / (area_km2 + eps),
        "ref_objects_per_km2": n_ref / (area_km2 + eps),
        "obj_prevalence_per_km2": n_ref / (area_km2 + eps),
        "min_area_m2": min_area_m2,
        "iou_thresh": iou_thresh,
        "gsd_m": gsd_m,
    }


def object_metrics_multi(truth, pred, valid=None, gsd_m=DEFAULT_GSD_M,
                         min_area_m2=0.25, iou_threshs=(0.1, 0.25, 0.5)):
    """Object metrics at several IoU thresholds, flattened for one CSV row."""
    out = {}
    for thr in iou_threshs:
        r = object_counts(truth, pred, gsd_m, min_area_m2, thr, valid)
        suffix = f"_iou{thr:g}"
        for k, v in r.items():
            if k in ("min_area_m2", "gsd_m", "assessed_area_km2",
                     "ref_objects_per_km2", "obj_prevalence_per_km2",
                     "obj_n_reference"):
                out[k] = v            # threshold-independent, store once
            elif k == "iou_thresh":
                continue
            else:
                out[k + suffix] = v
    return out
