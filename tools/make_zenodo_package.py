"""Assemble a Zenodo upload tree from data/, plus a SHA-256 manifest.

The full data/ folder is ~97.8 GB, over Zenodo's 50 GB per-record limit, so the
material is tiered. Each tier is a superset of the one before it.

    core      7.4 GB   reproduces every model result in the paper  <- default
    imagery  26.4 GB   core + the source orthomosaic
    extended 45.9 GB   imagery + multiband, haralick and hard-negative tiles

The orthomosaic is published separately on OpenAerialMap, so the `imagery` tier
is normally unnecessary: deposit `core` and cite OpenAerialMap for the source.
Every tile was cut from that orthomosaic, and all training and evaluation run
from the tiles, so `core` is fully reproducible without it.

Derived products that any reproducer can regenerate are never included:
predictions/ (31.6 GB), annotations/ (15.0 GB), otherdata/, finetuning/, and the
superseded intermediate tiling folders.

Usage
-----
    python tools/make_zenodo_package.py --manifest
    python tools/make_zenodo_package.py --tier core --out ../zenodo_core
    python tools/make_zenodo_package.py --tier imagery --out D:/zenodo --copy

By default entries are hard-linked, so staging 26 GB costs no extra disk and is
near-instantaneous. Hard links require the destination to be on the same volume
as the repository; use --copy otherwise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

# Paths are relative to data/. Order is cosmetic; tiers are cumulative.
TIERS = {
    "core": [
        "tiled_data/partitioned",
        "tiled_data/masks",
        "tiled_data/masks_bin",
        "tiled_data/spatial_metadata",
        "tiled_data/finetuning_hard_negatives_complete",
        "tiled_data/training_sample_selection",
        "splits",
        "vectors",
        "aoi",
        "waste",
        "river",
        "mask",
        "pretrained_weights",
        "trained_models",
        "README.md",
    ],
    "imagery": [
        "optical.tiff",
        "optical.tiff.aux.xml",
    ],
    "extended": [
        "tiled_data/multiband",
        "tiled_data/haralick",
        "tiled_data/complementary_hard_negatives_v2",
        "tiled_data/segments",
        "tiled_data/segments_with_var",
        "tiled_data/vectors",
    ],
}
ORDER = ["core", "imagery", "extended"]

EXCLUDED_REASON = {
    "predictions": "derived; regenerate with 04_inference/ (31.6 GB)",
    "annotations": "derived preview chips; regenerate from optical.tiff + vectors (15.0 GB)",
    "otherdata": "imagery for other dates, not used by the reported results (3.0 GB)",
    "finetuning": "intermediate fine-tuning scratch (1.5 GB)",
    "geobia_predictions": "derived; regenerate with 01_geobia/ ",
    "raster_segmentation_results": "derived; regenerate with 01_geobia/",
    "visuals": "figure scratch",
    "tiled_data/images": "superseded by tiled_data/partitioned/",
    "tiled_data/keep": "superseded by tiled_data/partitioned/",
    "tiled_data/redigitized": "superseded by tiled_data/partitioned/",
    "tiled_data/test": "superseded by tiled_data/partitioned/Validation/",
    "tiled_data/preview_cache": "cache",
    "tiled_data/finetuning_hard_negatives": "superseded by *_complete",
    "tiled_data/complementary_hard_negatives": "empty; superseded by _v2",
}


def paths_for(tier: str) -> list[str]:
    out: list[str] = []
    for name in ORDER:
        out += TIERS[name]
        if name == tier:
            return out
    raise SystemExit(f"unknown tier {tier!r}; choose from {ORDER}")


def walk_size(path: str) -> tuple[int, int]:
    if os.path.isfile(path):
        return 1, os.path.getsize(path)
    n = b = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                b += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return n, b


def sha256(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(buf)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def tier_of(rel: str) -> str:
    """Lowest tier that includes `rel`, or 'mixed' for a partly-included parent."""
    for name in ORDER:
        for p in TIERS[name]:
            if rel == p or rel.startswith(p + "/"):
                return name
    for name in ORDER:
        for p in TIERS[name]:
            if p.startswith(rel + "/"):
                return "mixed"          # parent of included children; see its rows
    return "excluded"


def write_manifest() -> None:
    """Describe every top-level entry of data/ with size and tier."""
    rows = []
    for name in sorted(os.listdir(DATA)):
        if name in ("MANIFEST.csv",):
            continue
        p = os.path.join(DATA, name)
        n, b = walk_size(p)
        rows.append({
            "path": name,
            "kind": "directory" if os.path.isdir(p) else "file",
            "files": n,
            "bytes": b,
            "gigabytes": round(b / 1e9, 3),
            "zenodo_tier": tier_of(name),
            "note": EXCLUDED_REASON.get(name, ""),
        })
    # tiled_data is mixed, so break it out one level deeper
    td = os.path.join(DATA, "tiled_data")
    if os.path.isdir(td):
        for name in sorted(os.listdir(td)):
            rel = f"tiled_data/{name}"
            p = os.path.join(td, name)
            n, b = walk_size(p)
            rows.append({
                "path": rel,
                "kind": "directory" if os.path.isdir(p) else "file",
                "files": n,
                "bytes": b,
                "gigabytes": round(b / 1e9, 3),
                "zenodo_tier": tier_of(rel),
                "note": EXCLUDED_REASON.get(rel, ""),
            })
    out = os.path.join(DATA, "MANIFEST.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # Count each byte once: use the tiled_data/* breakout rows in place of the
    # tiled_data parent row, and every other top-level row as-is.
    leaves = [r for r in rows
              if r["path"].startswith("tiled_data/")
              or (r["path"] != "tiled_data" and "/" not in r["path"])]
    tot = sum(r["bytes"] for r in leaves)
    print(f"wrote {out}  ({len(rows)} entries, {tot/1e9:.1f} GB total)")
    running = 0.0
    for t in ORDER:
        b = sum(r["bytes"] for r in leaves if r["zenodo_tier"] == t)
        running += b / 1e9
        print(f"  tier {t:9s} adds {b/1e9:6.1f} GB  -> cumulative {running:6.1f} GB")
    exc = sum(r["bytes"] for r in leaves if r["zenodo_tier"] == "excluded")
    print(f"  excluded (regenerable) {exc/1e9:6.1f} GB")


def build(tier: str, out_dir: str, copy: bool, checksums: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    staged = 0
    total_bytes = 0
    for rel in paths_for(tier):
        src = os.path.join(DATA, rel.replace("/", os.sep))
        if not os.path.exists(src):
            print(f"  skip (missing): {rel}")
            continue
        dst = os.path.join(out_dir, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isfile(src):
            pairs = [(src, dst)]
        else:
            pairs = []
            for root, _, files in os.walk(src):
                for f in files:
                    s = os.path.join(root, f)
                    pairs.append((s, os.path.join(dst, os.path.relpath(s, src))))
        for s, d in pairs:
            if os.path.exists(d):
                continue
            os.makedirs(os.path.dirname(d), exist_ok=True)
            try:
                if copy:
                    shutil.copy2(s, d)
                else:
                    os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
            staged += 1
            total_bytes += os.path.getsize(d)
        print(f"  staged {rel}")
    print(f"\n{staged} files, {total_bytes/1e9:.2f} GB -> {out_dir}")

    if checksums:
        man = os.path.join(out_dir, "SHA256SUMS.txt")
        print("computing checksums (this reads every byte) ...")
        with open(man, "w", encoding="utf-8", newline="\n") as fh:
            for root, _, files in os.walk(out_dir):
                for f in sorted(files):
                    if f == "SHA256SUMS.txt":
                        continue
                    p = os.path.join(root, f)
                    rel = os.path.relpath(p, out_dir).replace(os.sep, "/")
                    fh.write(f"{sha256(p)}  {rel}\n")
        print(f"wrote {man}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="store_true",
                    help="(re)write data/MANIFEST.csv and exit")
    ap.add_argument("--tier", choices=ORDER, default="core")
    ap.add_argument("--out", default=os.path.join(ROOT, os.pardir, "zenodo_package"))
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of hard-linking (needed across volumes)")
    ap.add_argument("--no-checksums", action="store_true")
    a = ap.parse_args()

    if not os.path.isdir(DATA):
        print(f"data/ not found at {DATA}", file=sys.stderr)
        return 1
    if a.manifest:
        write_manifest()
        return 0
    build(a.tier, os.path.abspath(a.out), a.copy, not a.no_checksums)
    return 0


if __name__ == "__main__":
    sys.exit(main())
