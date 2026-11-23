"""Run pretrained OpenEarthMap land-cover segmentation on an RGB GeoTIFF.

The script uses the official OpenEarthMap-SAR optical U-Net/EfficientNet-B4
checkpoint. It writes four georeferenced rasters without changing or filtering
the project's waste predictions:

* ``*_landcover.tif``: OpenEarthMap class IDs (0-8, 255 = nodata)
* ``*_building_probability.tif``: building probability
* ``*_vegetation_probability.tif``: grass + tree + cropland probability
* ``*_developed_surface_probability.tif``: pavement + road probability

Probability rasters are uint16 values from 0 to 10000. Their GeoTIFF scale is
0.0001, so a stored value of 7500 represents probability 0.75.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.jit\.(script|interface)` is deprecated.*",
    category=FutureWarning,
)

import segmentation_models_pytorch as smp


OPENEARTHMAP_CHECKPOINT_ID = "16REK9OcnhzrUte68hnBj7SJCqLpt1c_U"
N_CLASSES = 9
BACKGROUND_IDX = 0
BARELAND_IDX = 1
GRASS_IDX = 2
PAVEMENT_IDX = 3
ROAD_IDX = 4
TREE_IDX = 5
WATER_IDX = 6
CROPLAND_IDX = 7
BUILDING_IDX = 8

CLASS_NAMES = {
    0: "background",
    1: "bareland",
    2: "grass_rangeland",
    3: "pavement_developed_space",
    4: "road",
    5: "tree",
    6: "water",
    7: "cropland",
    8: "building",
}

CLASS_COLORS = {
    0: (0, 0, 0, 255),
    1: (128, 0, 0, 255),
    2: (0, 255, 36, 255),
    3: (148, 148, 148, 255),
    4: (255, 255, 255, 255),
    5: (34, 97, 38, 255),
    6: (0, 69, 255, 255),
    7: (75, 181, 73, 255),
    8: (222, 31, 7, 255),
    255: (0, 0, 0, 0),
}

PROBABILITY_SCALE = 10_000
PROBABILITY_NODATA = np.uint16(65_535)
CLASS_NODATA = np.uint8(255)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Map buildings, vegetation, developed surfaces, and all "
            "OpenEarthMap classes in an RGB aerial GeoTIFF."
        )
    )
    parser.add_argument("input", type=Path, help="Input RGB GeoTIFF.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            project_root
            / "models"
            / "openearthmap"
            / "RGB_Real_5_u-efficientnet-b4.pth"
        ),
        help="Official optical OpenEarthMap checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "openearthmap_landcover",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the official checkpoint from Google Drive if missing.",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--halo",
        type=int,
        default=64,
        help="Context pixels discarded on each tile edge to reduce seams.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--input-scale",
        type=float,
        default=255.0,
        help="Divide RGB values by this number; official RGB inference uses 255.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs for this input.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input raster not found: {args.input}")
    if args.tile_size <= 0 or args.tile_size % 32 != 0:
        raise ValueError("--tile-size must be positive and divisible by 32.")
    if args.halo < 0 or 2 * args.halo >= args.tile_size:
        raise ValueError("--halo must satisfy 0 <= 2 * halo < tile-size.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.input_scale <= 0:
        raise ValueError("--input-scale must be positive.")


def ensure_checkpoint(path: Path, download: bool) -> None:
    if path.is_file():
        return
    if not download:
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run again with --download-model, or download the official optical "
            "checkpoint manually."
        )

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required for --download-model. Install the packages in "
            "env/requirements-openearthmap.txt."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".download")
    print(f"Downloading official OpenEarthMap checkpoint to {path} ...")
    result = gdown.download(
        id=OPENEARTHMAP_CHECKPOINT_ID,
        output=str(temporary_path),
        quiet=False,
    )
    if result is None or not temporary_path.is_file():
        raise RuntimeError("Checkpoint download failed.")
    temporary_path.replace(path)


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint is not a state dictionary.")
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break
    state_dict = {}
    for key, value in checkpoint.items():
        clean_key = key.removeprefix("module.")
        state_dict[clean_key] = value
    return state_dict


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = smp.Unet(
        classes=N_CLASSES,
        in_channels=3,
        activation=None,
        encoder_weights=None,
        encoder_name="efficientnet-b4",
        decoder_attention_type="scse",
    )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:  # PyTorch before weights_only was introduced.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
    return model.to(device).eval()


def core_windows(
    width: int,
    height: int,
    core_size: int,
) -> Iterator[Window]:
    for row in range(0, height, core_size):
        for col in range(0, width, core_size):
            yield Window(
                col_off=col,
                row_off=row,
                width=min(core_size, width - col),
                height=min(core_size, height - row),
            )


def batches(items: Sequence[Window], size: int) -> Iterator[Sequence[Window]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_model_tile(
    src: rasterio.io.DatasetReader,
    core: Window,
    tile_size: int,
    halo: int,
    input_scale: float,
) -> np.ndarray:
    model_window = Window(
        col_off=int(core.col_off) - halo,
        row_off=int(core.row_off) - halo,
        width=tile_size,
        height=tile_size,
    )
    image = src.read(
        (1, 2, 3),
        window=model_window,
        boundless=True,
        fill_value=0,
        out_dtype="float32",
    )
    image /= input_scale
    return image


def probability_to_uint16(probability: np.ndarray) -> np.ndarray:
    return np.rint(
        np.clip(probability, 0.0, 1.0) * PROBABILITY_SCALE
    ).astype("uint16")


def output_paths(output_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "landcover": output_dir / f"{stem}_landcover.tif",
        "building": output_dir / f"{stem}_building_probability.tif",
        "vegetation": output_dir / f"{stem}_vegetation_probability.tif",
        "developed": output_dir / f"{stem}_developed_surface_probability.tif",
    }


def prepare_output_paths(
    final_paths: dict[str, Path], overwrite: bool
) -> dict[str, Path]:
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not overwrite:
        listing = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n"
            f"{listing}"
        )
    partial_paths = {
        name: path.with_name(path.stem + ".partial" + path.suffix)
        for name, path in final_paths.items()
    }
    stale = [path for path in partial_paths.values() if path.exists()]
    if stale and not overwrite:
        listing = "\n".join(str(path) for path in stale)
        raise FileExistsError(
            "Incomplete files from an earlier run exist. Remove them or use "
            f"--overwrite:\n{listing}"
        )
    if overwrite:
        for path in (*existing, *stale):
            path.unlink()
    return partial_paths


def make_profiles(src: rasterio.io.DatasetReader) -> tuple[dict, dict]:
    common = {
        "driver": "GTiff",
        "width": src.width,
        "height": src.height,
        "count": 1,
        "crs": src.crs,
        "transform": src.transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    class_profile = {**common, "dtype": "uint8", "nodata": int(CLASS_NODATA)}
    probability_profile = {
        **common,
        "dtype": "uint16",
        "nodata": int(PROBABILITY_NODATA),
        "predictor": 2,
    }
    return class_profile, probability_profile


def write_metadata(
    writers: dict[str, rasterio.io.DatasetWriter],
    checkpoint: Path,
) -> None:
    writers["landcover"].write_colormap(1, CLASS_COLORS)
    writers["landcover"].set_band_description(1, "OpenEarthMap class ID")
    writers["landcover"].update_tags(
        class_names=json.dumps(CLASS_NAMES),
        model="OpenEarthMap-SAR optical U-Net EfficientNet-B4",
        checkpoint=checkpoint.name,
    )
    descriptions = {
        "building": "Building probability",
        "vegetation": "Vegetation probability: grass + tree + cropland",
        "developed": "Developed-surface probability: pavement + road",
    }
    for name, description in descriptions.items():
        writer = writers[name]
        writer.scales = (1.0 / PROBABILITY_SCALE,)
        writer.offsets = (0.0,)
        writer.set_band_description(1, description)
        writer.update_tags(
            probability_scale=str(1.0 / PROBABILITY_SCALE),
            model="OpenEarthMap-SAR optical U-Net EfficientNet-B4",
            checkpoint=checkpoint.name,
        )


def run_inference(args: argparse.Namespace) -> dict[str, Path]:
    validate_args(args)
    ensure_checkpoint(args.checkpoint, args.download_model)
    device = select_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = output_paths(args.output_dir, args.input.stem)
    partial_paths = prepare_output_paths(final_paths, args.overwrite)
    core_size = args.tile_size - 2 * args.halo

    try:
        with rasterio.open(args.input) as src:
            if src.count < 3:
                raise ValueError(
                    f"OpenEarthMap requires RGB input, but {args.input} has "
                    f"only {src.count} band(s)."
                )
            windows = list(core_windows(src.width, src.height, core_size))
            total_batches = math.ceil(len(windows) / args.batch_size)
            class_profile, probability_profile = make_profiles(src)

            with ExitStack() as stack:
                writers = {
                    "landcover": stack.enter_context(
                        rasterio.open(partial_paths["landcover"], "w", **class_profile)
                    ),
                    "building": stack.enter_context(
                        rasterio.open(partial_paths["building"], "w", **probability_profile)
                    ),
                    "vegetation": stack.enter_context(
                        rasterio.open(partial_paths["vegetation"], "w", **probability_profile)
                    ),
                    "developed": stack.enter_context(
                        rasterio.open(partial_paths["developed"], "w", **probability_profile)
                    ),
                }
                write_metadata(writers, args.checkpoint)

                with torch.inference_mode():
                    for batch_number, window_batch in enumerate(
                        batches(windows, args.batch_size), start=1
                    ):
                        images = np.stack(
                            [
                                read_model_tile(
                                    src,
                                    window,
                                    args.tile_size,
                                    args.halo,
                                    args.input_scale,
                                )
                                for window in window_batch
                            ]
                        )
                        tensor = torch.from_numpy(images).to(device)
                        probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()

                        for window, probability in zip(window_batch, probabilities):
                            height = int(window.height)
                            width = int(window.width)
                            row_slice = slice(args.halo, args.halo + height)
                            col_slice = slice(args.halo, args.halo + width)
                            probability = probability[:, row_slice, col_slice]
                            valid = src.dataset_mask(window=window) > 0

                            landcover = probability.argmax(axis=0).astype("uint8")
                            building = probability[BUILDING_IDX]
                            vegetation = probability[
                                [GRASS_IDX, TREE_IDX, CROPLAND_IDX]
                            ].sum(axis=0)
                            developed = probability[[PAVEMENT_IDX, ROAD_IDX]].sum(axis=0)

                            landcover[~valid] = CLASS_NODATA
                            writers["landcover"].write(landcover, 1, window=window)

                            for name, layer in (
                                ("building", building),
                                ("vegetation", vegetation),
                                ("developed", developed),
                            ):
                                encoded = probability_to_uint16(layer)
                                encoded[~valid] = PROBABILITY_NODATA
                                writers[name].write(encoded, 1, window=window)

                        print(
                            f"Batch {batch_number:,}/{total_batches:,} "
                            f"({min(batch_number * args.batch_size, len(windows)):,}/"
                            f"{len(windows):,} tiles)",
                            end="\r",
                            flush=True,
                        )
                print()

        for name, partial_path in partial_paths.items():
            partial_path.replace(final_paths[name])

        manifest = args.output_dir / f"{args.input.stem}_landcover_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "input": str(args.input.resolve()),
                    "checkpoint": str(args.checkpoint.resolve()),
                    "model": "OpenEarthMap-SAR optical U-Net EfficientNet-B4",
                    "class_names": CLASS_NAMES,
                    "vegetation_channels": [GRASS_IDX, TREE_IDX, CROPLAND_IDX],
                    "developed_surface_channels": [PAVEMENT_IDX, ROAD_IDX],
                    "building_channel": BUILDING_IDX,
                    "probability_storage": {
                        "dtype": "uint16",
                        "scale": 1.0 / PROBABILITY_SCALE,
                        "nodata": int(PROBABILITY_NODATA),
                    },
                    "tile_size": args.tile_size,
                    "halo": args.halo,
                    "input_scale": args.input_scale,
                    "outputs": {name: str(path.resolve()) for name, path in final_paths.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("Land-cover inference complete:")
        for path in final_paths.values():
            print(f"  {path}")
        print(f"  {manifest}")
        return final_paths
    except Exception:
        print(
            "Inference did not complete. Any .partial.tif files are incomplete "
            "and are never treated as final outputs."
        )
        raise


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
