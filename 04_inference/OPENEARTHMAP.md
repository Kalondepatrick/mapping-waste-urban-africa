# Pretrained aerial land-cover inference

`openearthmap_landcover.py` runs the official optical OpenEarthMap-SAR
U-Net/EfficientNet-B4 checkpoint without modifying the waste-model outputs.

It creates four georeferenced GeoTIFFs:

- `*_landcover.tif`: complete OpenEarthMap classification
- `*_building_probability.tif`: building probability
- `*_vegetation_probability.tif`: combined grass, tree, and cropland probability
- `*_developed_surface_probability.tif`: combined pavement and road probability

The probability rasters are stored as scaled `uint16`. A stored value of 7500
means probability 0.75. The scale (`0.0001`) is also recorded in the GeoTIFF.

## Environment

Use a separate environment because the existing model training code uses
TensorFlow while this official checkpoint uses PyTorch.

```powershell
py -m venv .venv-openearthmap
.venv-openearthmap\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r env\requirements-openearthmap.txt
```

## Run

The first command downloads the official checkpoint and processes an RGB
orthomosaic:

```powershell
python 04_inference_openearthmap_landcover.py `
  data\otherdata\644f83dd5874aa0006657732.tif `
  --download-model `
  --batch-size 4
```

Later runs reuse `data/pretrained_weights/openearthmap/RGB_Real_5_u-efficientnet-b4.pth`, so
`--download-model` is no longer necessary.

To use another raster or output directory:

```powershell
python 04_inference_openearthmap_landcover.py INPUT_RGB.tif `
  --output-dir results\openearthmap_landcover `
  --batch-size 4
```

Use `--batch-size 1` if GPU memory is limited. CPU inference is supported but
will be substantially slower. The input must contain at least three RGB bands;
a one-band raster is rejected with a clear error.

## Class IDs

| ID | OpenEarthMap class |
|---:|---|
| 0 | Background |
| 1 | Bareland |
| 2 | Grass/rangeland |
| 3 | Pavement/developed space |
| 4 | Road |
| 5 | Tree |
| 6 | Water |
| 7 | Cropland |
| 8 | Building |
| 255 | NoData |

Inspect these outputs against manually checked areas before using them to
filter waste detections. Buildings and vegetation should initially be treated
as contextual evidence, not hard exclusion masks.

## Model provenance

- Project: https://github.com/cliffbb/OpenEarthMap-SAR
- Checkpoint: `RGB_Real_5_u-efficientnet-b4.pth`
- Google Drive ID: `16REK9OcnhzrUte68hnBj7SJCqLpt1c_U`
- Architecture: U-Net, EfficientNet-B4 encoder, SCSE decoder attention
