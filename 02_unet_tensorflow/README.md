# 02 — U-Net, TensorFlow/Keras

The original trainer. Retained because the published baseline was produced with
it. **For new work use `../03_unet_pytorch/`**, which reproduces this baseline to
within 0.002 F1 and supports GPU on Windows.

| script | does |
|---|---|
| `01_train_unet_v4.py` | the canonical TensorFlow trainer |
| `02_check_initial_errors.py` | error inspection on the first trained model |
| `03_create_morechips.py` | targeted chip generation from observed errors |
| `04_inference.py` | tile-wise inference |

## Label semantics — important

Run with `--labels 1 2 3 4 5 6 7 8 9 --ignore-label 99`. **99 is a no-data fill,
not a class** (0.03 % of pixels). Versions v1–v3 in `../NotNeeded/` defaulted to
`range(1, 11)` and trained 99 as a tenth class; v4 fixes this and rejects a value
that appears in both `--labels` and `--ignore-label`.

TensorFlow has had no native Windows GPU support since 2.10. Run under WSL2 or
Linux for GPU; see `../env/README.md`.
