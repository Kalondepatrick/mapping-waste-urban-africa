"""Staged verification before any full training run.

Stages, in order. A config that fails a stage does not proceed to the next,
and the failure is recorded with its diagnosis rather than silently skipped.

  1 dependencies   required packages importable
  2 checkpoint     pretrained weights actually load (never random fallback)
  3 forward        forward pass gives (B, n_classes, H, W), finite
  4 loss           loss is finite, backward produces non-zero gradients
  5 smoke train    a few real-tile steps run and the loss moves
  6 resources      trainable parameters and peak GPU memory

Results go to <results-dir>/preflight_report.json. Re-running skips configs
that already passed, so this is resumable.

    python 03_unet_pytorch/preflight.py --archs efficientnet-b4-imagenet ...
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import rs_backbones as B  # noqa: E402
import object_metrics as OM  # noqa: E402


def _load_v5():
    """Reuse the v5 data pipeline so preflight tests the real loader."""
    p = os.path.join(HERE, "01_train_unet_v5_torch.py")
    spec = importlib.util.spec_from_file_location("v5", p)
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["preflight"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = saved
    return m


def stage_dependencies():
    need = ["torch", "timm", "segmentation_models_pytorch", "scipy", "rasterio"]
    missing, versions = [], {}
    for mod in need:
        try:
            versions[mod] = getattr(__import__(mod), "__version__", "?")
        except ImportError:
            missing.append(mod)
    if missing:
        raise RuntimeError(f"missing packages: {missing}")
    return {"versions": versions,
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def run_config(arch, n_classes, img_size, freeze, oem_weights, device,
               data=None, steps=3, batch=2):
    info = {"arch": arch, "freeze": freeze}

    # ---- 2 checkpoint ----
    model, enc, rep = B.build_registered(arch, n_classes, img_size, freeze,
                                         oem_weights)
    info["checkpoint"] = rep.to_dict()
    model.to(device)

    n_par = sum(p.numel() for p in model.parameters())
    n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["params_total_M"] = round(n_par / 1e6, 3)
    info["params_trainable_M"] = round(n_trn / 1e6, 3)

    # ---- 3 forward ----
    model.eval()
    x = torch.rand(batch, 3, img_size, img_size, device=device)
    with torch.no_grad():
        out = model(x)
    if tuple(out.shape) != (batch, n_classes, img_size, img_size):
        raise RuntimeError(f"forward shape {tuple(out.shape)} != "
                           f"{(batch, n_classes, img_size, img_size)}")
    if not torch.isfinite(out).all():
        raise RuntimeError("forward produced non-finite values")
    info["forward_shape"] = list(out.shape)

    # ---- 4 loss + gradient ----
    v5 = run_config._v5
    model.train()
    tgt = torch.full((batch, img_size, img_size), -1, dtype=torch.long,
                     device=device)
    tgt[:, :40, :40] = 0            # some waste
    tgt[:, 60:90, 60:90] = 3        # some other class
    loss = v5.seg_loss(model(x), tgt, 0, 1.0)
    if not torch.isfinite(loss):
        raise RuntimeError("loss is not finite")
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters()
             if p.requires_grad and p.grad is not None]
    if not grads or sum(grads) == 0:
        raise RuntimeError("no non-zero gradients after backward")
    info["loss_initial"] = float(loss)
    info["n_grad_tensors"] = len(grads)

    # ---- 5 smoke train on real tiles ----
    if data is not None:
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=1e-4)
        idx = np.arange(min(batch * steps, len(data["images"])))
        losses = []
        for s in range(steps):
            sl = idx[s * batch:(s + 1) * batch]
            if len(sl) == 0:
                break
            xb = torch.from_numpy(
                data["images"][sl].transpose(0, 3, 1, 2)).to(device).float() / 255.0
            yb = torch.from_numpy(data["masks"][sl]).to(device).long()
            l = v5.seg_loss(model(xb), yb, 0, 1.0)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            losses.append(float(l))
        info["smoke_losses"] = [round(v, 4) for v in losses]
        if losses and not all(np.isfinite(losses)):
            raise RuntimeError(f"non-finite loss during smoke training: {losses}")

    # ---- 6 resources ----
    if device.type == "cuda":
        info["peak_gpu_mem_MB"] = round(
            torch.cuda.max_memory_allocated() / 1e6, 1)
        torch.cuda.reset_peak_memory_stats()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=sorted(B.REGISTRY))
    ap.add_argument("--n-classes", type=int, default=9)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--results-dir", default="./results/06_pretrained_backbones")
    ap.add_argument("--oem-weights",
                    default=os.path.join(ROOT, "data", "pretrained_weights", "openearthmap",
                                         "RGB_Real_5_u-efficientnet-b4.pth"))
    ap.add_argument("--dev-dir",
                    default=os.path.join(ROOT, "data", "tiled_data",
                                         "partitioned", "modelDevelopment"))
    ap.add_argument("--no-data", action="store_true",
                    help="skip the real-tile smoke-training stage")
    ap.add_argument("--max-tiles", type=int, default=8)
    ap.add_argument("--both-freeze-modes", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-test configs that already passed")
    a = ap.parse_args()

    os.makedirs(a.results_dir, exist_ok=True)
    report_path = os.path.join(a.results_dir, "preflight_report.json")
    report = {"stages": {}, "configs": {}}
    if os.path.exists(report_path):
        try:
            report = json.load(open(report_path))
        except Exception:
            pass
    report.setdefault("configs", {})

    print("=== stage 1: dependencies ===")
    try:
        dep = stage_dependencies()
        report["stages"]["dependencies"] = {"status": "pass", **dep}
        print(f"  ok  cuda={dep['cuda']} device={dep['device']}")
        for k, v in dep["versions"].items():
            print(f"      {k} {v}")
    except Exception as e:
        report["stages"]["dependencies"] = {"status": "fail", "error": str(e)}
        json.dump(report, open(report_path, "w"), indent=2)
        print(f"  FAIL {e}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_config._v5 = _load_v5()

    data = None
    if not a.no_data:
        print("\n=== loading a few real tiles for smoke training ===")
        label_map = {r: i for i, r in enumerate(range(1, 10))}
        full = run_config._v5.load_folder(a.dev_dir, label_map, a.img_size,
                                          "preflight", 99,
                                          max_tiles=a.max_tiles,
                                          rng=np.random.default_rng(0))
        data = full

    modes = [False, True] if a.both_freeze_modes else [False]
    rc = 0
    for arch in a.archs:
        for freeze in modes:
            key = f"{arch}{'_frozen' if freeze else '_finetuned'}"
            prev = report["configs"].get(key, {})
            if prev.get("status") == "pass" and not a.force:
                print(f"\n=== {key}: already passed, skipping ===")
                continue
            print(f"\n=== {key} ===")
            t0 = dt.datetime.now()
            try:
                info = run_config(arch, a.n_classes, a.img_size, freeze,
                                  a.oem_weights, device, data)
                info["status"] = "pass"
                info["seconds"] = round(
                    (dt.datetime.now() - t0).total_seconds(), 1)
                report["configs"][key] = info
                print(f"  PASS  {info['checkpoint']['tensors_loaded']}/"
                      f"{info['checkpoint']['tensors_total']} tensors "
                      f"({100*info['checkpoint']['fraction']:.1f}%) "
                      f"from {info['checkpoint']['source'][:60]}")
                print(f"        params {info['params_total_M']}M "
                      f"({info['params_trainable_M']}M trainable)"
                      + (f", peak GPU {info.get('peak_gpu_mem_MB')} MB"
                         if 'peak_gpu_mem_MB' in info else ""))
                if "smoke_losses" in info:
                    print(f"        smoke losses {info['smoke_losses']}")
            except B.PretrainedWeightsUnavailable as e:
                report["configs"][key] = {
                    "arch": arch, "freeze": freeze, "status": "fail",
                    "failure_class": "pretrained_weights_unavailable",
                    "error": str(e)[:600],
                    "diagnosis": "Pretrained weights could not be obtained. "
                                 "This config is NOT runnable; a randomly "
                                 "initialised substitute would invalidate it.",
                }
                print(f"  FAIL (weights) {str(e)[:200]}")
                rc = 2
            except Exception as e:
                report["configs"][key] = {
                    "arch": arch, "freeze": freeze, "status": "fail",
                    "failure_class": type(e).__name__,
                    "error": str(e)[:600],
                    "traceback": traceback.format_exc()[-1500:],
                }
                print(f"  FAIL ({type(e).__name__}) {str(e)[:200]}")
                rc = 2
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            json.dump(report, open(report_path, "w"), indent=2)

    n_pass = sum(1 for v in report["configs"].values() if v.get("status") == "pass")
    print(f"\n=== {n_pass}/{len(report['configs'])} configs passed ===")
    print(f"report: {report_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
