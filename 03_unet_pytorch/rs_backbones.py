"""Remote-sensing and general-purpose pretrained backbones, with STRICT
checkpoint verification.

Design rule, non-negotiable
---------------------------
If pretrained weights are requested and cannot be loaded, this module raises.
It never silently returns a randomly initialised model. A row in a results
table labelled "pretrained" must have had pretrained weights in it.

Every loader returns a LoadReport recording exactly how many parameter tensors
were transferred, how many were left at their initialised values, and where
the checkpoint came from, so the paper can state this per model.

Checkpoint sources
------------------
  imagenet-b0/b4   torchvision/timm ImageNet-1k via segmentation_models_pytorch
  openearthmap-b4  data/pretrained_weights/openearthmap/RGB_Real_5_u-efficientnet-b4.pth
                   (official OpenEarthMap-SAR optical U-Net/EfficientNet-B4;
                   9 land-cover classes, scSE decoder attention)
  dinov3-vitl      timm vit_large_patch16_dinov3.lvd1689m   (LVD-1689M, web)
  dinov3-sat-vitl  timm vit_large_patch16_dinov3.sat493m    (SAT-493M, satellite)
  satlas-aerial    SatlasPretrain Aerial_SwinB_SI (single-image aerial, Swin-v2-B)
                   via the satlaspretrain_models package.

DINOv3 weights are released under the DINOv3 licence, not Apache/MIT. Check
the terms before publishing results derived from them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoadReport:
    """Record of what was actually transferred into a model."""
    name: str
    source: str
    pretrained_requested: bool
    tensors_total: int = 0
    tensors_loaded: int = 0
    tensors_missing: list = field(default_factory=list)
    tensors_shape_mismatch: list = field(default_factory=list)
    head_replaced: str = ""
    notes: str = ""

    @property
    def fraction(self):
        return self.tensors_loaded / max(self.tensors_total, 1)

    def summary(self):
        return (f"{self.name}: {self.tensors_loaded}/{self.tensors_total} "
                f"tensors ({100 * self.fraction:.1f}%) from {self.source}")

    def to_dict(self):
        d = asdict(self)
        d["fraction"] = self.fraction
        d["tensors_missing"] = self.tensors_missing[:20]
        d["tensors_shape_mismatch"] = self.tensors_shape_mismatch[:20]
        return d


class PretrainedWeightsUnavailable(RuntimeError):
    """Raised when pretrained weights were requested but could not be loaded.

    Deliberately fatal: a randomly initialised stand-in would silently
    invalidate the experiment it is standing in for.
    """


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalize(nn.Module):
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.register_buffer("m", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.m) / self.s


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


# =============================================================================
# smp U-Net encoders (ImageNet, and OpenEarthMap via checkpoint)
# =============================================================================
class SmpUNet(nn.Module):
    def __init__(self, encoder, n_classes, encoder_weights="imagenet",
                 decoder_attention_type=None):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.norm = Normalize()
        self.net = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                            in_channels=3, classes=n_classes,
                            decoder_attention_type=decoder_attention_type)

    @property
    def encoder(self):
        return self.net.encoder

    def forward(self, x):
        return self.net(self.norm(x))


def build_imagenet_unet(encoder, n_classes):
    """smp U-Net with ImageNet encoder. smp raises if the download fails."""
    try:
        m = SmpUNet(encoder, n_classes, encoder_weights="imagenet")
    except Exception as e:
        raise PretrainedWeightsUnavailable(
            f"ImageNet weights for {encoder} could not be loaded: "
            f"{type(e).__name__}: {e}") from e
    enc_sd = m.net.encoder.state_dict()
    rep = LoadReport(
        name=f"unet-{encoder}-imagenet",
        source="segmentation_models_pytorch / ImageNet-1k",
        pretrained_requested=True,
        tensors_total=len(enc_sd), tensors_loaded=len(enc_sd),
        head_replaced=f"smp segmentation_head -> {n_classes} classes",
        notes="encoder weights supplied by smp; decoder and head trained from scratch")
    return m, m.encoder, rep


def build_openearthmap_unet(n_classes, ckpt_path, min_fraction=0.90):
    """EfficientNet-B4 U-Net initialised from the OpenEarthMap checkpoint.

    The OEM head predicts its own 9 land-cover classes and is discarded; the
    encoder and decoder carry the aerial-domain pretraining.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise PretrainedWeightsUnavailable(
            f"OpenEarthMap checkpoint not found at {ckpt_path!r}. Refusing to "
            f"train a randomly initialised model under the label "
            f"'openearthmap-pretrained'.")
    m = SmpUNet("efficientnet-b4", n_classes, encoder_weights=None,
                decoder_attention_type="scse")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k: v for k, v in sd.items() if not k.startswith("segmentation_head")}

    tgt = m.net.state_dict()
    ok, mismatch = {}, []
    for k, v in sd.items():
        if k in tgt and tgt[k].shape == v.shape:
            ok[k] = v
        elif k in tgt:
            mismatch.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(tgt[k].shape)}")
    m.net.load_state_dict(ok, strict=False)

    body = [k for k in tgt if not k.startswith("segmentation_head")]
    missing = [k for k in body if k not in ok]
    rep = LoadReport(
        name="unet-efficientnet-b4-openearthmap",
        source=os.path.abspath(ckpt_path),
        pretrained_requested=True,
        tensors_total=len(body), tensors_loaded=len(ok),
        tensors_missing=missing, tensors_shape_mismatch=mismatch,
        head_replaced=f"OEM 9-class head dropped -> fresh {n_classes}-class head",
        notes="encoder + scSE decoder transferred from OpenEarthMap")
    if rep.fraction < min_fraction:
        raise PretrainedWeightsUnavailable(
            f"Only {rep.tensors_loaded}/{rep.tensors_total} "
            f"({100*rep.fraction:.1f}%) OpenEarthMap tensors matched; "
            f"below the {100*min_fraction:.0f}% threshold. First missing: "
            f"{missing[:5]}. Refusing to label this row 'OpenEarthMap-pretrained'.")
    return m, m.encoder, rep


# =============================================================================
# ViT (DINOv3 / DINOv2) backbone + U-Net decoder
# =============================================================================
class ViTUNet(nn.Module):
    """ViT bottleneck + trainable conv stem and U-Net decoder.

    Skips come from a stem run on the full-resolution tile, so boundary detail
    is not limited by the patch grid (which dominates commission error here).
    """

    def __init__(self, timm_name, n_classes, img_size=256, pretrained=True,
                 freeze=False):
        super().__init__()
        import re
        import timm
        self.norm = Normalize()
        patch = int(re.search(r"patch(\d+)", timm_name).group(1))
        self.vit_size = (img_size // patch) * patch
        self.grid = self.vit_size // patch
        try:
            self.vit = timm.create_model(timm_name, pretrained=pretrained,
                                         img_size=self.vit_size, num_classes=0)
        except Exception as e:
            if pretrained:
                raise PretrainedWeightsUnavailable(
                    f"Could not load pretrained weights for {timm_name}: "
                    f"{type(e).__name__}: {str(e)[:200]}") from e
            raise
        self.n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        c = self.vit.embed_dim
        self.freeze = freeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.stem1 = _conv_block(3, 32)
        self.stem2 = _conv_block(32, 64)
        self.stem3 = _conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.proj = nn.Conv2d(c, 128, 1)
        self.dec3 = _conv_block(128 + 128, 128)
        self.dec2 = _conv_block(128 + 64, 64)
        self.dec1 = _conv_block(64 + 32, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.vit.eval()
        return self

    def forward(self, x):
        xn = self.norm(x)
        s1 = self.stem1(xn)
        s2 = self.stem2(self.pool(s1))
        s3 = self.stem3(self.pool(s2))
        v = F.interpolate(xn, size=(self.vit_size, self.vit_size),
                          mode="bilinear", align_corners=False)
        if self.freeze:
            with torch.no_grad():
                tok = self.vit.forward_features(v)
        else:
            tok = self.vit.forward_features(v)
        tok = tok[:, self.n_prefix:, :]
        b, n, c = tok.shape
        f = self.proj(tok.transpose(1, 2).reshape(b, c, self.grid, self.grid).float())

        def up(t, ref):
            return F.interpolate(t, size=ref.shape[-2:], mode="bilinear",
                                 align_corners=False)

        d3 = self.dec3(torch.cat([up(f, s3), s3], 1))
        d2 = self.dec2(torch.cat([up(d3, s2), s2], 1))
        d1 = self.dec1(torch.cat([up(d2, s1), s1], 1))
        return self.head(d1)


def build_vit_unet(timm_name, n_classes, img_size, freeze, source_label):
    """Build a ViT-backboned U-Net and verify the backbone is really pretrained."""
    import timm
    m = ViTUNet(timm_name, n_classes, img_size, pretrained=True, freeze=freeze)

    # Independent check: a freshly built RANDOM model of the same spec must
    # differ from ours. Identical weights would mean pretraining silently
    # failed and we are about to mislabel the row.
    ref = timm.create_model(timm_name, pretrained=False,
                            img_size=m.vit_size, num_classes=0)
    ours = m.vit.state_dict()
    rand = ref.state_dict()
    differing = sum(1 for k in ours
                    if k in rand and ours[k].shape == rand[k].shape
                    and not torch.equal(ours[k].float(), rand[k].float()))
    if differing == 0:
        raise PretrainedWeightsUnavailable(
            f"{timm_name}: backbone is bit-identical to a random "
            f"initialisation -- pretrained weights did not load.")
    rep = LoadReport(
        name=f"vitunet-{timm_name}",
        source=f"timm {timm_name} ({source_label})",
        pretrained_requested=True,
        tensors_total=len(ours), tensors_loaded=differing,
        head_replaced=f"timm classification head removed (num_classes=0); "
                      f"fresh {n_classes}-class conv segmentation head",
        notes=(f"patch grid {m.grid}x{m.grid} @ {m.vit_size}px; "
               f"backbone {'FROZEN' if freeze else 'fine-tuned'}; "
               f"decoder and stem trained from scratch"))
    return m, (None if freeze else m.vit), rep


# =============================================================================
# SatlasPretrain aerial backbone
# =============================================================================
class SatlasUNet(nn.Module):
    """SatlasPretrain Swin-v2 aerial backbone + U-Net decoder.

    Satlas backbones return a feature pyramid; we attach the same decoder
    shape used for the other hierarchical backbones.
    """

    def __init__(self, backbone, feat_channels, n_classes):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone
        c = feat_channels
        self.d3 = _conv_block(c[3] + c[2], 256)
        self.d2 = _conv_block(256 + c[1], 128)
        self.d1 = _conv_block(128 + c[0], 64)
        self.d0 = _conv_block(64, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        feats = self.backbone(self.norm(x))
        if isinstance(feats, (list, tuple)):
            f0, f1, f2, f3 = feats[0], feats[1], feats[2], feats[3]
        else:
            raise RuntimeError("Satlas backbone did not return a feature list")

        def up(t, ref):
            return F.interpolate(t, size=ref.shape[-2:], mode="bilinear",
                                 align_corners=False)

        y = self.d3(torch.cat([up(f3, f2), f2], 1))
        y = self.d2(torch.cat([up(y, f1), f1], 1))
        y = self.d1(torch.cat([up(y, f0), f0], 1))
        y = self.d0(F.interpolate(y, size=x.shape[-2:], mode="bilinear",
                                  align_corners=False))
        return self.head(y)


def build_satlas_unet(n_classes, img_size=256, freeze=False,
                      model_id="Aerial_SwinB_SI"):
    try:
        import satlaspretrain_models
    except ImportError as e:
        raise PretrainedWeightsUnavailable(
            "satlaspretrain_models is not installed, so SatlasPretrain weights "
            "cannot be loaded. Install it (pip install satlaspretrain_models) "
            "and re-run. Not substituting a random backbone."
        ) from e
    try:
        weights_manager = satlaspretrain_models.Weights()
        backbone = weights_manager.get_pretrained_model(
            model_identifier=model_id, fpn=True)
    except Exception as e:
        raise PretrainedWeightsUnavailable(
            f"SatlasPretrain {model_id} download/load failed: "
            f"{type(e).__name__}: {str(e)[:200]}") from e

    with torch.no_grad():
        feats = backbone(torch.zeros(1, 3, img_size, img_size))
    if not isinstance(feats, (list, tuple)) or len(feats) < 4:
        raise RuntimeError(
            f"Unexpected Satlas feature output: {type(feats)}")
    chs = [f.shape[1] for f in feats[:4]]

    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
    m = SatlasUNet(backbone, chs, n_classes)
    n_bb = sum(1 for _ in backbone.state_dict())
    rep = LoadReport(
        name=f"satlas-{model_id}",
        source=f"SatlasPretrain {model_id} (single-image aerial, Swin-v2-B)",
        pretrained_requested=True,
        tensors_total=n_bb, tensors_loaded=n_bb,
        head_replaced=f"Satlas task heads dropped; fresh {n_classes}-class head",
        notes=(f"FPN feature channels {chs}; "
               f"backbone {'FROZEN' if freeze else 'fine-tuned'}"))
    return m, (None if freeze else backbone), rep


# =============================================================================
# Registry
# =============================================================================
REGISTRY = {
    "efficientnet-b4-imagenet": dict(
        kind="smp", encoder="efficientnet-b4",
        desc="U-Net, EfficientNet-B4 encoder, ImageNet-1k"),
    "efficientnet-b4-oem": dict(
        kind="oem",
        desc="U-Net, EfficientNet-B4 encoder, OpenEarthMap aerial pretraining"),
    "dinov3-vitl": dict(
        kind="vit", timm_name="vit_large_patch16_dinov3.lvd1689m",
        source_label="LVD-1689M general-purpose",
        desc="DINOv3 ViT-L/16 general-purpose + U-Net decoder"),
    "dinov3-sat-vitl": dict(
        kind="vit", timm_name="vit_large_patch16_dinov3.sat493m",
        source_label="SAT-493M satellite",
        desc="DINOv3 ViT-L/16 satellite-pretrained + U-Net decoder"),
    "satlas-aerial": dict(
        kind="satlas", model_id="Aerial_SwinB_SI",
        desc="SatlasPretrain single-image aerial Swin-v2-B + U-Net decoder"),
}


def build_registered(arch, n_classes, img_size=256, freeze=False,
                     oem_weights=None):
    """Build any registered backbone. Raises if pretrained weights are absent."""
    if arch not in REGISTRY:
        raise ValueError(f"unknown arch {arch!r}; known: {sorted(REGISTRY)}")
    spec = REGISTRY[arch]
    kind = spec["kind"]
    if kind == "smp":
        m, enc, rep = build_imagenet_unet(spec["encoder"], n_classes)
        if freeze:
            for p in enc.parameters():
                p.requires_grad = False
            enc = None
            rep.notes += "; encoder FROZEN"
        return m, enc, rep
    if kind == "oem":
        m, enc, rep = build_openearthmap_unet(n_classes, oem_weights)
        if freeze:
            for p in enc.parameters():
                p.requires_grad = False
            enc = None
            rep.notes += "; encoder FROZEN"
        return m, enc, rep
    if kind == "vit":
        return build_vit_unet(spec["timm_name"], n_classes, img_size, freeze,
                              spec["source_label"])
    if kind == "satlas":
        return build_satlas_unet(n_classes, img_size, freeze, spec["model_id"])
    raise ValueError(kind)
