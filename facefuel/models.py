"""
Shared model components.

- One DINOv2 ViT-S/14 instance shared by all three modalities (was loaded 3×).
- Region embedding that exactly reproduces the v4 feature-extraction scripts
  (pipeline/05_extract_features.py): resize → fixed-fraction regions → ImageNet
  normalisation → raw (un-normalised) CLS token. The severity MLPs were trained
  on exactly this, so inference must not deviate from it.
- Severity-MLP loading for the v4 checkpoint format
  {"state_dict","classes","active_idx","inactive_idx","in_dim","n_active"},
  with the active-head → full-class-list remapping done here, once, for everyone.
"""
import logging
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn

from . import paths

log = logging.getLogger("facefuel")

_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# (row_start, row_end, col_start, col_end) as fractions of the resized image.
FACE_REGIONS = {
    "forehead":    (0.00, 0.25, 0.10, 0.90),
    "left_eye":    (0.15, 0.40, 0.10, 0.55),
    "right_eye":   (0.15, 0.40, 0.45, 0.90),
    "nose":        (0.30, 0.55, 0.30, 0.70),
    "mouth":       (0.55, 0.80, 0.25, 0.75),
    "left_cheek":  (0.30, 0.65, 0.05, 0.40),
    "right_cheek": (0.30, 0.65, 0.60, 0.95),
    "chin":        (0.75, 1.00, 0.25, 0.75),
}
TONGUE_REGIONS = {
    "anterior":  (0.05, 0.45, 0.10, 0.90),
    "middle":    (0.35, 0.70, 0.15, 0.85),
    "posterior": (0.60, 0.95, 0.20, 0.80),
}
EYE_REGIONS = {
    "sclera_left":  (0.15, 0.85, 0.00, 0.45),
    "sclera_right": (0.15, 0.85, 0.55, 1.00),
    "periorbital":  (0.00, 0.35, 0.10, 0.90),
}
# (width, height) each modality's image is resized to before region slicing
REGION_SPECS = {
    "face":   (FACE_REGIONS,   (256, 256)),
    "tongue": (TONGUE_REGIONS, (256, 256)),
    "eye":    (EYE_REGIONS,    (256, 128)),
}

MC_PASSES = 20
MC_SEED = 0


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── DINOv2 ───────────────────────────────────────────────────────────────────
_dino = {}


def get_dino(device: str):
    if device not in _dino:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(paths.DINOV2_MODEL).to(device)
        m.eval()
        _dino[device] = m
        log.info("DINOv2 loaded (%s) on %s", paths.DINOV2_MODEL, device)
    return _dino[device]


def _prep_region(region_bgr: np.ndarray) -> torch.Tensor:
    # Matches torchvision ToPILImage → Resize((224,224)) → ToTensor → Normalize
    # used in training (PIL bilinear resize on the RGB region).
    from PIL import Image
    rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (t - _IMNET_MEAN) / _IMNET_STD


@torch.no_grad()
def embed_regions(img_bgr: np.ndarray, modality: str, device: str) -> torch.Tensor:
    """Concatenated DINOv2 CLS embeddings for the modality's regions (len(regions)*384)."""
    regions, size = REGION_SPECS[modality]
    img = cv2.resize(img_bgr, size)
    h, w = img.shape[:2]
    batch, empty = [], []
    for i, (r1, r2, c1, c2) in enumerate(regions.values()):
        reg = img[int(h * r1):int(h * r2), int(w * c1):int(w * c2)]
        if reg.size == 0:
            empty.append(i)
            batch.append(torch.zeros(3, 224, 224))
        else:
            batch.append(_prep_region(reg))
    x = torch.stack(batch).to(device)
    cls = get_dino(device)(pixel_values=x).last_hidden_state[:, 0]   # (R, 384)
    for i in empty:
        cls[i] = 0.0
    return cls.reshape(-1).float()


# ── Severity MLP (v4 architecture, identical to pipeline/06_train_severity_mlp.py) ──
class SeverityMLP(nn.Module):
    def __init__(self, in_dim: int, n_cls: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden * 2), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(hidden, 128),        nn.GELU(), nn.Dropout(0.20),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.20), nn.Linear(64, 1))
            for _ in range(n_cls)
        ])
        self.log_vars = nn.Parameter(torch.zeros(n_cls))

    def forward(self, x):
        h = self.backbone(x)
        return torch.cat([head(h) for head in self.heads], dim=-1), self.log_vars


class SeverityModel:
    """A loaded v4 severity MLP plus the mapping from its heads to the full class list."""

    def __init__(self, mlp, classes, active_idx, device):
        self.mlp = mlp
        self.classes = list(classes)
        self.active_idx = list(active_idx)
        self.inactive = {self.classes[i] for i in range(len(self.classes))
                         if i not in self.active_idx}
        self.device = device

    @torch.no_grad()
    def predict(self, feat: torch.Tensor, passes: int = MC_PASSES):
        """
        Monte-Carlo-dropout severity over the FULL class list.
        Returns (severity, uncertainty) arrays of len(classes). Inactive classes
        (no head) get severity 0 and uncertainty 1 — "no information", never a guess.
        """
        x = feat.unsqueeze(0).to(self.device).float()
        devices = [torch.device(self.device)] if str(self.device).startswith("cuda") else []
        # fixed seed → the same photo always gives the same result
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(MC_SEED)
            self.mlp.train()                   # enable dropout for MC sampling
            try:
                draws = torch.stack([torch.sigmoid(self.mlp(x)[0])[0] for _ in range(passes)])
            finally:
                self.mlp.eval()
        mean, std = draws.mean(0).cpu().numpy(), draws.std(0).cpu().numpy()
        sev = np.zeros(len(self.classes), dtype=np.float32)
        unc = np.ones(len(self.classes), dtype=np.float32)
        for head, cls_i in enumerate(self.active_idx):
            sev[cls_i], unc[cls_i] = mean[head], std[head]
        return sev, unc


def load_severity_mlp(key: str, expected_classes: list, expected_inactive: set,
                      device: str) -> SeverityModel:
    path = paths.resolve(key)
    ckpt = torch.load(str(path), map_location=device, weights_only=True)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:          # v4 format
        classes    = list(ckpt.get("classes", expected_classes))
        active_idx = list(ckpt.get("active_idx", range(len(classes))))
        in_dim     = int(ckpt["in_dim"])
        state      = ckpt["state_dict"]
    else:                                                          # raw state_dict (v3)
        raise ValueError(
            f"{path} is a pre-v4 checkpoint (raw state_dict). v4 inference requires the "
            "dict format written by pipeline/06_train_severity_mlp.py.")

    if classes != list(expected_classes):
        raise ValueError(f"{path}: checkpoint classes {classes} != schema {expected_classes}")

    mlp = SeverityMLP(in_dim, len(active_idx)).to(device)
    mlp.load_state_dict(state)
    mlp.eval()
    model = SeverityModel(mlp, classes, active_idx, device)
    if model.inactive != set(expected_inactive):
        warnings.warn(f"{key}: checkpoint inactive classes {sorted(model.inactive)} differ "
                      f"from schema {sorted(expected_inactive)} — update facefuel/schema.py")
    log.info("%s loaded: %d/%d active heads", key, len(active_idx), len(classes))
    return model


# ── YOLO ─────────────────────────────────────────────────────────────────────
def load_yolo(key: str, expected_classes: list):
    from ultralytics import YOLO
    path = paths.resolve(key)
    model = YOLO(str(path))
    names = [model.names[i] for i in range(len(model.names))]
    if names != list(expected_classes):
        raise ValueError(f"{path}: YOLO classes {names} != schema {expected_classes}")
    return model


def yolo_boxes(model, img_bgr, conf: float, device: str):
    """[(class_id, confidence, (x1,y1,x2,y2)), ...]"""
    res = model.predict(img_bgr, conf=conf, verbose=False, device=device)
    if not res or res[0].boxes is None:
        return []
    b = res[0].boxes
    return list(zip(b.cls.cpu().int().tolist(), b.conf.cpu().tolist(),
                    [tuple(map(float, xy)) for xy in b.xyxy.cpu().tolist()]))


def count_to_severity(n: int) -> float:
    if n == 0:  return 0.0
    if n == 1:  return 0.35
    if n <= 3:  return 0.50
    if n <= 6:  return 0.65
    if n <= 10: return 0.78
    if n <= 20: return 0.88
    return 1.0


def level(sev: float) -> str:
    return "high" if sev > 0.60 else "moderate" if sev > 0.35 else "mild"
