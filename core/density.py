# core/density.py
from typing import Dict, List, Tuple, Optional
import os
import cv2
import numpy as np
import torch
from core.dish import detect_petri_mask
from core.models.csrnet_lite import CSRNetLite

# --- Utils -------------------------------------------------------

def _prep_rgb(img_bgr: np.ndarray, long_side: int) -> Tuple[torch.Tensor, float]:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    s = long_side / max(H, W)
    if s != 1.0:
        rgb = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_CUBIC)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return t, s

def _apply_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.max() == 0:
        return bgr
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask
    return cv2.bitwise_and(bgr, m3)

def _draw_annotations(img: np.ndarray, centers: List[Tuple[int,int]], boxes: List[Tuple[int,int,int,int]]) -> np.ndarray:
    out = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
    for (cx, cy) in centers:
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    return out

# --- Model loading ------------------------------------------------

def load_density_model(ckpt_path: Optional[str] = None, device: str = "cpu") -> CSRNetLite:
    model = CSRNetLite().to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # حذف پیشوند 'module.' در صورت ذخیره با DDP
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        if isinstance(sd, dict):
            model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# --- API ----------------------------------------------------------

@torch.inference_mode()
def analyze_bgr_density(
    img_bgr: np.ndarray,
    cfg: Optional[object] = None,              # اختیاری؛ اگر داشتی
    weights_path: Optional[str] = None,        # اگر cfg نیست، از این استفاده می‌کنیم
    long_side: int = 1280,                     # اگر cfg نیست، از این استفاده می‌کنیم
) -> Dict:
    """
    مسیر Density: فقط داخل ظرف شمارش می‌کند.
    خروجی: dict سازگار با بقیه‌ی pipeline
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) ماسک ظرف و اعمال آن
    dish_mask, _ = detect_petri_mask(img_bgr, erode_px=max(4, min(img_bgr.shape[:2]) // 160))
    img_in = _apply_mask(img_bgr, dish_mask)

    # 2) ورودی مدل
    if cfg is not None and hasattr(cfg, "density_long_side"):
        long_side = int(getattr(cfg, "density_long_side"))
    tin, scale = _prep_rgb(img_in, long_side)
    tin = tin.to(device)

    # 3) مدل و وزن‌ها
    if cfg is not None and hasattr(cfg, "density_ckpt"):
        weights_path = getattr(cfg, "density_ckpt")
    model = load_density_model(weights_path, device=device)

    # 4) پیش‌بینی چگالی و شمارش
    den = model(tin)[0, 0].detach().cpu().numpy()  # H' x W'
    den_resized = cv2.resize(den, (img_in.shape[1], img_in.shape[0]), interpolation=cv2.INTER_CUBIC)
    total_count = float(den_resized.sum())

    # 5) استخراج مراکز تقریبی برای نمایش (از روی نقشه چگالی)
    den_norm = den_resized / (den_resized.max() + 1e-6)
    thr = 0.35  # قابل تیون؛ می‌تونی تطبیقی کنی با np.mean(den_norm)
    mask = (den_norm > thr).astype(np.uint8) * 255
    num, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)

    centers: List[Tuple[int, int]] = []
    boxes: List[Tuple[int, int, int, int]] = []
    H, W = img_in.shape[:2]
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        cx, cy = int(cent[i][0]), int(cent[i][1])
        # کلیپ مرزی
        cx = 0 if cx < 0 else (W - 1 if cx >= W else cx)
        cy = 0 if cy < 0 else (H - 1 if cy >= H else cy)
        centers.append((cx, cy))
        boxes.append((int(x), int(y), int(bw), int(bh)))

    overlay = _draw_annotations(img_in, centers[:300], boxes[:300])

    return {
        "count": int(round(total_count)),       # جمع چگالی
        "centers": centers[:300],               # محدود برای نمایش
        "boxes": boxes[:300],
        "chosen": "density",
        "overlay_bgr": overlay,
        "mask": mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
    }
