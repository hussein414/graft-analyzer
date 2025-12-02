# core/pipeline.py
import os
import cv2
import base64
import uuid
import numpy as np
from core.graft_counter import count_grafts  # استفاده از منطق جدید

ASSETS_DIR = os.getenv("GA_ASSETS_DIR", "assets/overlays")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _overlay_to_b64(img_bgr: np.ndarray, quality: int = 90) -> str:
    # ابتدا JPEG، اگر نشد PNG
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _save_overlay(img_bgr: np.ndarray) -> str:
    _ensure_dir(ASSETS_DIR)
    name = f"overlay_{uuid.uuid4().hex[:10]}.jpg"
    path = os.path.join(ASSETS_DIR, name)
    cv2.imwrite(path, img_bgr)
    return path

def analyze_bgr(img_bgr: np.ndarray):
    res = count_grafts(img_bgr, preset="clientdemo")

    # خروجی‌های ماژول جدید RGB هستند؛ برای ذخیره باید به BGR برگردند
    overlay_bgr = cv2.cvtColor(res["overlay_clean"], cv2.COLOR_RGB2BGR)
    debug_bgr   = cv2.cvtColor(res["overlay_debug"], cv2.COLOR_RGB2BGR)

    centers = res["points"].tolist()

    return {
        "count": int(res["count"]),
        "centers": [(int(x), int(y)) for (x, y) in centers],
        "boxes": [],
        "chosen": "clientdemo",
        "overlay_bgr": overlay_bgr,
        "debug_bgr": debug_bgr,
    }

def analyze_bytes(data: bytes):
    """بایت‌های تصویر → دیکد → تحلیل → Base64 + ذخیره فایل خروجی"""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes")

    res = analyze_bgr(img)

    overlay_b64 = _overlay_to_b64(res["overlay_bgr"], quality=90)
    debug_b64   = _overlay_to_b64(res["debug_bgr"], quality=90)
    overlay_path = _save_overlay(res["overlay_bgr"])

    return {
        "count": res["count"],
        "centers": res["centers"],
        "boxes": res["boxes"],
        "chosen": res["chosen"],
        "overlay_b64": overlay_b64,
        "overlay_debug_b64": debug_b64,  # برای تب دیباگ در UI
        "overlay_path": overlay_path,
    }
