# core/pipeline.py
import os
import cv2
import base64
import uuid
import numpy as np
from core.graft_counter import count_grafts

# تلاش برای import کردن YOLO
try:
    from core.yolo_detector import analyze_bgr_yolo, YOLO_AVAILABLE

    HAS_YOLO = True
except Exception as e:
    print(f"YOLO موجود نیست: {e}")
    HAS_YOLO = False
    YOLO_AVAILABLE = False

ASSETS_DIR = os.getenv("GA_ASSETS_DIR", "assets/overlays")
USE_YOLO = os.getenv("USE_YOLO", "true").lower() == "true"
YOLO_MODEL = os.getenv("YOLO_MODEL", "weights/yolo_graft/run1/weights/best.pt")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _overlay_to_b64(img_bgr: np.ndarray, quality: int = 90) -> str:
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
    """
    تحلیل تصویر - اول YOLO رو امتحان می‌کنه، اگر نبود CV
    """
    # اگر YOLO آموزش داده شده باشه، ازش استفاده کن
    if USE_YOLO and HAS_YOLO and YOLO_AVAILABLE and os.path.exists(YOLO_MODEL):
        try:
            print(f"🤖 استفاده از مدل YOLO: {YOLO_MODEL}")
            res = analyze_bgr_yolo(img_bgr, YOLO_MODEL)
            return res
        except Exception as e:
            print(f"⚠️ YOLO خطا داد، استفاده از CV: {e}")

    # اگر YOLO نبود، از روش CV استفاده کن
    print("🔧 استفاده از روش CV (کلاسیک)")
    res = count_grafts(img_bgr, preset="clientdemo")
    overlay_bgr = cv2.cvtColor(res["overlay_clean"], cv2.COLOR_RGB2BGR)
    debug_bgr = cv2.cvtColor(res["overlay_debug"], cv2.COLOR_RGB2BGR)
    centers = res["points"].tolist()

    return {
        "count": int(res["count"]),
        "centers": [(int(x), int(y)) for (x, y) in centers],
        "boxes": [],
        "chosen": res["params"]["preset"],
        "overlay_bgr": overlay_bgr,
        "debug_bgr": debug_bgr,
    }


def analyze_bytes(data: bytes):
    """بایت‌های تصویر → تحلیل → خروجی"""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("تصویر نامعتبر است")

    res = analyze_bgr(img)
    overlay_b64 = _overlay_to_b64(res["overlay_bgr"], quality=90)
    debug_b64 = _overlay_to_b64(res.get("debug_bgr", res["overlay_bgr"]), quality=90)
    overlay_path = _save_overlay(res["overlay_bgr"])

    return {
        "count": res["count"],
        "centers": res["centers"],
        "boxes": res["boxes"],
        "chosen": res["chosen"],
        "overlay_b64": overlay_b64,
        "overlay_debug_b64": debug_b64,
        "overlay_path": overlay_path,
    }