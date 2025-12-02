# core/cv_ops.py
# ------------------------------------------------------------
# شمارش خودکار گرافت‌ها داخل ظرف پتری با حذف انعکاس رینگ
# روش: Gray+CLAHE+Top-hat  →  ماسک سفید/شیری (HSV-S پایین)
#       → پاک‌سازی مورفولوژی → فیلتر مساحت
#       → DT-peaks  ∪  multi-scale LoG  → NMS → مراکز نهایی
# خروجی: (count, centers[Nx2], overlay[BGR], debug_dict)
# پیش‌نیاز: pip install opencv-python numpy scikit-image scipy
# ------------------------------------------------------------

from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Dict, Any
from skimage import exposure
from skimage.feature import blob_log
from scipy import ndimage as ndi


# ------------------------- Helpers -------------------------

def _odd(x: int) -> int:
    """بازگرداندن نزدیک‌ترین عدد فرد >= x"""
    return x + (1 - x % 2)


def _nms_points(points: np.ndarray, scores: np.ndarray, min_dist: int = 12) -> np.ndarray:
    """
    Non-Maximum Suppression روی نقاط (برای حذف نزدیک‌ها).
    points: [N,2]  ,  scores: [N]
    نتیجه: ایندکس‌های نقاط نگه‌داشته‌شده
    """
    if points.size == 0:
        return np.array([], dtype=int)
    idxs = np.argsort(-scores)  # بزرگ → کوچک
    selected = []
    taken = np.zeros(len(points), dtype=bool)
    for i in idxs:
        if taken[i]:
            continue
        selected.append(i)
        d = np.linalg.norm(points - points[i], axis=1)
        taken |= (d < min_dist)
    return np.array(selected, dtype=int)


def detect_petri_mask(bgr: np.ndarray, erode_px: int = 6) -> Tuple[np.ndarray, Tuple[int, int, int] | None]:
    """
    تشخیص تقریبی دایره ظرف پتری با Hough و ساخت ماسک داخل آن.
    خروجی:
      - mask: uint8 در بازه 0/255
      - circle: (cx, cy, r) یا None اگر تشخیص نشود
    """
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 3)
    edges = cv2.Canny(gray, 40, 120)

    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(H, W) // 3,
        param1=120, param2=50,
        minRadius=int(0.35 * min(H, W)), maxRadius=int(0.55 * min(H, W))
    )

    if circles is None:
        mask = np.ones((H, W), np.uint8) * 255
        return mask, None

    c = np.round(circles[0, 0]).astype(int)
    cx, cy, r = int(c[0]), int(c[1]), int(c[2])
    Y, X = np.ogrid[:H, :W]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= (r - erode_px) ** 2
    return (mask.astype(np.uint8) * 255), (cx, cy, r)


# ------------------------- Core -------------------------

def count_grafts_auto(bgr_image: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    نسخه V2: حذف انعکاس لبه + بازیابی کم‌شماری با LoG و ادغام با DT.
    ورودی:
        bgr_image: تصویر BGR (OpenCV)
    خروجی:
        count  : تعداد گرافت‌ها
        centers: Nx2 (x,y) مختصات مرکز هر گرافت
        overlay: تصویر BGR با نقطه‌گذاری و دایره ظرف
        debug  : دیکشنری پارامترها و آمار برای دیباگ
    """
    img = bgr_image.copy()
    H, W = img.shape[:2]
    m = min(H, W)

    # 0) ماسک ظرف + "نوارِ ممنوعه" نزدیک رینگ برای حذف انعکاس
    dish_mask, circle = detect_petri_mask(img, erode_px=max(4, m // 160))
    if circle is None:
        circle = (W // 2, H // 2, int(0.48 * m))
    cx, cy, r = circle
    rim_px = max(6, int(0.06 * m))  # ~6% قطر تصویر
    Y, X = np.ogrid[:H, :W]
    inside = (X - cx) ** 2 + (Y - cy) ** 2 <= (r - 1) ** 2
    rim_forbid = (X - cx) ** 2 + (Y - cy) ** 2 >= (r - rim_px) ** 2
    valid_mask = (inside & (~rim_forbid) & (dish_mask > 0))

    # 1) Gray + CLAHE + Top-hat  (برای حذف پس‌زمینه نرم)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    k_tophat = max(7, _odd(int(round(0.012 * m))))
    k_t = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_tophat, k_tophat))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k_t)

    # 2) ماسک «سفید/شیری» (S کانال پایین) + آستانه‌گذاری تطبیقی روی توپهت
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    _, s_bin = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    block = _odd(int(0.045 * m))  # اندازه پنجره adaptive
    block = max(21, block)
    th_adp = cv2.adaptiveThreshold(
        tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C=-5
    )

    # ترکیب ماسک‌ها و محدودسازی به ناحیه معتبر
    binmask = cv2.bitwise_and(th_adp, s_bin)
    binmask = cv2.bitwise_and(binmask, valid_mask.astype(np.uint8) * 255)

    k_sz = max(3, _odd(int(0.006 * m)))
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
    binmask = cv2.morphologyEx(binmask, cv2.MORPH_OPEN, k_small, iterations=1)
    binmask = cv2.morphologyEx(binmask, cv2.MORPH_CLOSE, k_small, iterations=1)

    # 3) حذف لکه‌های غیرواقعی با فیلتر مساحت
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binmask, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    a_min = int(0.00004 * H * W)  # این‌ها را در صورت نیاز فاین‌تیون کن
    a_max = int(0.0020 * H * W)
    keep_idx = np.where((areas >= a_min) & (areas <= a_max))[0] + 1
    comp_mask = np.isin(labels, keep_idx).astype(np.uint8) * 255

    # 4A) قله‌های Distance Transform
    dt = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 3)
    dt_vals = dt[comp_mask > 0]
    dt_thr = float(np.percentile(dt_vals, 70)) if dt_vals.size else 0.0
    dt_dil = cv2.dilate(dt, np.ones((3, 3), np.uint8))
    peaks = (dt == dt_dil) & (dt >= dt_thr) & (valid_mask > 0)
    ys, xs = np.where(peaks)
    pts_A = np.stack([xs, ys], axis=1) if xs.size else np.zeros((0, 2), dtype=int)
    sc_A = dt[ys, xs] if xs.size else np.array([], dtype=float)

    # 4B) LoG چندمقیاسی برای بازیابی گرافت‌های کم‌کنتراست
    th_norm = exposure.rescale_intensity(tophat, in_range='image', out_range=(0.0, 1.0))
    min_sigma = 0.006 * m
    max_sigma = 0.020 * m
    blobs = blob_log(
        th_norm, min_sigma=min_sigma, max_sigma=max_sigma,
        num_sigma=8, threshold=0.02, overlap=0.5
    )
    if blobs.size:
        xs_B = blobs[:, 1].astype(int)
        ys_B = blobs[:, 0].astype(int)
        valid = valid_mask[ys_B, xs_B] > 0
        xs_B, ys_B = xs_B[valid], ys_B[valid]
        pts_B = np.stack([xs_B, ys_B], axis=1)
        sc_B = th_norm[ys_B, xs_B]
    else:
        pts_B = np.zeros((0, 2), dtype=int)
        sc_B = np.array([], dtype=float)

    # 4C) اتحاد دو مرحله و NMS
    if pts_A.size and pts_B.size:
        pts = np.vstack([pts_A, pts_B])
        scs = np.concatenate([sc_A, sc_B])
    elif pts_A.size:
        pts, scs = pts_A, sc_A
    else:
        pts, scs = pts_B, sc_B

    min_dist = max(10, int(0.012 * m))
    keep = _nms_points(pts.astype(np.float32), scs.astype(np.float32), min_dist=min_dist)
    centers = pts[keep].astype(int)
    count = int(len(centers))

    # 5) ترسیم خروجی
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), r, (0, 255, 0), 2)            # رینگ ظرف
    cv2.circle(overlay, (cx, cy), r - rim_px, (0, 255, 0), 1)   # نوار ممنوعه
    for (x, y) in centers:
        cv2.circle(overlay, (int(x), int(y)), 3, (0, 0, 255), -1)

    debug = {
        "dish_circle": (int(cx), int(cy), int(r)),
        "rim_px": int(rim_px),
        "k_tophat": int(k_tophat),
        "block_adapt": int(block),
        "area_min": int(a_min),
        "area_max": int(a_max),
        "dt_thr": float(dt_thr),
        "n_stageA": int(pts_A.shape[0]),
        "n_stageB": int(pts_B.shape[0]),
        "n_after_nms": int(count),
        "min_dist": int(min_dist),
        "method": "DT+LoG+RimSuppression"
    }
    return count, centers, overlay, debug
