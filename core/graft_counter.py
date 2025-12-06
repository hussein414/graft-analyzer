# core/graft_counter.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class Preset:
    spec_quantile: float
    log_sigmas: Tuple[float, ...]
    adapt_block: int
    adapt_C: int
    watershed_peak_ratio: float
    elong_max: float
    circ_min: float
    th_min_px: float | None
    th_max_px: float | None
    tophat_kernel: int


PRESETS = {
    "clientdemo": Preset(
        spec_quantile=99.7,
        log_sigmas=(2.5, 3.5, 5.0, 7.0),
        adapt_block=51,
        adapt_C=-5,
        watershed_peak_ratio=0.65,
        elong_max=4.5,
        circ_min=0.10,
        th_min_px=4.0,
        th_max_px=35.0,
        tophat_kernel=31,
    ),
    "qc": Preset(
        spec_quantile=99.4,
        log_sigmas=(2.0, 3.0, 4.0, 5.0),
        adapt_block=39,
        adapt_C=-2,
        watershed_peak_ratio=0.55,
        elong_max=3.0,
        circ_min=0.15,
        th_min_px=7.0,
        th_max_px=24.0,
        tophat_kernel=25,
    ),
}


def _detect_petri_dish_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Detect the circular petri dish region instead of looking for white paper.
    Returns a mask of the dish interior.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 3)

    # Detect circular dish using Hough circles
    s = min(h, w)
    rmin = int(0.30 * s / 2)
    rmax = int(0.98 * s / 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=s // 2,
        param1=100,
        param2=30,
        minRadius=rmin,
        maxRadius=rmax,
    )

    mask = np.zeros((h, w), np.uint8)

    if circles is not None:
        # Use the largest/most centered circle
        circles = np.uint16(np.around(circles[0]))
        center = np.array([w / 2, h / 2], dtype=np.float32)

        def score(c):
            r = float(c[2])
            cxy = c[:2].astype(np.float32)
            d = float(np.linalg.norm(cxy - center))
            return r - 0.3 * d

        best = max(circles, key=score)
        cx, cy, r = int(best[0]), int(best[1]), int(best[2])

        # Create mask with slight erosion from rim
        rim_erode = max(10, int(0.03 * r))
        cv2.circle(mask, (cx, cy), r - rim_erode, 255, -1)
    else:
        # Fallback: use entire image
        mask[:] = 255

    return mask


def _specular_keep_mask(bgr: np.ndarray, q: float) -> np.ndarray:
    """Remove specular highlights (bright white spots)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    vth = max(245, int(np.percentile(V, q)))
    spec = ((S < 60) & (V >= vth)).astype(np.uint8) * 255
    spec = cv2.dilate(spec, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)
    return cv2.bitwise_not(spec)


def _log_multiscale_response(gray: np.ndarray, sigmas: Tuple[float, ...]) -> np.ndarray:
    """Multi-scale Laplacian of Gaussian response."""
    resps = []
    for s in sigmas:
        k = int(6 * s + 1) | 1
        g = cv2.GaussianBlur(gray, (k, k), s)
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        resps.append(-lap)
    logmax = np.max(resps, axis=0)
    logmax = cv2.normalize(logmax, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return logmax


def _watershed_split(bin_mask: np.ndarray, peak_ratio: float) -> np.ndarray:
    """Split touching grafts using watershed."""
    dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
    th = float(peak_ratio) * dist.max()
    seeds = (dist > th).astype(np.uint8) * 255
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_DILATE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    _, markers = cv2.connectedComponents(seeds)
    markers = markers.astype(np.int32)
    ws_img = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(ws_img, markers)
    labels_ws = (markers > 1).astype(np.uint8) * 255
    labels_ws = cv2.bitwise_and(labels_ws, bin_mask)
    return labels_ws


def _measure_and_filter(lbl_img: np.ndarray,
                        stats: np.ndarray,
                        centroids: np.ndarray,
                        th_min: float, th_max: float,
                        elong_max: float, circ_min: float
                        ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Filter detected regions based on shape and size criteria."""
    accepted, rejected = [], []
    num = int(lbl_img.max())

    for i in range(1, num + 1):
        mask_i = (lbl_img == i).astype(np.uint8) * 255
        cs, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        cnt = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True) + 1e-6
        circ = 4.0 * np.pi * area / (peri * peri)

        if len(cnt) >= 5:
            (cx, cy), (MA, ma), _ = cv2.fitEllipse(cnt)
            minor = min(MA, ma)
            major = max(MA, ma)
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            minor = min(w, h)
            major = max(w, h)
            cx, cy = x + w / 2, y + h / 2

        elong = major / (minor + 1e-6)
        A_min = max(15, int((th_min ** 2) * 0.3))
        A_max = int((th_max ** 2) * 50)

        c = centroids[i] if i < len(centroids) else (cx, cy)

        if (minor >= th_min) and (minor <= th_max) and \
                (area >= A_min) and (area <= A_max) and \
                (elong <= elong_max) and (circ >= circ_min):
            accepted.append((float(c[0]), float(c[1])))
        else:
            rejected.append((float(c[0]), float(c[1])))

    return accepted, rejected


def count_grafts(image_bgr: np.ndarray, preset: str = "clientdemo") -> Dict:
    """
    Main graft counting pipeline with improved petri dish detection.
    """
    P = PRESETS[preset]

    # Use petri dish detection instead of paper mask
    dish_mask = _detect_petri_dish_mask(image_bgr)
    keep = _specular_keep_mask(image_bgr, P.spec_quantile)

    # Enhanced preprocessing
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Larger top-hat kernel to remove uneven background
    ksize = P.tophat_kernel | 1
    toph = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    )

    # Multi-scale LoG response
    logmax = _log_multiscale_response(toph, P.log_sigmas)

    # Adaptive threshold
    cand = cv2.adaptiveThreshold(logmax, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 P.adapt_block, P.adapt_C)

    # Apply masks
    cand = cv2.bitwise_and(cand, dish_mask)
    cand = cv2.bitwise_and(cand, keep)

    # Morphological cleanup
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 2)

    # Watershed splitting
    labels_ws = _watershed_split(cand, P.watershed_peak_ratio)

    # Connected components
    num, lbl, stats, cent = cv2.connectedComponentsWithStats(labels_ws, 8)

    # Adaptive size thresholds
    th_min, th_max = P.th_min_px, P.th_max_px
    if th_min is None or th_max is None:
        minors = []
        for i in range(1, num):
            mask_i = (lbl == i).astype(np.uint8) * 255
            cs, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cs:
                continue
            cnt = max(cs, key=cv2.contourArea)
            if len(cnt) >= 5:
                (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
                minor = min(MA, ma)
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                minor = min(w, h)
            minors.append(minor)

        if len(minors) > 0:
            minors = np.array(minors)
            lo = float(np.percentile(minors, 5)) * 0.8
            hi = float(np.percentile(minors, 98)) * 1.3
            th_min = max(4.0, lo)
            th_max = max(35.0, hi)
        else:
            th_min, th_max = 4.0, 35.0

    # Filter and classify
    accepted_pts, rejected_pts = _measure_and_filter(
        lbl, stats, cent,
        th_min, th_max,
        P.elong_max, P.circ_min
    )

    # Visualization
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay_debug = rgb.copy()
    overlay_clean = rgb.copy()

    # Draw dish boundary
    contours, _ = cv2.findContours(dish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_debug, contours, -1, (0, 255, 0), 3)
    cv2.drawContours(overlay_clean, contours, -1, (0, 255, 0), 3)

    # Draw accepted grafts (red)
    for (x, y) in accepted_pts:
        cv2.circle(overlay_debug, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.circle(overlay_clean, (int(x), int(y)), 5, (255, 0, 0), -1)

    # Draw rejected (yellow, debug only)
    for (x, y) in rejected_pts:
        cv2.circle(overlay_debug, (int(x), int(y)), 3, (255, 255, 0), -1)

    return {
        "count": len(accepted_pts),
        "points": np.array(accepted_pts, dtype=np.float32),
        "rejected_points": np.array(rejected_pts, dtype=np.float32),
        "overlay_clean": overlay_clean,
        "overlay_debug": overlay_debug,
        "dish_mask": dish_mask,
        "params": {
            "preset": preset,
            "thickness_px": {"min": float(th_min), "max": float(th_max)},
            "elong_max": float(P.elong_max),
            "circ_min": float(P.circ_min),
            "spec_quantile": float(P.spec_quantile),
            "tophat_kernel": int(P.tophat_kernel),
        }
    }