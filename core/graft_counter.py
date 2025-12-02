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

PRESETS = {
    "clientdemo": Preset(
        spec_quantile=99.6,
        log_sigmas=(2.0, 3.0, 4.5, 6.0),
        adapt_block=41,
        adapt_C=-3,
        watershed_peak_ratio=0.45,
        elong_max=6.0,
        circ_min=0.08,
        th_min_px=6.0,
        th_max_px=30.0,
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
    ),
}

def _build_paper_mask(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    neutral = (np.abs(A.astype(np.int16)-128) <= 14) & \
              (np.abs(B.astype(np.int16)-128) <= 14) & \
              (L > 140)
    mask = (neutral.astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 2000:
            out[labels == i] = 255
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1)
    return out

def _specular_keep_mask(bgr: np.ndarray, q: float) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    vth = max(240, int(np.percentile(V, q)))
    spec = ((S < 80) & (V >= vth)).astype(np.uint8) * 255
    spec = cv2.dilate(spec, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    return cv2.bitwise_not(spec)

def _log_multiscale_response(gray: np.ndarray, sigmas: Tuple[float, ...]) -> np.ndarray:
    resps = []
    for s in sigmas:
        k = int(6*s + 1) | 1
        g = cv2.GaussianBlur(gray, (k, k), s)
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        resps.append(-lap)
    logmax = np.max(resps, axis=0)
    logmax = cv2.normalize(logmax, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return logmax

def _watershed_split(bin_mask: np.ndarray, peak_ratio: float) -> np.ndarray:
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

        elong = major / (minor + 1e-6)
        A_min = max(30, int((th_min ** 2) * 0.4))
        A_max = int((th_max ** 2) * 30)
        c = centroids[i] if i < len(centroids) else (cx, cy)

        if (minor >= th_min) and (minor <= th_max) and \
           (area >= A_min) and (area <= A_max) and \
           (elong <= elong_max) and (circ >= circ_min):
            accepted.append((float(c[0]), float(c[1])))
        else:
            rejected.append((float(c[0]), float(c[1])))

    return accepted, rejected

def count_grafts(image_bgr: np.ndarray, preset: str = "clientdemo") -> Dict:
    P = PRESETS[preset]

    paper_mask = _build_paper_mask(image_bgr)
    keep = _specular_keep_mask(image_bgr, P.spec_quantile)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    toph = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    logmax = _log_multiscale_response(toph, P.log_sigmas)

    cand = cv2.adaptiveThreshold(logmax, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 P.adapt_block, P.adapt_C)
    cand = cv2.bitwise_and(cand, paper_mask)
    cand = cv2.bitwise_and(cand, keep)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)

    labels_ws = _watershed_split(cand, P.watershed_peak_ratio)

    num, lbl, stats, cent = cv2.connectedComponentsWithStats(labels_ws, 8)

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
            lo = float(np.percentile(minors, 10)) * 0.9
            hi = float(np.percentile(minors, 97)) * 1.25
            th_min = max(6.0, lo)
            th_max = max(30.0, hi)
        else:
            th_min, th_max = 6.0, 30.0

    accepted_pts, rejected_pts = _measure_and_filter(
        lbl, stats, cent,
        th_min, th_max,
        P.elong_max, P.circ_min
    )

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay_debug = rgb.copy()
    overlay_clean = rgb.copy()
    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_debug, contours, -1, (0, 255, 0), 3)
    cv2.drawContours(overlay_clean, contours, -1, (0, 255, 0), 3)

    for (x, y) in accepted_pts:
        cv2.circle(overlay_debug, (int(x), int(y)), 4, (255, 0, 0), -1)
        cv2.circle(overlay_clean, (int(x), int(y)), 4, (255, 0, 0), -1)
    for (x, y) in rejected_pts:
        cv2.circle(overlay_debug, (int(x), int(y)), 3, (255, 255, 0), -1)

    return {
        "count": len(accepted_pts),
        "points": np.array(accepted_pts, dtype=np.float32),
        "rejected_points": np.array(rejected_pts, dtype=np.float32),
        "overlay_clean": overlay_clean,
        "overlay_debug": overlay_debug,
        "paper_mask": paper_mask,
        "params": {
            "preset": preset,
            "thickness_px": {"min": float(th_min), "max": float(th_max)},
            "elong_max": float(P.elong_max),
            "circ_min": float(P.circ_min),
            "spec_quantile": float(P.spec_quantile),
        }
    }
