# core/dish.py
import cv2
import numpy as np

def _odd(n: int) -> int:
    n = int(max(1, n))
    return n if n % 2 == 1 else n + 1

def detect_petri_mask(bgr: np.ndarray, erode_px: int = 8):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 3)

    s = min(h, w)
    rmin = max(20, int(0.25 * s / 2))
    rmax = max(rmin + 5, int(0.98 * s / 2))

    center = np.array([w / 2, h / 2], dtype=np.float32)
    best = None

    def run_hough(param2: int):
        return cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=s // 2,
            param1=120,
            param2=param2,
            minRadius=rmin,
            maxRadius=rmax,
        )

    circles = run_hough(param2=40)

    if circles is None:
        circles = run_hough(param2=30)

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))

        def score(c):
            r = float(c[2])
            cxy = c[:2].astype(np.float32)
            d = float(np.linalg.norm(cxy - center))
            # بزرگ‌تر + نزدیک‌تر به مرکز
            return (r) - 0.6 * d

        best = max(circles, key=score)

    if best is None:
        edges = cv2.Canny(gray, 60, 180)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(cnt)
            best = np.array([int(x), int(y), int(r)], dtype=np.int32)

    mask = np.zeros((h, w), np.uint8)
    circle = None
    if best is not None:
        cx, cy, r = int(best[0]), int(best[1]), int(best[2])
        cx = 0 if cx < 0 else (w - 1 if cx >= w else cx)
        cy = 0 if cy < 0 else (h - 1 if cy >= h else cy)
        r = int(max(5, min(r, int(0.99 * s / 2))))

        cv2.circle(mask, (cx, cy), r, 255, -1)

        erode_px = int(max(0, min(erode_px, max(2, s // 120))))
        if erode_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(erode_px * 2 + 1), _odd(erode_px * 2 + 1)))
            mask = cv2.erode(mask, k, iterations=1)

        circle = (cx, cy, r)

    return mask, circle
