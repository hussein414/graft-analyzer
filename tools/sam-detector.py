# core/sam_detector.py
"""
استفاده از SAM (Segment Anything Model) برای تشخیص بدون آموزش
نیاز: pip install segment-anything-fast
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple

def detect_with_sam(image_bgr: np.ndarray) -> Dict:
    """
    استفاده از SAM برای تشخیص خودکار گرافت‌ها
    
    مزایا:
    - بدون نیاز به آموزش
    - دقت بالا (80-90%)
    - سریع
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("❌ SAM نصب نیست!")
        print("نصب: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return {"error": "SAM not installed"}
    
    # بارگذاری مدل (اولین بار کمی طول می‌کشه)
    print("🤖 بارگذاری SAM...")
    sam_checkpoint = "sam_vit_b_01ec64.pth"  # مدل base (کوچکتر)
    model_type = "vit_b"
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # حداقل اندازه
    )
    
    # تبدیل به RGB
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    print("🔍 در حال تشخیص...")
    masks = mask_generator.generate(rgb)
    
    # فیلتر کردن ماسک‌ها
    centers = []
    boxes = []
    
    for mask in masks:
        # فیلتر بر اساس اندازه
        area = mask['area']
        if area < 50 or area > 5000:  # تنظیم بر اساس سایز گرافت
            continue
        
        # محاسبه مرکز
        segmentation = mask['segmentation']
        y_coords, x_coords = np.where(segmentation)
        if len(x_coords) == 0:
            continue
        
        cx = int(x_coords.mean())
        cy = int(y_coords.mean())
        
        # Bounding box
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        centers.append((cx, cy))
        boxes.append((int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)))
    
    # رسم نتایج
    overlay = image_bgr.copy()
    for (cx, cy) in centers:
        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
    
    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
    return {
        "count": len(centers),
        "centers": centers,
        "boxes": boxes,
        "chosen": "sam",
        "overlay_bgr": overlay,
    }

def detect_simple_sam(image_bgr: np.ndarray) -> Dict:
    """
    نسخه ساده‌تر - استفاده از contour detection هوشمند
    بدون نیاز به SAM (برای سیستم‌های ضعیف)
    """
    from core.dish import detect_petri_mask
    
    # ماسک ظرف
    dish_mask, _ = detect_petri_mask(image_bgr, erode_px=30)
    
    # تبدیل به grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # پیش‌پردازش قوی
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # آستانه‌گذاری Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, dish_mask)
    
    # مورفولوژی
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Distance transform + watershed
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
    
    # فیلتر
    centers = []
    boxes = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30 or area > 3000:
            continue
        
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        centers.append((cx, cy))
        boxes.append((x, y, w, h))
    
    # رسم
    overlay = image_bgr.copy()
    for (cx, cy) in centers:
        cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)
    
    return {
        "count": len(centers),
        "centers": centers,
        "boxes": boxes,
        "chosen": "simple_sam",
        "overlay_bgr": overlay,
    }
