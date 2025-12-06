# core/yolo_detector.py
"""
تشخیص گرافت با استفاده از YOLOv8
"""
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class GraftDetectorYOLO:
    """تشخیص‌دهنده گرافت با YOLO"""

    def __init__(self, model_path: Optional[str] = None):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics نصب نیست. اجرا کن: pip install ultralytics")

        # بارگذاری مدل
        if model_path and os.path.exists(model_path):
            print(f"🤖 بارگذاری مدل: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("⚠️ مدل پیدا نشد. از مدل پایه استفاده می‌شود")
            self.model = YOLO('yolov8n.pt')

        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

    def detect_grafts(self, image_bgr: np.ndarray) -> Dict:
        """
        تشخیص گرافت‌ها در تصویر
        """
        # تشخیص
        results = self.model.predict(
            image_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        # استخراج نتایج
        boxes = []
        centers = []
        confidences = []

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                # محاسبه مرکز
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                centers.append((cx, cy))
                confidences.append(conf)

        # رسم نتایج
        overlay = self._draw_detections(image_bgr, boxes, centers, confidences)

        return {
            "count": len(centers),
            "boxes": boxes,
            "centers": centers,
            "confidences": confidences,
            "overlay_bgr": overlay,
            "method": "yolov8"
        }

    def _draw_detections(self, image: np.ndarray, boxes: List,
                         centers: List, confidences: List) -> np.ndarray:
        """رسم جعبه‌ها و مراکز روی تصویر"""
        overlay = image.copy()

        for box, (cx, cy), conf in zip(boxes, centers, confidences):
            x1, y1, x2, y2 = box

            # رسم جعبه
            color = (0, 255, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # رسم نقطه مرکز
            cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)

            # نوشتن اطمینان
            label = f"{conf:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # نوشتن تعداد
        cv2.putText(overlay, f"Count: {len(centers)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return overlay


def analyze_bgr_yolo(
        image_bgr: np.ndarray,
        model_path: str = "weights/yolo_graft/run1/weights/best.pt"
) -> Dict:
    """
    تابع اصلی برای تحلیل تصویر با YOLO
    """
    detector = GraftDetectorYOLO(model_path)
    result = detector.detect_grafts(image_bgr)

    # تبدیل به فرمت pipeline
    return {
        "count": result["count"],
        "centers": result["centers"],
        "boxes": [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in result["boxes"]],
        "chosen": "yolov8",
        "overlay_bgr": result["overlay_bgr"],
        "debug_bgr": result["overlay_bgr"],
    }