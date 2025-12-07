#!/usr/bin/env python3
# tools/generate_synthetic_data.py
"""
ساخت دیتاست مصنوعی شبیه گرافت‌ها برای آموزش YOLO
این عکس‌ها شبیه‌سازی واقعی گرافت‌ها در ظرف پتری هستن!
"""
import cv2
import numpy as np
import json
from pathlib import Path
import random

class SyntheticGraftGenerator:
    def __init__(self, output_dir: str = "synthetic_dataset"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
    
    def create_petri_dish_background(self, size: tuple = (2000, 2000)) -> np.ndarray:
        """ساخت پس‌زمینه ظرف پتری"""
        H, W = size
        
        # پس‌زمینه آبی-سبز (مثل ظرف پتری)
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        
        # رنگ پایه (آبی-سبز روشن)
        base_color = np.array([180, 200, 160], dtype=np.uint8)  # BGR
        bg[:] = base_color
        
        # اضافه کردن نویز برای طبیعی‌تر شدن
        noise = np.random.normal(0, 15, (H, W, 3)).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # گرادیانت نرم (شبیه‌سازی نور)
        Y, X = np.ogrid[:H, :W]
        center_x, center_y = W // 2, H // 2
        gradient = 1.0 - 0.2 * np.sqrt((X - center_x)**2 + (Y - center_y)**2) / (W // 2)
        gradient = np.clip(gradient, 0.8, 1.2)
        
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c] * gradient, 0, 255).astype(np.uint8)
        
        # رسم دایره ظرف
        cv2.circle(bg, (center_x, center_y), int(W * 0.48), (120, 120, 120), 15)
        
        # اضافه کردن بافت (texture) به پس‌زمینه
        texture = np.random.normal(0, 3, (H, W, 3)).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + texture, 0, 255).astype(np.uint8)
        
        return bg
    
    def create_graft(self, size: int = 20) -> tuple:
        """
        ساخت یه گرافت مصنوعی
        برگشت: (image, mask) - گرافت شبیه‌سازی شده
        """
        # اندازه تصادفی
        w = random.randint(int(size * 0.7), int(size * 1.3))
        h = random.randint(int(size * 0.7), int(size * 1.3))
        
        # ساخت ماسک بیضی (گرافت‌ها معمولاً بیضی شکلن)
        mask = np.zeros((h * 3, w * 3), dtype=np.uint8)
        center = (w * 3 // 2, h * 3 // 2)
        cv2.ellipse(mask, center, (w, h), random.randint(0, 180), 0, 360, 255, -1)
        
        # ساخت تصویر گرافت
        graft = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)
        
        # رنگ گرافت (سفید-کرم-خاکستری روشن)
        base_color = random.choice([
            [220, 220, 200],  # کرم
            [230, 230, 220],  # سفید کمی زرد
            [200, 210, 200],  # خاکستری-سبز
            [240, 240, 235],  # سفید روشن
        ])
        
        graft[mask > 0] = base_color
        
        # اضافه کردن بافت (texture) داخلی
        for i in range(3, 7):
            noise = np.random.normal(0, random.uniform(5, 12), (h * 3, w * 3)).astype(np.int16)
            for c in range(3):
                temp = graft[:, :, c].astype(np.int16) + noise
                graft[:, :, c] = np.clip(temp, 0, 255).astype(np.uint8)
        
        # اضافه کردن خطوط (شبیه رشته‌های گرافت)
        num_lines = random.randint(3, 8)
        for _ in range(num_lines):
            pt1 = (random.randint(0, w * 3), random.randint(0, h * 3))
            pt2 = (random.randint(0, w * 3), random.randint(0, h * 3))
            color = [random.randint(180, 220) for _ in range(3)]
            cv2.line(graft, pt1, pt2, color, 1)
        
        # اعمال ماسک
        graft[mask == 0] = 0
        
        # اضافه کردن سایه نرم
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow = cv2.dilate(mask, kernel, iterations=2)
        shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
        
        return graft, mask, shadow
    
    def place_grafts_on_dish(self, bg: np.ndarray, num_grafts: int = 500) -> tuple:
        """
        قرار دادن گرافت‌ها روی ظرف پتری
        """
        H, W = bg.shape[:2]
        result = bg.copy()
        
        centers = []
        boxes = []
        
        # محدوده قرار دادن (داخل دایره ظرف)
        center_x, center_y = W // 2, H // 2
        radius = int(W * 0.42)  # کمی کوچکتر از لبه
        
        attempts = 0
        max_attempts = num_grafts * 10
        
        while len(centers) < num_grafts and attempts < max_attempts:
            attempts += 1
            
            # موقعیت تصادفی
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(0, radius)
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            
            # چک کن که خیلی نزدیک به گرافت‌های دیگه نباشه
            too_close = False
            min_distance = 15  # حداقل فاصله
            for (cx, cy) in centers:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # ساخت گرافت
            graft_size = random.randint(12, 28)
            graft, mask, shadow = self.create_graft(graft_size)
            
            gh, gw = graft.shape[:2]
            
            # مطمئن شو که داخل فریم هست
            if x - gw // 2 < 0 or x + gw // 2 >= W or y - gh // 2 < 0 or y + gh // 2 >= H:
                continue
            
            # محاسبه موقعیت
            x1 = x - gw // 2
            y1 = y - gh // 2
            x2 = x1 + gw
            y2 = y1 + gh
            
            # اضافه کردن سایه
            shadow_region = result[y1:y2, x1:x2]
            shadow_3d = cv2.cvtColor(shadow, cv2.COLOR_GRAY2BGR)
            shadow_3d = (shadow_3d.astype(np.float32) / 255.0 * 0.3).astype(np.float32)
            shadow_region = (shadow_region.astype(np.float32) * (1.0 - shadow_3d)).astype(np.uint8)
            result[y1:y2, x1:x2] = shadow_region
            
            # اضافه کردن گرافت با alpha blending
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            region = result[y1:y2, x1:x2]
            
            blended = (graft.astype(np.float32) * mask_3d + 
                      region.astype(np.float32) * (1.0 - mask_3d))
            result[y1:y2, x1:x2] = blended.astype(np.uint8)
            
            # ذخیره مرکز و bounding box
            centers.append((x, y))
            
            # محاسبه bounding box واقعی گرافت
            ys, xs = np.where(mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                bbox_x = x1 + xs.min()
                bbox_y = y1 + ys.min()
                bbox_w = xs.max() - xs.min()
                bbox_h = ys.max() - ys.min()
                boxes.append((bbox_x, bbox_y, bbox_w, bbox_h))
        
        return result, centers, boxes
    
    def generate_dataset(self, num_images: int = 100, grafts_per_image: tuple = (300, 800)):
        """
        ساخت دیتاست کامل
        """
        print(f"🎨 شروع ساخت {num_images} عکس مصنوعی...")
        print(f"📊 تعداد گرافت در هر عکس: {grafts_per_image[0]}-{grafts_per_image[1]}")
        
        for img_idx in range(num_images):
            # تعداد تصادفی گرافت
            num_grafts = random.randint(grafts_per_image[0], grafts_per_image[1])
            
            # ساخت پس‌زمینه
            bg = self.create_petri_dish_background()
            
            # اضافه کردن گرافت‌ها
            result, centers, boxes = self.place_grafts_on_dish(bg, num_grafts)
            
            # ذخیره تصویر
            img_name = f"synthetic_{img_idx:04d}.jpg"
            img_path = self.images_dir / img_name
            cv2.imwrite(str(img_path), result)
            
            # ذخیره labels (YOLO format)
            H, W = result.shape[:2]
            label_lines = []
            
            for (bx, by, bw, bh) in boxes:
                # محاسبه مرکز و نرمالیزه کردن
                cx = (bx + bw / 2) / W
                cy = (by + bh / 2) / H
                w_norm = bw / W
                h_norm = bh / H
                
                label_lines.append(f"0 {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            label_path = self.labels_dir / f"synthetic_{img_idx:04d}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))
            
            if (img_idx + 1) % 10 == 0:
                print(f"✅ ساخته شد: {img_idx + 1}/{num_images} ({len(centers)} گرافت)")
        
        # ساخت data.yaml
        self.create_yaml()
        
        print(f"\n{'='*60}")
        print(f"✅ تمام! {num_images} عکس مصنوعی ساخته شد")
        print(f"📁 مسیر: {self.output_dir}")
        print(f"📊 آماده برای آموزش YOLO!")
        print(f"{'='*60}\n")
    
    def create_yaml(self):
        """ساخت فایل data.yaml برای YOLO"""
        yaml_content = f"""# Synthetic Graft Dataset
path: {self.output_dir.absolute()}
train: images
val: images  # برای شروع، از همون images استفاده می‌کنیم

nc: 1
names: ['graft']
"""
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"📝 فایل config: {yaml_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ساخت دیتاست مصنوعی گرافت")
    parser.add_argument("-n", "--num_images", type=int, default=100,
                       help="تعداد عکس‌های مصنوعی (پیش‌فرض: 100)")
    parser.add_argument("-g", "--grafts_min", type=int, default=300,
                       help="حداقل تعداد گرافت در هر عکس")
    parser.add_argument("-G", "--grafts_max", type=int, default=800,
                       help="حداکثر تعداد گرافت در هر عکس")
    parser.add_argument("-o", "--output", default="synthetic_dataset",
                       help="مسیر خروجی")
    
    args = parser.parse_args()
    
    generator = SyntheticGraftGenerator(args.output)
    generator.generate_dataset(
        num_images=args.num_images,
        grafts_per_image=(args.grafts_min, args.grafts_max)
    )
    
    print("\n💡 مرحله بعد:")
    print("   آموزش YOLO با دستور زیر:")
    print(f"   yolo detect train data={args.output}/data.yaml model=yolov8n.pt epochs=100")
