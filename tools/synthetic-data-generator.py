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
    def __init__(
        self,
        output_dir: str = "synthetic_dataset",
        bg_variants: int = 3,
        overlap_chance: float = 0.35,
        blur_chance: float = 0.3,
        val_ratio: float = 0.1,
        image_size: tuple[int, int] = (2000, 2000),
        dish_radius_ratio: float = 0.42,
        preview_count: int = 8,
        seed: int | None = None,
    ):
        """ابزار ساخت دیتاست مصنوعی برای آموزش YOLO.

        پارامترها:
        - ``bg_variants``: تعداد تمِ نوری/رنگی پس‌زمینه برای کاهش فاصله‌ی domain.
        - ``overlap_chance``: احتمال کاهش فاصله‌ی بین گرافت‌ها برای شبیه‌سازی چسبندگی.
        - ``blur_chance``: احتمال اضافه‌کردن بلور/defocus به بعضی گرافت‌ها.
        - ``val_ratio``: نسبت اعتبارسنجی (به‌طور تصادفی از تصاویر ساخته‌شده جدا می‌شود).
        - ``seed``: مقدار ثابت برای تولید دیتاست تکرارپذیر.
        """

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

        self.bg_variants = max(1, int(bg_variants))
        self.overlap_chance = max(0.0, min(1.0, float(overlap_chance)))
        self.blur_chance = max(0.0, min(1.0, float(blur_chance)))
        self.val_ratio = max(0.0, min(0.9, float(val_ratio)))
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.dish_radius_ratio = max(0.1, min(0.49, float(dish_radius_ratio)))
        self.preview_count = max(0, int(preview_count))

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def create_petri_dish_background(self) -> np.ndarray:
        """ساخت پس‌زمینه ظرف پتری"""
        H, W = self.image_size

        # تم‌های مختلف ظرف برای کاهش overfitting به یک نور خاص
        palette = [
            np.array([180, 200, 160], dtype=np.uint8),  # سبز مایل به آبی روشن
            np.array([165, 185, 175], dtype=np.uint8),  # خاکستری-سبز
            np.array([195, 210, 185], dtype=np.uint8),  # سبز-زرد روشن
        ][: self.bg_variants]

        base_color = random.choice(palette)

        # پس‌زمینه پایه
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        bg[:] = base_color

        # نویز برای بافت
        noise_sigma = random.uniform(10, 20)
        noise = np.random.normal(0, noise_sigma, (H, W, 3)).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # گرادیانت نور با شدت تصادفی
        Y, X = np.ogrid[:H, :W]
        center_x, center_y = W // 2, H // 2
        grad_strength = random.uniform(0.15, 0.35)
        gradient = 1.0 - grad_strength * np.sqrt(
            (X - center_x) ** 2 + (Y - center_y) ** 2
        ) / (W // 2)
        gradient = np.clip(gradient, 0.75, 1.25)
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c] * gradient, 0, 255).astype(np.uint8)

        # هاله نور/ویگنت معکوس برای نور موضعی
        vignette_strength = random.uniform(-0.08, 0.12)
        vignette = 1.0 + vignette_strength * gradient
        bg = np.clip(bg.astype(np.float32) * vignette[..., None], 0, 255).astype(
            np.uint8
        )

        # دایره ظرف
        cv2.circle(bg, (center_x, center_y), int(W * 0.48), (120, 120, 120), 15)

        # بافت ریز
        texture = np.random.normal(0, 4, (H, W, 3)).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + texture, 0, 255).astype(np.uint8)

        return bg

    def create_graft(self, size: int = 20) -> tuple:
        """
        ساخت یه گرافت مصنوعی
        برگشت: (image, mask, shadow) - گرافت شبیه‌سازی شده
        """
        # اندازه تصادفی
        w = random.randint(int(size * 0.7), int(size * 1.3))
        h = random.randint(int(size * 0.7), int(size * 1.3))

        # ساخت ماسک بیضی (گرافت‌ها معمولاً بیضی شکلن)
        mask = np.zeros((h * 3, w * 3), dtype=np.uint8)
        center = (w * 3 // 2, h * 3 // 2)
        cv2.ellipse(
            mask,
            center,
            (w, h),
            random.randint(0, 180),
            0,
            360,
            255,
            -1,
        )

        # ساخت تصویر گرافت
        graft = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)

        # رنگ گرافت (سفید-کرم-خاکستری روشن)
        base_color = random.choice(
            [
                [220, 220, 200],  # کرم
                [230, 230, 220],  # سفید کمی زرد
                [200, 210, 200],  # خاکستری-سبز
                [240, 240, 235],  # سفید روشن
            ]
        )

        graft[mask > 0] = base_color

        # اضافه کردن بافت (texture) داخلی
        for _ in range(3, 7):
            noise = np.random.normal(
                0, random.uniform(5, 12), (h * 3, w * 3)
            ).astype(np.int16)
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

        # اضافه کردن کمی بلور برای شبیه‌سازی فوکوس ناقص
        if random.random() < self.blur_chance:
            k = random.choice([3, 5])
            graft = cv2.GaussianBlur(graft, (k, k), random.uniform(0.5, 1.2))

        # اضافه کردن سایه نرم
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow = cv2.dilate(mask, kernel, iterations=2)
        shadow = cv2.GaussianBlur(shadow, (15, 15), 0)

        return graft, mask, shadow

    def place_grafts_on_dish(
        self, bg: np.ndarray, num_grafts: int = 500
    ) -> tuple:
        """
        قرار دادن گرافت‌ها روی ظرف پتری
        """
        H, W = bg.shape[:2]
        result = bg.copy()

        centers = []
        boxes = []

        # محدوده قرار دادن (داخل دایره ظرف)
        center_x, center_y = W // 2, H // 2
        radius = int(W * self.dish_radius_ratio)  # کمی کوچکتر از لبه

        attempts = 0
        max_attempts = num_grafts * 12

        while len(centers) < num_grafts and attempts < max_attempts:
            attempts += 1

            # موقعیت تصادفی
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(0, radius)
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))

            # چک کن که خیلی نزدیک به گرافت‌های دیگه نباشه
            too_close = False
            min_distance = 15

            # احتمالاً اجازهٔ فاصله کمتر برای ایجاد چسبندگی
            if random.random() < self.overlap_chance:
                min_distance = random.randint(4, 10)

            for (cx, cy) in centers:
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
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
            if (
                x - gw // 2 < 0
                or x + gw // 2 >= W
                or y - gh // 2 < 0
                or y + gh // 2 >= H
            ):
                continue

            # محاسبه موقعیت
            x1 = x - gw // 2
            y1 = y - gh // 2
            x2 = x1 + gw
            y2 = y1 + gh

            # اضافه کردن سایه
            shadow_region = result[y1:y2, x1:x2]
            shadow_3d = cv2.cvtColor(shadow, cv2.COLOR_GRAY2BGR)
            shadow_3d = (
                shadow_3d.astype(np.float32) / 255.0 * 0.3
            ).astype(np.float32)
            shadow_region = (
                shadow_region.astype(np.float32) * (1.0 - shadow_3d)
            ).astype(np.uint8)
            result[y1:y2, x1:x2] = shadow_region

            # اضافه کردن گرافت با alpha blending
            mask_3d = (
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            )
            region = result[y1:y2, x1:x2]

            blended = graft.astype(np.float32) * mask_3d + region.astype(
                np.float32
            ) * (1.0 - mask_3d)
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

    def _save_preview(
        self,
        image: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        out_path: Path,
    ) -> None:
        """ذخیره نسخه پیش‌نمایش با باکس و شمارش برای چک دستی سریع"""
        preview = image.copy()

        for idx, (bx, by, bw, bh) in enumerate(boxes, start=1):
            pt1 = (int(bx), int(by))
            pt2 = (int(bx + bw), int(by + bh))
            cv2.rectangle(preview, pt1, pt2, (80, 30, 200), 2)
            cv2.putText(
                preview,
                str(idx),
                (int(bx), int(max(10, by - 6))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (70, 20, 180),
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(out_path), preview)

    def generate_dataset(
        self, num_images: int = 100, grafts_per_image: tuple = (300, 800)
    ):
        """
        ساخت دیتاست کامل
        """
        print(f"🎨 شروع ساخت {num_images} عکس مصنوعی...")
        print(
            f"📊 تعداد گرافت در هر عکس: {grafts_per_image[0]}-{grafts_per_image[1]}"
        )
        print(
            f"🌈 تم‌های پس‌زمینه: {self.bg_variants} | چسبندگی: {self.overlap_chance:.2f} | بلور: {self.blur_chance:.2f}"
        )
        print(f"🧪 نسبت اعتبارسنجی: {int(self.val_ratio * 100)}%")
        print(
            f"🖼️  اندازه تصویر: {self.image_size[0]}x{self.image_size[1]} | شعاع ظرف: {self.dish_radius_ratio:.2f}"
        )

        manifest = []

        preview_indices = set(
            random.sample(range(num_images), min(self.preview_count, num_images))
        )

        train_images_dir = self.images_dir / "train"
        val_images_dir = self.images_dir / "val"
        train_labels_dir = self.labels_dir / "train"
        val_labels_dir = self.labels_dir / "val"

        for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

        val_count = int(num_images * self.val_ratio)
        val_indices = (
            set(random.sample(range(num_images), val_count))
            if val_count > 0
            else set()
        )

        for img_idx in range(num_images):
            # تعداد تصادفی گرافت
            num_grafts = random.randint(
                grafts_per_image[0], grafts_per_image[1]
            )

            # ساخت پس‌زمینه
            bg = self.create_petri_dish_background()

            # اضافه کردن گرافت‌ها
            result, centers, boxes = self.place_grafts_on_dish(bg, num_grafts)

            # ذخیره تصویر
            img_name = f"synthetic_{img_idx:04d}.jpg"
            split = "val" if img_idx in val_indices else "train"
            img_dir = val_images_dir if split == "val" else train_images_dir
            label_dir = val_labels_dir if split == "val" else train_labels_dir

            img_path = img_dir / img_name
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

                label_lines.append(
                    f"0 {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}"
                )

            label_path = label_dir / f"synthetic_{img_idx:04d}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

            if img_idx in preview_indices:
                preview_dir = self.output_dir / "previews"
                preview_dir.mkdir(parents=True, exist_ok=True)
                preview_path = preview_dir / img_name
                self._save_preview(result, boxes, preview_path)

            manifest.append(
                {
                    "file": img_name,
                    "count": len(centers),
                    "boxes": boxes,
                    "split": split,
                }
            )

            if (img_idx + 1) % 10 == 0:
                print(
                    f"✅ ساخته شد: {img_idx + 1}/{num_images} ({len(centers)} گرافت)"
                )

        # ساخت data.yaml
        self.create_yaml()

        # ذخیره manifest برای چک دستی سریع
        manifest_data = {
            "settings": {
                "bg_variants": self.bg_variants,
                "overlap_chance": self.overlap_chance,
                "blur_chance": self.blur_chance,
                "val_ratio": self.val_ratio,
                "image_size": self.image_size,
                "dish_radius_ratio": self.dish_radius_ratio,
                "preview_count": self.preview_count,
                "num_images": num_images,
                "grafts_per_image": grafts_per_image,
            },
            "samples": manifest,
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest_data, mf, ensure_ascii=False, indent=2)

        print(f"🗒️  فایل manifest: {manifest_path}")

        print(f"\n{'=' * 60}")
        print(f"✅ تمام! {num_images} عکس مصنوعی ساخته شد")
        print(f"📁 مسیر: {self.output_dir}")
        print("📊 آماده برای آموزش YOLO!")
        print(f"{'=' * 60}\n")

    def create_yaml(self):
        """ساخت فایل data.yaml برای YOLO"""
        yaml_content = f"""# Synthetic Graft Dataset
path: {self.output_dir.absolute()}
train: images/train
val: images/val

nc: 1
names: ['graft']
"""
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(f"📝 فایل config: {yaml_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ساخت دیتاست مصنوعی گرافت")
    parser.add_argument(
        "-n",
        "--num_images",
        type=int,
        default=100,
        help="تعداد عکس‌های مصنوعی (پیش‌فرض: 100)",
    )
    parser.add_argument(
        "-g",
        "--grafts_min",
        type=int,
        default=300,
        help="حداقل تعداد گرافت در هر عکس",
    )
    parser.add_argument(
        "-G",
        "--grafts_max",
        type=int,
        default=800,
        help="حداکثر تعداد گرافت در هر عکس",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="synthetic_dataset",
        help="مسیر خروجی",
    )
    parser.add_argument(
        "--bg-variants",
        type=int,
        default=3,
        help="تعداد تم نوری/رنگی پس‌زمینه",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.35,
        help="احتمال نزدیک شدن گرافت‌ها برای شبیه‌سازی چسبندگی",
    )
    parser.add_argument(
        "--blur",
        type=float,
        default=0.3,
        help="احتمال بلور کردن گرافت برای فوکوس ناقص",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="نسبت اعتبارسنجی (پیش‌فرض 0.1 = ده درصد)",
    )
    parser.add_argument(
        "--img-size",
        nargs=2,
        type=int,
        default=[2000, 2000],
        metavar=("WIDTH", "HEIGHT"),
        help="اندازه تصویر خروجی (پیش‌فرض 2000 2000)",
    )
    parser.add_argument(
        "--dish-radius-ratio",
        type=float,
        default=0.42,
        help="نسبت شعاع ظرف به عرض تصویر (0.1-0.49)",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=8,
        help="تعداد پیش‌نمایش باکس‌گذاری برای بررسی برچسب‌ها",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed ثابت برای تکرارپذیری",
    )

    args = parser.parse_args()

    generator = SyntheticGraftGenerator(
        args.output,
        bg_variants=args.bg_variants,
        overlap_chance=args.overlap,
        blur_chance=args.blur,
        val_ratio=args.val_ratio,
        image_size=tuple(args.img_size),
        dish_radius_ratio=args.dish_radius_ratio,
        preview_count=args.preview_count,
        seed=args.seed,
    )
    generator.generate_dataset(
        num_images=args.num_images,
        grafts_per_image=(args.grafts_min, args.grafts_max),
    )

    print("\n💡 مرحله بعد:")
    print("   آموزش YOLO با دستور زیر:")
    print(
        f"   yolo detect train data={args.output}/data.yaml model=yolov8n.pt epochs=100"
    )
