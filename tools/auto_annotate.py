# tools/auto_annotate.py
"""
برچسب‌گذاری خودکار با استفاده از روش‌های CV
این کد خودکار گرافت‌ها رو پیدا می‌کنه و فایل YOLO می‌سازه!
"""
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import argparse

# Import از core
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.graft_counter import count_grafts


def auto_annotate_image(image_path: str, output_dir: str = "dataset_yolo/train"):
    """
    یه تصویر رو به صورت خودکار برچسب‌گذاری می‌کنه
    """
    # خواندن تصویر
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ نمی‌تونم تصویر رو بخونم: {image_path}")
        return False

    print(f"🔍 در حال تحلیل: {image_path}")

    # تشخیص گرافت‌ها با روش CV
    result = count_grafts(img, preset="clientdemo")

    points = result["points"]  # نقاط مرکزی
    count = len(points)

    print(f"✅ پیدا شد: {count} گرافت")

    if count == 0:
        print("⚠️ هیچ گرافتی پیدا نشد!")
        return False

    # ساخت پوشه‌ها
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # نام فایل
    stem = Path(image_path).stem

    # ذخیره تصویر
    img_save_path = images_dir / f"{stem}.jpg"
    cv2.imwrite(str(img_save_path), img)

    # ساخت فایل YOLO
    # فرمت: class_id center_x center_y width height (normalized)
    h, w = img.shape[:2]
    lines = []
    box_size = 32  # اندازه جعبه پیش‌فرض

    for (x, y) in points:
        # نرمالیزه کردن
        cx_norm = float(x) / w
        cy_norm = float(y) / h
        w_norm = box_size / w
        h_norm = box_size / h

        lines.append(f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    # ذخیره فایل label
    label_path = labels_dir / f"{stem}.txt"
    with open(label_path, 'w') as f:
        f.write('\n'.join(lines))

    # ذخیره تصویر debug (با نقاط)
    overlay = cv2.cvtColor(result["overlay_clean"], cv2.COLOR_RGB2BGR)
    debug_path = images_dir / f"{stem}_debug.jpg"
    cv2.imwrite(str(debug_path), overlay)

    print(f"💾 ذخیره شد:")
    print(f"   📁 تصویر: {img_save_path}")
    print(f"   🏷️  برچسب: {label_path}")
    print(f"   🔍 Debug: {debug_path}")

    return True


def auto_annotate_folder(input_folder: str, output_dir: str = "dataset_yolo/train"):
    """
    همه تصاویر یه پوشه رو برچسب‌گذاری می‌کنه
    """
    input_path = Path(input_folder)

    if not input_path.exists():
        print(f"❌ پوشه پیدا نشد: {input_folder}")
        return

    # پیدا کردن همه تصاویر
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in extensions:
        images.extend(list(input_path.glob(f"*{ext}")))
        images.extend(list(input_path.glob(f"*{ext.upper()}")))

    total = len(images)
    print(f"📊 تعداد تصاویر پیدا شده: {total}")

    if total == 0:
        print("❌ هیچ تصویری پیدا نشد!")
        return

    # پردازش تصاویر
    success_count = 0
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{total}] ", end="")
        if auto_annotate_image(str(img_path), output_dir):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"✅ تمام شد!")
    print(f"📊 موفق: {success_count}/{total}")
    print(f"📁 خروجی: {output_dir}")

    # ساخت فایل data.yaml
    yaml_path = Path(output_dir).parent / "data.yaml"
    yaml_content = f"""# Dataset Config
path: {Path(output_dir).parent.absolute()}
train: train/images
val: val/images

nc: 1
names: ['graft']
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"📝 فایل config: {yaml_path}")
    print(f"\n💡 مرحله بعد:")
    print(f"   1. تصاویر رو بررسی کن (فایل‌های *_debug.jpg)")
    print(f"   2. اگه خوب بود، 80% رو بذار توی train/ و 20% رو توی val/")
    print(f"   3. مدل رو آموزش بده!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="برچسب‌گذاری خودکار تصاویر گرافت")
    parser.add_argument("input", help="مسیر تصویر یا پوشه تصاویر")
    parser.add_argument("-o", "--output", default="dataset_yolo/train",
                        help="پوشه خروجی (پیش‌فرض: dataset_yolo/train)")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # یه تصویر
        auto_annotate_image(args.input, args.output)
    elif input_path.is_dir():
        # یه پوشه
        auto_annotate_folder(args.input, args.output)
    else:
        print(f"❌ فایل یا پوشه پیدا نشد: {args.input}")