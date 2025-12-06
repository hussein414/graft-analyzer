# tools/annotate_grafts.py
"""
ابزار ساده برای برچسب‌گذاری گرافت‌ها
اجرا: streamlit run tools/annotate_grafts.py
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path

st.set_page_config(page_title="🔬 Graft Annotator", layout="wide")
st.title("🔬 ابزار برچسب‌گذاری گرافت‌ها")

# Session state
if 'points' not in st.session_state:
    st.session_state.points = []
if 'image' not in st.session_state:
    st.session_state.image = None
if 'image_name' not in st.session_state:
    st.session_state.image_name = None

# Sidebar
st.sidebar.header("⚙️ تنظیمات")
output_dir = st.sidebar.text_input("📁 پوشه ذخیره", "dataset_yolo/train")
Path(output_dir).mkdir(parents=True, exist_ok=True)
Path(output_dir + "/images").mkdir(parents=True, exist_ok=True)
Path(output_dir + "/labels").mkdir(parents=True, exist_ok=True)

# آپلود تصویر
uploaded = st.file_uploader("📤 تصویر ظرف پتری رو آپلود کن", type=['jpg', 'jpeg', 'png'])

if uploaded:
    # بارگذاری تصویر
    img_bytes = uploaded.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # اگر تصویر جدیده، نقاط رو پاک کن
    if st.session_state.image_name != uploaded.name:
        st.session_state.points = []
        st.session_state.image_name = uploaded.name

    st.session_state.image = img_rgb

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("🖼️ تصویر (کلیک کن تا نقطه بذاری)")

        # رسم نقاط روی تصویر
        display_img = img_rgb.copy()
        for i, (x, y) in enumerate(st.session_state.points):
            cv2.circle(display_img, (x, y), 7, (255, 0, 0), -1)
            cv2.circle(display_img, (x, y), 8, (255, 255, 255), 2)
            cv2.putText(display_img, str(i + 1), (x + 12, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # نمایش تصویر
        st.image(display_img, use_container_width=True)

        # راهنما
        st.info("💡 **راهنما:** برای افزودن نقطه از ورودی دستی استفاده کن")

    with col2:
        st.subheader(f"📊 آمار")
        st.metric("تعداد نقاط", len(st.session_state.points))
        st.metric("نام فایل", uploaded.name)

        st.markdown("---")
        st.subheader("➕ افزودن نقطه")

        # ورودی دستی نقاط
        h, w = img_rgb.shape[:2]
        x_input = st.number_input("X", min_value=0, max_value=w - 1, value=w // 2, key="x")
        y_input = st.number_input("Y", min_value=0, max_value=h - 1, value=h // 2, key="y")

        if st.button("➕ اضافه کن", use_container_width=True):
            st.session_state.points.append((int(x_input), int(y_input)))
            st.rerun()

        st.markdown("---")

        # دکمه‌های کنترل
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ حذف آخرین", use_container_width=True):
                if st.session_state.points:
                    st.session_state.points.pop()
                    st.rerun()

        with col_b:
            if st.button("🧹 پاک کن همه", use_container_width=True):
                st.session_state.points = []
                st.rerun()

        st.markdown("---")

        # نمایش لیست نقاط
        if st.session_state.points:
            st.subheader("📍 لیست نقاط")
            for i, (x, y) in enumerate(st.session_state.points):
                st.text(f"{i + 1}. ({x}, {y})")

        st.markdown("---")

        # ذخیره
        st.subheader("💾 ذخیره")

        if st.button("💾 ذخیره", type="primary", use_container_width=True):
            if not st.session_state.points:
                st.warning("⚠️ نقطه‌ای وجود ندارد!")
            else:
                h, w = img_rgb.shape[:2]
                stem = Path(uploaded.name).stem

                # ذخیره فرمت YOLO
                lines = []
                box_size = 32

                for x, y in st.session_state.points:
                    cx_norm = x / w
                    cy_norm = y / h
                    w_norm = box_size / w
                    h_norm = box_size / h
                    lines.append(f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

                # ذخیره
                label_path = Path(output_dir) / "labels" / f"{stem}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(lines))

                img_path = Path(output_dir) / "images" / f"{stem}.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

                st.success(f"✅ ذخیره شد! {len(st.session_state.points)} گرافت")
                st.info(f"📁 {img_path}")

else:
    st.info("👆 یه تصویر آپلود کن")
    st.markdown("""
    ### راهنما:
    1. تصویر آپلود کن
    2. X و Y هر گرافت رو وارد کن
    3. ذخیره کن
    4. حداقل 100 تصویر برچسب‌گذاری کن
    """)

st.sidebar.success(f"نقاط: {len(st.session_state.points)}")