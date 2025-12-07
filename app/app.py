# app/app.py
import os, sys, io, base64
from typing import Dict
import streamlit as st
from PIL import Image

# import core/*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.graft_counter import count_grafts
from core.ml_infer import detect_accelerators
import cv2
import numpy as np

st.set_page_config(page_title="Graft Analyzer", layout="wide")
st.title("🧪 Graft Analyzer — با تنظیمات پیشرفته")

# --- Sidebar ---
st.sidebar.header("⚙️ تنظیمات")

# انتخاب Preset
preset_option = st.sidebar.selectbox(
    "🎯 Preset",
    ["clientdemo", "ultra_dense", "aggressive", "qc"],  # ← اضافه کردم aggressive
    help="clientdemo: عادی | ultra_dense: تراکم بالا | aggressive: خیلی حساس | qc: کنترل کیفیت"
)
# تنظیمات پیشرفته
with st.sidebar.expander("🔧 تنظیمات پیشرفته (اختیاری)"):
    st.info("اگه نتیجه خوب نبود، اینا رو تغییر بده")

    use_custom = st.checkbox("استفاده از تنظیمات دستی", value=False)

    if use_custom:
        adapt_block = st.slider("Adaptive Block", 11, 91, 31, step=2)
        adapt_C = st.slider("Adaptive C", -20, 0, -8)
        watershed_ratio = st.slider("Watershed Ratio", 0.3, 0.8, 0.5, step=0.05)
        min_size = st.slider("حداقل اندازه گرافت", 1.0, 10.0, 3.0, step=0.5)
        max_size = st.slider("حداکثر اندازه گرافت", 20.0, 80.0, 40.0, step=5.0)
    else:
        adapt_block = None
        adapt_C = None
        watershed_ratio = None
        min_size = None
        max_size = None

st.sidebar.markdown("---")
st.sidebar.subheader("💻 Hardware")
st.sidebar.write(detect_accelerators())


# --- Helpers ---
def _show_overlay_from_b64(b64: str, caption: str):
    if not b64:
        st.warning("خروجی تصویری موجود نیست.")
        return
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    st.image(img, caption=caption, use_column_width=True)


def _img_to_b64(img_rgb: np.ndarray) -> str:
    """تبدیل numpy array به base64"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def analyze_with_params(img_bgr, preset, custom_params=None):
    """تحلیل با پارامترهای قابل تنظیم"""
    if custom_params and custom_params.get("use_custom"):
        # استفاده از پارامترهای سفارشی
        from core.graft_counter import Preset
        custom_preset = Preset(
            spec_quantile=99.8,
            log_sigmas=(1.5, 2.5, 3.5, 5.0),
            adapt_block=custom_params.get("adapt_block", 31),
            adapt_C=custom_params.get("adapt_C", -8),
            watershed_peak_ratio=custom_params.get("watershed_ratio", 0.5),
            elong_max=5.0,
            circ_min=0.05,
            th_min_px=custom_params.get("min_size", 3.0),
            th_max_px=custom_params.get("max_size", 40.0),
            tophat_kernel=25,
        )

        # اجرای مستقیم با preset سفارشی
        from core.graft_counter import _detect_petri_dish_mask, _specular_keep_mask
        from core.graft_counter import _log_multiscale_response, _watershed_split
        from core.graft_counter import _measure_and_filter

        P = custom_preset
        dish_mask = _detect_petri_dish_mask(img_bgr)
        keep = _specular_keep_mask(img_bgr, P.spec_quantile)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        ksize = P.tophat_kernel | 1
        toph = cv2.morphologyEx(
            gray, cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        )

        logmax = _log_multiscale_response(toph, P.log_sigmas)
        cand = cv2.adaptiveThreshold(logmax, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     P.adapt_block, P.adapt_C)

        cand = cv2.bitwise_and(cand, dish_mask)
        cand = cv2.bitwise_and(cand, keep)
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 2)

        labels_ws = _watershed_split(cand, P.watershed_peak_ratio)
        num, lbl, stats, cent = cv2.connectedComponentsWithStats(labels_ws, 8)

        accepted_pts, rejected_pts = _measure_and_filter(
            lbl, stats, cent,
            P.th_min_px, P.th_max_px,
            P.elong_max, P.circ_min
        )

        # رسم
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        overlay_clean = rgb.copy()
        overlay_debug = rgb.copy()

        contours, _ = cv2.findContours(dish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_clean, contours, -1, (0, 255, 0), 3)
        cv2.drawContours(overlay_debug, contours, -1, (0, 255, 0), 3)

        for (x, y) in accepted_pts:
            cv2.circle(overlay_clean, (int(x), int(y)), 5, (255, 0, 0), -1)
            cv2.circle(overlay_debug, (int(x), int(y)), 5, (255, 0, 0), -1)

        for (x, y) in rejected_pts:
            cv2.circle(overlay_debug, (int(x), int(y)), 3, (255, 255, 0), -1)

        return {
            "count": len(accepted_pts),
            "points": np.array(accepted_pts),
            "rejected_points": np.array(rejected_pts),
            "overlay_clean": overlay_clean,
            "overlay_debug": overlay_debug,
            "preset": "custom"
        }
    else:
        # استفاده از preset معمولی
        return count_grafts(img_bgr, preset=preset)


# --- Main ---
st.markdown("### 📤 آپلود تصویر")
uploaded = st.file_uploader("تصویر ظرف پتری رو انتخاب کن", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded:
    # نمایش تصویر اصلی
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ تصویر اصلی")
        original_img = Image.open(uploaded)
        st.image(original_img, use_column_width=True)

    with col2:
        st.subheader("ℹ️ اطلاعات")
        st.info(f"""
        **نام فایل:** {uploaded.name}
        **اندازه:** {uploaded.size / 1024:.1f} KB
        **Preset انتخابی:** {preset_option}
        """)

        # دکمه تحلیل
        analyze_btn = st.button("🔍 شمارش گرافت‌ها", type="primary", use_container_width=True)

    # اگه دکمه زده شد
    if analyze_btn:
        with st.spinner("در حال تحلیل..."):
            try:
                # تبدیل به numpy
                img_pil = Image.open(uploaded).convert("RGB")
                img_rgb = np.array(img_pil)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # تحلیل
                custom_params = None
                if use_custom:
                    custom_params = {
                        "use_custom": True,
                        "adapt_block": adapt_block,
                        "adapt_C": adapt_C,
                        "watershed_ratio": watershed_ratio,
                        "min_size": min_size,
                        "max_size": max_size
                    }

                result = analyze_with_params(img_bgr, preset_option, custom_params)

                # نمایش نتیجه
                st.markdown("---")
                st.success(f"✅ **تعداد گرافت‌ها: {result['count']}**")

                # تب‌ها برای نمایش
                tab1, tab2, tab3 = st.tabs(["🎯 نتیجه نهایی", "🔍 جزئیات (Debug)", "📊 اطلاعات"])

                with tab1:
                    overlay_clean_b64 = _img_to_b64(result["overlay_clean"])
                    _show_overlay_from_b64(overlay_clean_b64, "نتیجه نهایی")

                with tab2:
                    overlay_debug_b64 = _img_to_b64(result["overlay_debug"])
                    _show_overlay_from_b64(overlay_debug_b64, "Debug (قرمز=قبول شده، زرد=رد شده)")

                    if "rejected_points" in result:
                        rejected_count = len(result["rejected_points"])
                        st.warning(f"⚠️ تعداد نقاط رد شده: {rejected_count}")

                with tab3:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("✅ قبول شده", result["count"])
                    with col_b:
                        if "rejected_points" in result:
                            st.metric("❌ رد شده", len(result["rejected_points"]))

                    st.json({
                        "preset": result.get("preset", preset_option),
                        "total_detected": result["count"] + len(result.get("rejected_points", [])),
                        "accepted": result["count"],
                        "rejected": len(result.get("rejected_points", []))
                    })

                # دکمه دانلود
                st.markdown("---")
                result_img = cv2.cvtColor(result["overlay_clean"], cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode('.jpg', result_img)
                st.download_button(
                    label="💾 دانلود تصویر نتیجه",
                    data=buf.tobytes(),
                    file_name=f"result_{uploaded.name}",
                    mime="image/jpeg"
                )

            except Exception as e:
                st.error(f"❌ خطا در پردازش: {str(e)}")
                st.exception(e)
else:
    st.info("👆 یه تصویر آپلود کن تا شروع کنیم")

    # راهنما
    with st.expander("📖 راهنمای استفاده"):
        st.markdown("""
        ### چطور استفاده کنم؟

        1. **آپلود تصویر** 📤
           - یه عکس از ظرف پتری رو انتخاب کن

        2. **انتخاب Preset** 🎯
           - **clientdemo**: برای عکس‌های معمولی
           - **ultra_dense**: برای گرافت‌های خیلی زیاد (مثل عکس تو!)
           - **qc**: برای کنترل کیفیت

        3. **تنظیمات پیشرفته** (اختیاری) 🔧
           - اگه نتیجه خوب نبود، تنظیمات رو باز کن
           - پارامترها رو تغییر بده و دوباره تست کن

        4. **شمارش** 🔍
           - دکمه "شمارش گرافت‌ها" رو بزن
           - صبر کن تا تحلیل تموم بشه

        5. **بررسی نتیجه** ✅
           - تب "نتیجه نهایی" رو ببین
           - اگه مشکل داشت، تب "Debug" رو ببین
           - نقاط زرد = رد شده (احتمالاً اشتباه تشخیص داده)

        ### نکات مهم:
        - برای عکس‌هایی مثل عکس تو، **ultra_dense** رو امتحان کن
        - اگه تعداد کم شمارش شد، slider های تنظیمات پیشرفته رو تغییر بده
        - می‌تونی نتیجه رو دانلود کنی
        """)

st.markdown("---")
st.caption("© Graft Analyzer — نسخه بهبود یافته با تنظیمات")