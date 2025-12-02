# app/app.py
import os, sys, io, base64, json
from typing import Dict
import streamlit as st
from PIL import Image

# import core/*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.pipeline import analyze_bytes
from core.ml_infer import detect_accelerators

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

st.set_page_config(page_title="Graft Analyzer — ClientDemo", layout="wide")
st.title("🧪 Graft Analyzer — ClientDemo")
st.caption("شمارش خودکار گرافت‌ها (Recall بالا، جداسازی چسبیده‌ها).")

# --- Sidebar ---
run_mode = st.sidebar.radio("Run Mode", options=["Local (in-app)", "Via API"], index=0)
api_url = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000")
st.sidebar.markdown("---")
st.sidebar.subheader("Hardware")
st.sidebar.write(detect_accelerators())

# --- Helpers ---
def _show_overlay_from_b64(b64: str, caption: str):
    if not b64:
        st.warning("خروجی تصویری موجود نیست.")
        return
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    st.image(img, caption=caption, use_column_width=True)

def _show_result(res: Dict):
    st.success(f"✅ Count: {res.get('count','-')}  |  Method: {res.get('chosen','-')}")
    tabs = st.tabs(["Overlay", "Debug", "Details"])
    with tabs[0]:
        _show_overlay_from_b64(res.get("overlay_b64",""), "Overlay (ClientDemo)")
    with tabs[1]:
        _show_overlay_from_b64(res.get("overlay_debug_b64",""), "Debug (Accepted=Red, Rejected=Yellow)")
    with tabs[2]:
        st.json({
            "count": res.get("count"),
            "chosen": res.get("chosen"),
            "centers (first 10)": (res.get("centers") or [])[:10],
            "overlay_path": res.get("overlay_path","")
        })

# --- Main ---
colL, colR = st.columns([1,1])
with colL:
    up = st.file_uploader("تصویر را آپلود کن", type=["jpg","jpeg","png","bmp","tif","tiff"])
    run_btn = st.button("Count Grafts", type="primary", use_container_width=True)
with colR:
    st.info("راهنما:\n- تصویر عمود از بالای ظرف\n- نور یکنواخت بدون انعکاس\n- پس‌زمینه‌ی تمیز")

placeholder = st.empty()

if run_btn:
    if up is None:
        st.warning("ابتدا فایل تصویر را انتخاب کن.")
    else:
        try:
            if run_mode == "Local (in-app)":
                data = up.getvalue()  # مهم: به‌جای up.read()
                if not data:
                    st.error("فایل خالی است یا دوباره بارگذاری کن.")
                else:
                    res = analyze_bytes(data)
                    with placeholder.container():
                        _show_result(res)
            else:
                if not HAS_REQUESTS:
                    st.error("کتابخانه requests نصب نیست. `pip install requests`")
                else:
                    files = {"file": (up.name, up.getvalue(), up.type or "application/octet-stream")}
                    resp = requests.post(f"{api_url}/count", files=files, timeout=120)
                    if resp.status_code == 200:
                        res = resp.json()
                        with placeholder.container():
                            _show_result(res)
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.exception(e)

st.markdown("---")
st.caption("© graft-analyzer — ClientDemo preset.")
