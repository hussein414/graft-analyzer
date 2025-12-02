# api/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

from core.pipeline import analyze_bytes   # نسخهٔ اتوماتیک؛ پارامتری نمی‌گیرد
from core.ml_infer import detect_accelerators

app = FastAPI(title="Graft Analyzer API", version="1.0.0")

# CORS برای توسعه لوکال (Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # در پروDUCTION دامنه‌های مجاز را محدود کن
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CountResponse(BaseModel):
    count: int
    centers: list
    boxes: list
    chosen: str
    overlay_b64: str
    overlay_path: str

@app.get("/health")
def health() -> Dict:
    return {
        "status": "ok",
        "api": "graft-analyzer",
        "version": "1.0.0",
        "accelerators": detect_accelerators(),
    }

@app.post("/count", response_model=CountResponse)
async def count_grafts(file: UploadFile = File(...)) -> Dict:
    """
    ورودی: فقط فایل تصویر.
    خروجی: count و تصویر Annotated به‌صورت Base64 + مسیر ذخیره‌شده.
    """
    data = await file.read()
    result = analyze_bytes(data)  # همه‌چیز داخل pipeline اتوماتیکه
    return result
