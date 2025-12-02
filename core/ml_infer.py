# core/ml_infer.py
import cv2
import platform

def detect_accelerators():
    info = {}
    # OpenCV CUDA (اختیاری)
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        info["opencv_cuda_devices"] = int(count)
    except Exception:
        info["opencv_cuda_devices"] = 0
    # نسخه OpenCV/OS
    info["opencv_version"] = cv2.__version__
    info["platform"] = platform.platform()
    # پیام دوستانه
    if info["opencv_cuda_devices"] > 0:
        info["summary"] = "CUDA available"
    else:
        info["summary"] = "CPU mode"
    return info
