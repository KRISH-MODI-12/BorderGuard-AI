from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
print("PROJECT:", BASE)
print()

files = [
    "yolo26n.pt",
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2026may.onnx",
    "face_recognition_sface_2021dec.onnx",
    "license_plate_yolov8n.pt",
]

for name in files:
    p1 = BASE / name
    p2 = BASE / "models" / name
    print(f"{name}")
    print("  root  :", "FOUND" if p1.exists() else "-")
    print("  models:", "FOUND" if p2.exists() else "-")

print()
try:
    import cv2
    print("OpenCV:", cv2.__version__)
    print("FaceDetectorYN:", hasattr(cv2, "FaceDetectorYN"))
    print("FaceRecognizerSF:", hasattr(cv2, "FaceRecognizerSF"))
except Exception as e:
    print("OpenCV ERROR:", e)

try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch ERROR:", e)

try:
    import ultralytics
    print("Ultralytics:", ultralytics.__version__)
except Exception as e:
    print("Ultralytics ERROR:", e)

try:
    import pytesseract
    print("pytesseract: installed")
    try:
        print("Tesseract:", pytesseract.get_tesseract_version())
    except Exception as e:
        print("Tesseract executable ERROR:", e)
except Exception as e:
    print("pytesseract ERROR:", e)
