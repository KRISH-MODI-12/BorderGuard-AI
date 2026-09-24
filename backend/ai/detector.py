from pathlib import Path
from typing import Any
import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

BASE_DIR = Path(__file__).resolve().parents[2]
YOLO_PATH = BASE_DIR / "models" / "yolo26n.pt"
PLATE_PATH = BASE_DIR / "models" / "license_plate_yolov8n.pt"

PERSON = 0
VEHICLE_IDS = {2, 3, 5, 7}
BAG_IDS = {24, 26, 28}

_vehicle_model = None
_plate_model = None

def _load(path: Path):
    if YOLO is None or not path.exists():
        return None
    try:
        return YOLO(str(path))
    except Exception:
        return None

def vehicle_model():
    global _vehicle_model
    if _vehicle_model is None:
        _vehicle_model = _load(YOLO_PATH)
    return _vehicle_model

def plate_model():
    global _plate_model
    if _plate_model is None:
        _plate_model = _load(PLATE_PATH)
    return _plate_model

def detect_frame(frame, confidence=0.35):
    model = vehicle_model()
    if model is None:
        return {"people": 0, "vehicles": 0, "bags": 0, "detections": [], "model_ready": False}
    result = model.predict(frame, conf=confidence, imgsz=640, verbose=False)[0]
    detections = []
    people = vehicles = bags = 0
    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = str(names.get(cls, cls))
            if cls == PERSON:
                people += 1
            elif cls in VEHICLE_IDS:
                vehicles += 1
            elif cls in BAG_IDS:
                bags += 1
            detections.append({"class_id": cls, "label": label, "confidence": round(conf, 3), "box": [x1, y1, x2, y2]})
    return {"people": people, "vehicles": vehicles, "bags": bags, "detections": detections, "model_ready": True}

def detect_plates(frame, confidence=0.25):
    model = plate_model()
    if model is None:
        return {"plates": [], "model_ready": False}
    result = model.predict(frame, conf=confidence, imgsz=640, verbose=False)[0]
    plates = []
    if result.boxes is not None:
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            text = ""
            if crop.size:
                try:
                    import pytesseract
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    text = pytesseract.image_to_string(gray, config="--psm 7").strip().replace(" ", "")
                except Exception:
                    pass
            plates.append({"confidence": round(conf, 3), "box": [x1, y1, x2, y2], "text": text})
    return {"plates": plates, "model_ready": True}
