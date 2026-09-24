from pathlib import Path
import base64
import re
import os
import shutil

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import pytesseract
except Exception:
    pytesseract = None

BASE = Path(__file__).resolve().parents[2]
MODELS = BASE / "models"
AUTH_FACES = BASE / "database" / "authorized_faces"
AUTH_PLATES = BASE / "database" / "authorized_plates"

# Accept models whether the user stores them in the project root or models/.
# Also search common alternate filenames so the AI tools do not silently report MISSING.
def first_existing(*paths):
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None

YOLO_PATH = first_existing(
    BASE / "yolo26n.pt",
    MODELS / "yolo26n.pt",
)

PLATE_PATH = first_existing(
    BASE / "license_plate_yolov8n.pt",
    MODELS / "license_plate_yolov8n.pt",
    BASE / "license-plate-finetune-v1n.pt",
    MODELS / "license-plate-finetune-v1n.pt",
)

# Prefer the 2023 YuNet model, but accept the user's 2026 model as fallback.
YUNET_PATH = first_existing(
    BASE / "face_detection_yunet_2023mar.onnx",
    MODELS / "face_detection_yunet_2023mar.onnx",
    BASE / "face_detection_yunet_2026may.onnx",
    MODELS / "face_detection_yunet_2026may.onnx",
)

SFACE_PATH = first_existing(
    BASE / "face_recognition_sface_2021dec.onnx",
    MODELS / "face_recognition_sface_2021dec.onnx",
)

PERSON = 0
VEHICLE_IDS = {2, 3, 5, 7}
BAG_IDS = {24, 26, 28}

# Validated in Tests 10-14 on the supplied demo video.
FACE_MATCH_THRESHOLD = 0.45
FACE_IDENTITY_MARGIN = 0.05
FACE_MIN_W = 16
FACE_MIN_H = 20

# Strict track-level validation rules from Test 14.
STRICT_MIN_QUALIFIED = 5
STRICT_MIN_VOTES = 5
STRICT_MIN_VOTE_RATIO = 0.70
STRICT_MIN_AVG_SCORE = 0.60
STRICT_MIN_AVG_MARGIN = 0.05
STRICT_EVIDENCE_DOMINANCE = 3.0

class AIEngine:
    def __init__(self):
        self.yolo = None
        self.plate = None
        self.face_detector = None
        self.face_recognizer = None
        self.load_errors = {}
        self.device = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
        self._face_db_cache = None
        self._face_db_mtime = None
        self._plate_db_cache = None
        self._plate_db_signature_value = None
        self._load()
        self.configure_tesseract()

    def _load(self):
        if YOLO is None:
            self.load_errors["YOLO"] = "ultralytics is not installed in this Python environment."
        elif YOLO_PATH is not None:
            try:
                self.yolo = YOLO(str(YOLO_PATH))
                if self.device.startswith("cuda"):
                    self.yolo.to(self.device)
            except Exception as e:
                self.load_errors["YOLO"] = f"{type(e).__name__}: {e}"
        else:
            self.load_errors["YOLO"] = "yolo26n.pt not found in project root or models/."

        if YOLO is None:
            self.load_errors["Plate"] = "ultralytics is not installed."
        elif PLATE_PATH is not None:
            try:
                self.plate = YOLO(str(PLATE_PATH))
                if self.device.startswith("cuda"):
                    self.plate.to(self.device)
            except Exception as e:
                self.load_errors["Plate"] = f"{type(e).__name__}: {e}"
        else:
            self.load_errors["Plate"] = "license_plate_yolov8n.pt not found in project root or models/."

        if not hasattr(cv2, "FaceDetectorYN"):
            self.load_errors["YuNet"] = "This OpenCV build does not expose FaceDetectorYN."
        elif YUNET_PATH is not None:
            try:
                self.face_detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.70, 0.30, 5000)
            except Exception as e:
                self.load_errors["YuNet"] = f"{type(e).__name__}: {e}"
        else:
            self.load_errors["YuNet"] = "YuNet ONNX model not found in project root or models/."

        if not hasattr(cv2, "FaceRecognizerSF"):
            self.load_errors["SFace"] = "This OpenCV build does not expose FaceRecognizerSF."
        elif SFACE_PATH is not None:
            try:
                self.face_recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
            except Exception as e:
                self.load_errors["SFace"] = f"{type(e).__name__}: {e}"
        else:
            self.load_errors["SFace"] = "SFace ONNX model not found in project root or models/."

    @property
    def ocr_available(self):
        if pytesseract is None:
            return False
        try:
            _ = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def configure_tesseract(self):
        """Find the Tesseract executable on Windows if it is installed."""
        if pytesseract is None:
            return False

        candidates = [
            os.environ.get("TESSERACT_CMD"),
            shutil.which("tesseract"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                pytesseract.pytesseract.tesseract_cmd = str(candidate)
                try:
                    pytesseract.get_tesseract_version()
                    return True
                except Exception:
                    pass
        return False

    def status(self):
        cuda = bool(torch is not None and torch.cuda.is_available())
        return {
            "yolo_loaded": self.yolo is not None,
            "plate_loaded": self.plate is not None,
            "yunet_loaded": self.face_detector is not None,
            "sface_loaded": self.face_recognizer is not None,
            "ocr_loaded": self.ocr_available,
            "cuda": cuda,
            "device": torch.cuda.get_device_name(0) if cuda else "CPU",
            "paths": {
                "YOLO": str(YOLO_PATH),
                "Plate": str(PLATE_PATH),
                "AuthorizedPlates": sorted(self.authorized_plates()),
                "YuNet": str(YUNET_PATH),
                "SFace": str(SFACE_PATH),
            },
            "face_recognition_config": {
                "min_face": [FACE_MIN_W, FACE_MIN_H],
                "threshold": FACE_MATCH_THRESHOLD,
                "identity_margin": FACE_IDENTITY_MARGIN,
                "strict_min_qualified": STRICT_MIN_QUALIFIED,
                "strict_min_votes": STRICT_MIN_VOTES,
                "strict_vote_ratio": STRICT_MIN_VOTE_RATIO,
                "strict_avg_score": STRICT_MIN_AVG_SCORE,
                "strict_avg_margin": STRICT_MIN_AVG_MARGIN,
                "strict_evidence_dominance": STRICT_EVIDENCE_DOMINANCE,
            },
            "errors": dict(self.load_errors),
        }

    def detect(self, frame, confidence=0.20, track=False):
        if self.yolo is None:
            raise RuntimeError(self.load_errors.get("YOLO", "YOLO model is unavailable."))
        kwargs = dict(conf=float(confidence), iou=0.50, imgsz=640, verbose=False)
        if track:
            result = self.yolo.track(source=frame, persist=True, tracker="bytetrack.yaml", **kwargs)[0]
        else:
            result = self.yolo.predict(source=frame, **kwargs)[0]
        out = []
        names = result.names
        if result.boxes is None:
            return out
        ids = result.boxes.id
        for i, b in enumerate(result.boxes):
            cls = int(b.cls[0]); conf = float(b.conf[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
            tid = None
            if ids is not None:
                try: tid = int(ids[i].item())
                except Exception: pass
            name = str(names.get(cls, cls))
            group = "person" if cls == PERSON else ("vehicle" if cls in VEHICLE_IDS else ("bag" if cls in BAG_IDS else "object"))
            out.append({"class_id": cls, "name": name, "confidence": round(conf, 3), "box": [x1, y1, x2, y2], "group": group, "track_id": tid})
        return out

    @staticmethod
    def _draw(frame, detections, color_by_group=True):
        out = frame.copy()
        colors = {"person": (67, 211, 158), "vehicle": (53, 169, 255), "bag": (255, 174, 82), "object": (180, 150, 255)}
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            c = colors.get(d["group"], (255, 255, 255)) if color_by_group else (0, 220, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
            tid = f" ID:{d['track_id']}" if d.get("track_id") is not None else ""
            label = f"{d['name']} {d['confidence']:.0%}{tid}"
            cv2.rectangle(out, (x1, max(0, y1 - 25)), (min(out.shape[1], x1 + max(120, len(label) * 8)), y1), c, -1)
            cv2.putText(out, label, (x1 + 5, max(17, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .50, (10, 20, 30), 2)
        return out

    def detect_annotated(self, frame, confidence=0.20, track=False):
        detections = self.detect(frame, confidence, track=track)
        return self._draw(frame, detections), detections

    def faces(self, frame, min_face_w=FACE_MIN_W, min_face_h=FACE_MIN_H, score_threshold=0.0):
        """Detect faces and optionally filter by minimum pixel size.

        Production validation showed that faces smaller than 40x40 can still
        produce usable SFace evidence, so the default gate is 16x20.
        """
        if self.face_detector is None:
            raise RuntimeError(self.load_errors.get("YuNet", "YuNet is unavailable."))
        h, w = frame.shape[:2]
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(frame)
        out = []
        if faces is None:
            return out
        for f in faces:
            x, y, fw, fh = [int(v) for v in f[:4]]
            score = float(f[-1])
            if fw < int(min_face_w) or fh < int(min_face_h) or score < float(score_threshold):
                continue
            x = max(0, x); y = max(0, y); x2 = min(w, x + fw); y2 = min(h, y + fh)
            clipped = np.asarray(f, dtype=np.float32).copy()
            clipped[0] = x; clipped[1] = y; clipped[2] = max(0, x2 - x); clipped[3] = max(0, y2 - y)
            out.append({"box": [x, y, x2, y2], "center": [x + fw / 2, y + fh / 2], "score": round(score, 3), "raw": clipped})
        return out

    def face_annotated(self, frame):
        faces = self.faces(frame)
        out = frame.copy()
        for f in faces:
            x1, y1, x2, y2 = f["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (67, 211, 158), 2)
            cv2.putText(out, f"FACE {f['score']:.0%}", (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, (67,211,158), 2)
        return out, faces

    def _identity_from_filename(self, p):
        """Group multi-angle samples by their stable AUTH-### identity prefix."""
        stem = p.stem.strip().upper()
        match = re.match(r"^(AUTH-\d+)", stem)
        if match:
            return match.group(1)
        return stem

    @staticmethod
    def _cosine_similarity(a, b):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _face_db(self):
        """Build/cache SFace features using the same native feature path used in validation tests."""
        if not AUTH_FACES.exists():
            return {}
        files = sorted(
            p for p in AUTH_FACES.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        mtime = max((p.stat().st_mtime_ns for p in files), default=0)
        cache_key = (mtime, len(files))
        if self._face_db_cache is not None and self._face_db_mtime == cache_key:
            return self._face_db_cache
        db = {}
        if self.face_detector is None or self.face_recognizer is None:
            return db
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            try:
                h, w = img.shape[:2]
                self.face_detector.setInputSize((w, h))
                _, detected = self.face_detector.detect(img)
                if detected is None or len(detected) == 0:
                    continue
                face = max(detected, key=lambda f: f[2] * f[3])
                if float(face[2]) < 20 or float(face[3]) < 20:
                    continue
                feature = self.face_recognizer.feature(img, face)
                db.setdefault(self._identity_from_filename(p), []).append({
                    "path": p.name,
                    "feature": feature,
                })
            except Exception:
                continue
        self._face_db_cache, self._face_db_mtime = db, cache_key
        return db

    def _compare_feature(self, feature, db):
        identity_scores = {}
        for identity, samples in db.items():
            scores = [self._cosine_similarity(feature, sample["feature"]) for sample in samples]
            if scores:
                identity_scores[identity] = max(scores)
        if not identity_scores:
            return None
        ordered = sorted(identity_scores.items(), key=lambda x: x[1], reverse=True)
        best_identity, best_score = ordered[0]
        if len(ordered) >= 2:
            second_identity, second_score = ordered[1]
            margin = float(best_score - second_score)
        else:
            second_identity, second_score = "NONE", 0.0
            margin = float(best_score)
        if best_score < FACE_MATCH_THRESHOLD:
            decision = "BELOW_THRESHOLD"
        elif margin < FACE_IDENTITY_MARGIN:
            decision = "AMBIGUOUS"
        else:
            decision = "QUALIFIED"
        return {
            "best_identity": best_identity,
            "best_score": float(best_score),
            "second_identity": second_identity,
            "second_score": float(second_score),
            "margin": margin,
            "decision": decision,
            "identity_scores": identity_scores,
        }

    def recognize_person_roi(self, person_roi, threshold=FACE_MATCH_THRESHOLD, margin_threshold=FACE_IDENTITY_MARGIN, min_face_w=FACE_MIN_W, min_face_h=FACE_MIN_H):
        """Run YuNet + SFace inside ONE YOLO person ROI.

        This is the validated association path: the face belongs directly to the
        person Track ID that supplied the ROI. No global nearest-person guess is needed.
        """
        if self.face_detector is None:
            raise RuntimeError(self.load_errors.get("YuNet", "YuNet is unavailable."))
        if self.face_recognizer is None:
            raise RuntimeError(self.load_errors.get("SFace", "SFace is unavailable."))
        db = self._face_db()
        if not db:
            return {"has_face": False, "reason": "EMPTY_FACE_DATABASE"}
        h, w = person_roi.shape[:2]
        if w < 20 or h < 60:
            return {"has_face": False, "reason": "ROI_TOO_SMALL"}
        self.face_detector.setInputSize((w, h))
        _, detected = self.face_detector.detect(person_roi)
        if detected is None or len(detected) == 0:
            return {"has_face": False, "reason": "NO_FACE"}
        face = max(detected, key=lambda f: f[2] * f[3])
        fw, fh = float(face[2]), float(face[3])
        face_box = [int(face[0]), int(face[1]), int(face[0] + face[2]), int(face[1] + face[3])]
        if fw < float(min_face_w) or fh < float(min_face_h):
            return {"has_face": True, "reason": "FACE_TOO_SMALL", "face_box": face_box, "face_w": fw, "face_h": fh, "face_score": float(face[-1]), "raw_face": face}
        try:
            feature = self.face_recognizer.feature(person_roi, face)
        except Exception as exc:
            return {"has_face": True, "reason": "SFACE_ERROR", "error": str(exc), "face_box": face_box, "face_w": fw, "face_h": fh, "face_score": float(face[-1]), "raw_face": face}
        # Use the same native cosine comparison as Tests 9-14.
        previous_threshold = FACE_MATCH_THRESHOLD
        previous_margin = FACE_IDENTITY_MARGIN
        comparison = self._compare_feature(feature, db)
        # Per-call thresholds are kept for API consistency; validated defaults remain 0.45 / 0.05.
        if comparison is not None:
            best_score = comparison["best_score"]
            margin = comparison["margin"]
            if best_score < float(threshold):
                comparison["decision"] = "BELOW_THRESHOLD"
            elif margin < float(margin_threshold):
                comparison["decision"] = "AMBIGUOUS"
            else:
                comparison["decision"] = "QUALIFIED"
        return {
            "has_face": True,
            "reason": comparison["decision"] if comparison else "NO_DATABASE_MATCH",
            "face_box": face_box,
            "face_w": fw,
            "face_h": fh,
            "face_score": float(face[-1]),
            "raw_face": face,
            **(comparison or {}),
        }

    @staticmethod
    def strict_track_decision(track_state):
        """Return Test-14-style strict decision from complete-track evidence."""
        qualified = int(track_state.get("qualified", 0))
        scores_by_id = track_state.get("scores", {})
        margins_by_id = track_state.get("margins", {})
        evidence = []
        for identity, scores in scores_by_id.items():
            if not scores:
                continue
            margins = margins_by_id.get(identity, [])
            ev = sum(
                max(float(score) - FACE_MATCH_THRESHOLD, 0.0) * max(float(m), 0.0)
                for score, m in zip(scores, margins)
            )
            evidence.append({
                "identity": identity,
                "votes": len(scores),
                "avg_score": float(np.mean(scores)),
                "best_score": float(np.max(scores)),
                "avg_margin": float(np.mean(margins)) if margins else 0.0,
                "best_margin": float(np.max(margins)) if margins else 0.0,
                "evidence": float(ev),
            })
        evidence.sort(key=lambda x: x["evidence"], reverse=True)
        if not evidence:
            return {"status": "UNVERIFIED", "identity": "UNKNOWN", "evidence": [], "evidence_gap": 0.0, "strict_checks": {}}
        winner = evidence[0]
        runner = evidence[1] if len(evidence) >= 2 else None
        vote_ratio = winner["votes"] / max(qualified, 1)
        evidence_ratio = None if not runner or runner["evidence"] <= 0 else winner["evidence"] / runner["evidence"]
        competition_ok = runner is None or runner["evidence"] <= 0 or evidence_ratio >= STRICT_EVIDENCE_DOMINANCE
        checks = {
            "min_qualified": qualified >= STRICT_MIN_QUALIFIED,
            "min_votes": winner["votes"] >= STRICT_MIN_VOTES,
            "vote_ratio": vote_ratio >= STRICT_MIN_VOTE_RATIO,
            "avg_score": winner["avg_score"] >= STRICT_MIN_AVG_SCORE,
            "avg_margin": winner["avg_margin"] >= STRICT_MIN_AVG_MARGIN,
            "evidence_dominance": competition_ok,
        }
        if all(checks.values()):
            status = "AUTHORIZED"
        elif runner and not competition_ok and all(checks[k] for k in ("min_qualified", "min_votes", "vote_ratio", "avg_score", "avg_margin")):
            status = "AMBIGUOUS"
        else:
            status = "UNVERIFIED"
        evidence_gap = float(winner["evidence"] - (runner["evidence"] if runner else 0.0))
        return {
            "status": status,
            "identity": winner["identity"] if status == "AUTHORIZED" else winner["identity"],
            "winner": winner,
            "runner_up": runner,
            "evidence": evidence,
            "evidence_gap": evidence_gap,
            "evidence_ratio": evidence_ratio,
            "vote_ratio": vote_ratio,
            "strict_checks": checks,
        }

    def recognize(self, frame, threshold=FACE_MATCH_THRESHOLD, margin_threshold=FACE_IDENTITY_MARGIN, min_face_w=FACE_MIN_W, min_face_h=FACE_MIN_H):
        """Single-image recognition tool using the same face/feature path as validation."""
        faces = self.faces(frame, min_face_w=min_face_w, min_face_h=min_face_h)
        if self.face_recognizer is None:
            raise RuntimeError(self.load_errors.get("SFace", "SFace is unavailable."))
        db = self._face_db()
        matches = []
        annotated = frame.copy()
        for f in faces:
            try:
                feature = self.face_recognizer.feature(frame, f["raw"])
                cmp = self._compare_feature(feature, db)
            except Exception:
                cmp = None
            if cmp is None:
                best_id = "UNKNOWN"; best = 0.0; second_id = "NONE"; second = 0.0; margin = 0.0; decision = "UNVERIFIED"
            else:
                best_id = cmp["best_identity"]; best = cmp["best_score"]; second_id = cmp["second_identity"]; second = cmp["second_score"]; margin = cmp["margin"]
                if best < float(threshold):
                    decision = "BELOW_THRESHOLD"
                elif margin < float(margin_threshold):
                    decision = "AMBIGUOUS"
                else:
                    decision = "QUALIFIED"
            authorized = decision == "QUALIFIED"
            x1, y1, x2, y2 = f["box"]
            color = (67,211,158) if authorized else (60,60,255)
            label = f"AUTHORIZED • {best_id}" if authorized else decision
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 3)
            cv2.putText(annotated, f"{label} {best:.2f}", (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            matches.append({
                "box":f["box"],
                "identity":best_id,
                "score":round(best,3),
                "second_identity":second_id,
                "second_score":round(second,3),
                "margin":round(margin,3),
                "decision":decision,
                "authorized":authorized,
            })
        return {
            "available": True,
            "faces": len(faces),
            "matches": matches,
            "threshold": threshold,
            "margin_threshold": margin_threshold,
            "min_face": [min_face_w, min_face_h],
            "authorized_database_identities": sorted(db.keys()),
            "image": encode_image(annotated),
        }

    @staticmethod
    def normalize_plate(text):
        """Normalize OCR/database plate text for exact comparison.

        Spaces, dashes and punctuation are removed. We deliberately do not
        auto-correct ambiguous OCR characters (O/0, I/1, B/8, etc.) because
        that could turn an OCR mistake into an incorrect authorization.
        """
        if text is None:
            return ""
        return "".join(ch for ch in str(text).upper() if ch.isalnum())

    @staticmethod
    def _looks_like_plate(text):
        text = AIEngine.normalize_plate(text)
        if len(text) < 5:
            return False
        letters = sum(ch.isalpha() for ch in text)
        digits = sum(ch.isdigit() for ch in text)
        return letters >= 2 and digits >= 2

    def _get_plate_db_signature(self):
        try:
            entries = []
            for p in sorted(AUTH_PLATES.iterdir()):
                if p.is_file():
                    st = p.stat()
                    entries.append((p.name, st.st_mtime_ns, st.st_size))
            return tuple(entries)
        except Exception:
            return ()

    def _ocr_single_image(self, image):
        """OCR helper used for authorized-plate image records."""
        if pytesseract is None or not self.ocr_available or image is None or image.size == 0:
            return ""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            variants = [
                gray,
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11),
            ]
            candidates = []
            for img in variants:
                text = pytesseract.image_to_string(
                    img,
                    config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ).strip()
                norm = self.normalize_plate(text)
                if self._looks_like_plate(norm):
                    candidates.append(norm)
            if not candidates:
                return ""
            return max(candidates, key=lambda x: len(x))
        except Exception:
            return ""

    def authorized_plates(self):
        """Read the authorized plate database from database/authorized_plates.

        Supported records:
          * plate numbers in .txt files, one per line
          * plate numbers in .csv files (first field per row)
          * .json files containing plate strings
          * image files whose filename is the plate number
          * image files with non-plate filenames are OCR'd once and cached
        """
        sig = self._get_plate_db_signature()
        if self._plate_db_cache is not None and sig == self._plate_db_signature_value:
            return set(self._plate_db_cache)

        db = set()
        try:
            AUTH_PLATES.mkdir(parents=True, exist_ok=True)
            for p in sorted(AUTH_PLATES.iterdir()):
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext == ".txt":
                    try:
                        for line in p.read_text(errors="ignore").splitlines():
                            n = self.normalize_plate(line)
                            if self._looks_like_plate(n):
                                db.add(n)
                    except Exception:
                        pass
                elif ext == ".csv":
                    try:
                        for line in p.read_text(errors="ignore").splitlines():
                            parts = [x.strip() for x in line.split(",")]
                            for value in parts:
                                n = self.normalize_plate(value)
                                if self._looks_like_plate(n):
                                    db.add(n)
                    except Exception:
                        pass
                elif ext == ".json":
                    try:
                        import json
                        data = json.loads(p.read_text(errors="ignore"))
                        def walk(v):
                            if isinstance(v, str):
                                n = self.normalize_plate(v)
                                if self._looks_like_plate(n):
                                    db.add(n)
                            elif isinstance(v, dict):
                                for vv in v.values():
                                    walk(vv)
                            elif isinstance(v, list):
                                for vv in v:
                                    walk(vv)
                        walk(data)
                    except Exception:
                        pass
                elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    stem = self.normalize_plate(p.stem)
                    if self._looks_like_plate(stem):
                        db.add(stem)
                    else:
                        img = cv2.imread(str(p))
                        ocr = self._ocr_single_image(img)
                        if ocr:
                            db.add(ocr)
        except Exception:
            pass

        self._plate_db_cache = sorted(db)
        self._plate_db_signature_value = sig
        return set(self._plate_db_cache)

    def plate_verify(self, frame, confidence=0.25):
        """Detect plates, OCR each plate and compare OCR text to authorized DB."""
        plates = self.plate_boxes(frame, confidence)
        texts, error = self.plate_ocr(frame, plates)
        db = self.authorized_plates()
        for p in plates:
            raw = p.get("ocr_text", "")
            norm = self.normalize_plate(raw)
            p["normalized_plate"] = norm
            p["authorized"] = bool(norm and norm in db)
            if not norm:
                p["status"] = "OCR UNREADABLE"
            elif p["authorized"]:
                p["status"] = "AUTHORIZED VEHICLE"
            else:
                p["status"] = "UNRECOGNIZED VEHICLE"
        return plates, texts, error, sorted(db)

    def plate_boxes(self, frame, confidence=0.25):
        if self.plate is None:
            raise RuntimeError(self.load_errors.get("Plate", "Plate model is unavailable."))
        result = self.plate.predict(source=frame, conf=float(confidence), iou=.50, imgsz=640, verbose=False)[0]
        out=[]
        if result.boxes is None: return out
        for b in result.boxes:
            x1,y1,x2,y2=[int(v) for v in b.xyxy[0]]
            out.append({"box":[x1,y1,x2,y2],"confidence":round(float(b.conf[0]),3),"class_id":int(b.cls[0])})
        return out

    def plate_ocr(self, frame, plates):
        results=[]
        if pytesseract is None:
            return results, "pytesseract is not installed in this Python environment."
        if not self.ocr_available:
            return results, "Tesseract OCR executable was not found. Install Tesseract OCR or set TESSERACT_CMD."

        for p in plates:
            x1,y1,x2,y2=p["box"]
            crop=frame[max(0,y1):min(frame.shape[0],y2),max(0,x1):min(frame.shape[1],x2)]
            if crop.size==0:
                continue

            gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            gray=cv2.resize(gray,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)

            variants=[
                gray,
                cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11),
            ]

            candidates=[]
            for variant in variants:
                try:
                    data=pytesseract.image_to_data(
                        variant,
                        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        output_type=pytesseract.Output.DICT,
                    )
                    pieces=[]; confs=[]
                    for txt, conf in zip(data.get('text',[]), data.get('conf',[])):
                        norm=self.normalize_plate(txt)
                        if norm:
                            pieces.append(norm)
                            try:
                                value=float(conf)
                                if value >= 0:
                                    confs.append(value)
                            except Exception:
                                pass
                    text="".join(pieces)
                    if not text:
                        raw=pytesseract.image_to_string(
                            variant,
                            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                        ).strip()
                        text=self.normalize_plate(raw)
                    avg_conf=float(np.mean(confs)) if confs else 0.0
                    if text:
                        candidates.append((text,avg_conf))
                except Exception:
                    continue

            if candidates:
                # Prefer plate-like strings, then confidence, then length.
                candidates.sort(key=lambda item: (self._looks_like_plate(item[0]), item[1], len(item[0])), reverse=True)
                best_text,best_conf=candidates[0]
                p["ocr_text"]=best_text
                p["ocr_confidence"]=round(best_conf,1)
                results.append(best_text)
            else:
                p["ocr_text"]=""
                p["ocr_confidence"]=0.0

        return results, None

    def plate_annotated(self, frame, confidence=.25):
        plates, texts, error, authorized_db = self.plate_verify(frame, confidence)
        out=frame.copy()
        for p in plates:
            x1,y1,x2,y2=p["box"]
            status=p.get("status", "OCR UNREADABLE")
            if status == "AUTHORIZED VEHICLE":
                color=(67,211,158)
            elif status == "UNRECOGNIZED VEHICLE":
                color=(60,60,255)
            else:
                color=(245,184,61)
            label=f"PLATE {p['confidence']:.0%}"
            if p.get("ocr_text"):
                label += f" • {p['ocr_text']}"
            label += f" • {status}"
            cv2.rectangle(out,(x1,y1),(x2,y2),color,3)
            cv2.rectangle(out,(x1,max(0,y1-28)),(min(out.shape[1],x1+max(260,len(label)*8)),y1),color,-1)
            cv2.putText(out,label,(x1+5,max(19,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,(10,20,30),2,cv2.LINE_AA)
        return out,plates,texts,error

def encode_image(frame):
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None

engine = AIEngine()

def summarize(frame, yolo_conf=.20, plate_conf=.25):
    detections=engine.detect(frame,yolo_conf)
    faces=engine.faces(frame)
    plates=engine.plate_boxes(frame,plate_conf)
    return {
        "people":sum(d["group"]=="person" for d in detections),
        "vehicles":sum(d["group"]=="vehicle" for d in detections),
        "bags":sum(d["group"]=="bag" for d in detections),
        "objects":sum(d["group"]=="object" for d in detections),
        "faces":len(faces),"plates":len(plates),
        "detections":detections,"faces_data":[{k:v for k,v in f.items() if k!="raw"} for f in faces],"plates_data":plates,
        "models":engine.status()
    }
