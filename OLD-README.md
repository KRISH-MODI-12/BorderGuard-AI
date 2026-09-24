# AI Intelligent Surveillance Platform — HTML + FastAPI

## AI model files
The backend accepts model files in either the project root or `models/`:

- `yolo26n.pt`
- `face_detection_yunet_2023mar.onnx` or `face_detection_yunet_2026may.onnx`
- `face_recognition_sface_2021dec.onnx`
- `models/license_plate_yolov8n.pt`

## Run

```bat
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`.

## AI Tools

The **AI Tools** page calls the Python backend directly:

1. YOLO — person, vehicle, bag and other COCO objects
2. YuNet — face detection
3. SFace — face recognition against `database/authorized_faces`
4. License-plate YOLO — plate detection + optional Tesseract OCR

The result image is returned to the browser so you can see the actual bounding boxes.

## Real IP cameras

Only real IPv4 addresses are accepted. The stream URL must use the same IP, for example:

`192.168.1.50` + `rtsp://192.168.1.50:554/...`

The backend attempts to open and read the actual stream before adding it. Demo values, `0`, `localhost`, filenames and mismatched hosts are rejected.

Browser camera tiles use `/ai-stream`, where FastAPI reads the RTSP/HTTP source and overlays YOLO detections.

## Railway unattended object

The demo threshold is 1–30 seconds and is applied by the backend. A short detector loss is tolerated for up to 3 seconds before the timer resets.


## AI toolchain included
- OpenCV: video/frame I/O, drawing, privacy masking and processed-video writing.
- YOLO26n: person, vehicle, bag and general object detection.
- ByteTrack via Ultralytics: stable Track IDs for video analysis.
- YuNet: face detection.
- SFace: authorized-face recognition using database/authorized_faces.
- License Plate YOLO: plate-region detection using license_plate_yolov8n.pt.
- Tesseract OCR: optional plate text extraction when installed.
- Railway unattended-object tracker: selected 1–30 second threshold with detection grace period.
- FastAPI + HTML/CSS/JavaScript: API and live analysis interface.


## Validated face-recognition integration

The demo video face pipeline has been integrated using the validated Test 14 configuration:

- YOLO26 + ByteTrack for person track IDs
- YuNet face detection inside each person ROI
- SFace native features (no upscaling)
- minimum face size 16x20
- SFace similarity threshold 0.45
- identity margin 0.05
- complete-track evidence aggregation
- strict track decision: at least 5 qualified observations, 5 votes, 70% vote ratio, average score 0.60, average margin 0.05, and 3x evidence dominance when a competing identity has evidence
- confirmed hidden faces are handled separately as the demo's explicit FACE HIDDEN rule
- UNVERIFIED is distinct from UNAUTHORIZED; it means insufficient evidence

The implementation is in `backend/ai/engine.py` and `backend/main.py`; the browser result view is updated in `frontend/js/app.js`.

See `FINAL_FACE_PIPELINE.md` for the validation notes.


## Face hidden vs verified identity
A track can first be verified as an authorized demo identity and later trigger the restricted-area hidden-face rule. The UI now keeps the last verified identity in the Identity column while showing `FACE HIDDEN` in the current Status column. A hidden-face event is a security-rule alert; it does not erase the earlier identity evidence.

## Alarm sound
The backend exposes `GET /api/alarm` for `alarm/alarm.mp3`. On the Demo Videos page, click **Enable Alarm Sound** once before starting analysis. When the backend raises a critical scenario alarm (for example, a confirmed hidden face or unattended-object threshold), the frontend plays the alarm once for that analysis job.

## Analysis status/API fix
The current build publishes progress before expensive AI inference, sanitizes NumPy/Pydantic values for status JSON, removes live JPEG bytes from JSON snapshots, and retries transient status polling errors in the browser. If the processed video is slow, the status stage remains visible instead of appearing stuck at 0%.


IMPORTANT DEMO RUN MODE
========================
Run FastAPI WITHOUT Uvicorn --reload during video analysis. Auto-reload can restart the Python process when files in the project change, clearing in-memory analysis jobs and causing the browser to show “Job not found” or appear stuck at a low percentage. Use:

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

or double-click RUN_SERVER.bat.
