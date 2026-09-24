# AI Security Command Center — HTML/CSS/JavaScript + FastAPI

This is the HTML adaptation of the supplied `frontend_v2_solved.py` architecture. It keeps the same top navigation and major sections while moving the UI out of Streamlit.

## Run

```bat
cd "C:\Users\KRISH\OneDrive\Desktop\mystreamlitapp\New folder\surveillance_html"
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Open Chrome: `http://127.0.0.1:8000`

## Put your existing models/data

Copy these into this project's root where available:

- `yolo26n.pt`
- `yolo26n.engine` (optional)
- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`
- `models/license_plate_yolov8n.pt`
- `alarm/alarm.mp3`
- `database/authorized_faces/*`
- `database/authorized_plates/*`
- `database/demo_videos/*`

## Architecture

Browser native video/UI → FastAPI → Python AI modules → database.

The browser does not refresh the whole page for every video frame, which avoids the Streamlit `st.image()` per-frame bottleneck.

## Important

The frontend pages and API foundation are separated. The supplied Streamlit file contains additional in-process AI/video-analysis logic. That logic should be migrated into `backend/ai/` rather than copied into one large JavaScript file.
