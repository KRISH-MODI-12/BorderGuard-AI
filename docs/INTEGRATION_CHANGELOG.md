# Integration Changelog — Test 14 Face Pipeline

The final face-recognition path from the validation tests is integrated into the HTML + FastAPI project.

Changed:
- `backend/ai/engine.py`
  - Correct `AUTH-###` filename grouping for multi-angle samples.
  - Native SFace feature extraction using the same feature path validated in Tests 9–14.
  - Per-person ROI face recognition.
  - 16x20 minimum face support.
  - SFace threshold 0.45 and identity margin 0.05.
  - Test-14 strict track-level evidence decision.
- `backend/main.py`
  - Demo video analysis now associates face evidence directly with ByteTrack person IDs.
  - Complete-track evidence aggregation replaces simple frame-vote authorization.
  - Small-face support is used without upscaling.
  - Hidden-face rule remains separate.
  - Results expose qualified votes, score, margin, evidence and strict checks.
- `frontend/js/app.js`
  - Shows validated face-AI configuration and detailed person/track results.
  - Distinguishes AUTHORIZED, UNVERIFIED, AMBIGUOUS and FACE HIDDEN.

Validated settings:
- YOLO confidence: 0.35
- SFace threshold: 0.45
- Identity margin: 0.05
- Minimum face: 16x20
- Upscaling: disabled
- Strict qualified observations: >= 5
- Strict votes: >= 5
- Strict vote ratio: >= 70%
- Strict average score: >= 0.60
- Strict average margin: >= 0.05
- Strict evidence dominance: >= 3x when a competing identity has evidence

`UNVERIFIED` is intentionally not the same as `UNAUTHORIZED`.
