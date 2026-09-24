# Dashboard Demo Data Integration

The Command Dashboard and Analytics page are driven by completed Demo Video analyses.

Each successful demo analysis stores a compact summary in:

`database/demo_analysis_history.json`

Stored summary fields include scenario, final result, alarm state, duration, AI frames, unique people, authorized/hidden/unverified/ambiguous people, vehicles, bags and plate OCR outcomes.

The dashboard exposes aggregated counters and bar charts from this history through:

`GET /api/dashboard`

The Dashboard also shows the currently running demo job and refreshes automatically while it is open.

The history stores only compact demo-analysis summaries; raw video frames and `live_jpg` bytes are not persisted there.
