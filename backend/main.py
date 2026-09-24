from pathlib import Path
import base64, ipaddress, json, time, uuid, threading
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .ai.engine import engine, summarize

BASE=Path(__file__).resolve().parents[1]
FRONTEND=BASE/'frontend'; DB=BASE/'database'; VIDEOS=DB/'demo_videos'; ALARM=BASE/'alarm'/'alarm.mp3'; CONFIG=BASE/'backend'/'config.json'
VIDEOS.mkdir(parents=True,exist_ok=True); (DB/'authorized_faces').mkdir(parents=True,exist_ok=True); (DB/'authorized_plates').mkdir(parents=True,exist_ok=True)
AUTHORIZED=DB/'authorized_faces'; PLATES=DB/'authorized_plates'

app=FastAPI(title='AI Intelligent Surveillance Platform')
app.mount('/static',StaticFiles(directory=FRONTEND),name='static')

def _json_safe(value):
    """Recursively convert runtime values to JSON-safe Python types.

    The AI stack can produce NumPy scalars/arrays. FastAPI's generic encoder
    does not serialize every NumPy type, so API polling must sanitize them.
    Raw live JPEG bytes are omitted by the job/state snapshots before this
    function is called.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in list(value.items())}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in list(value)]
    try:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if hasattr(value, 'model_dump'):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, '__dict__'):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    return str(value)

state={'theme':'dark','yolo_confidence':.35,'plate_confidence':.25,'unattended_threshold':30,'cameras':[],'alerts':[],'incidents':[],
'vehicle_history':[{'Plate':'GJXX1234','Direction':'IN','Camera':'Gate 1','Status':'Authorized'},{'Plate':'GJXX1234','Direction':'OUT','Camera':'Gate 2','Status':'Authorized'},{'Plate':'GJXX9876','Direction':'IN','Camera':'Gate 1','Status':'Unrecognized'}],'jobs':{}}
if CONFIG.exists():
    try:
        saved=json.loads(CONFIG.read_text())
        for k in ('theme','yolo_confidence','plate_confidence','unattended_threshold'):
            if k in saved: state[k]=saved[k]
    except Exception: pass

class CameraIn(BaseModel):
    name:str=Field(min_length=1,max_length=80); ip:str; stream_url:str; zone:str='Zone A'; scenario:str='Normal Situation'; camera_type:str='Normal'
class SettingsIn(BaseModel):
    theme:str='dark'; yolo_confidence:float=.20; plate_confidence:float=.25; unattended_threshold:int=30
class AlertIn(BaseModel):
    type:str; camera:str='Demo Video'; zone:str='Demo'; message:str; severity:str='Critical'
SCENARIOS={'Company Restricted Area','Railway - Unattended Object','Vehicle / Number Plate Monitoring','Normal Situation'}

def valid_ipv4(value):
    try:
        ip=ipaddress.ip_address(value.strip()); return ip.version==4 and not ip.is_unspecified and not ip.is_loopback and not ip.is_multicast
    except ValueError:return False

def stream_host(url): return urlparse(url).hostname

def validate_camera(c):
    if not valid_ipv4(c.ip): raise HTTPException(400,'Camera IP must be a real IPv4 address. Demo, 0, localhost and filenames are not allowed.')
    if not c.stream_url.startswith(('rtsp://','http://','https://')): raise HTTPException(400,'Stream URL must start with rtsp://, http:// or https://.')
    host=stream_host(c.stream_url)
    if not host or host != c.ip: raise HTTPException(400,'The stream URL host must be the same real IPv4 address entered above.')
    if c.scenario not in SCENARIOS: raise HTTPException(400,'Invalid scenario.')

def model_status():
    s=engine.status()
    return {'YOLO':s['yolo_loaded'],'YuNet':s['yunet_loaded'],'SFace':s['sface_loaded'],'Plate model':s['plate_loaded'],'Alarm':ALARM.exists(),'Authorized faces':AUTHORIZED.exists(),'Authorized plates':PLATES.exists(),'AI engine':s}

@app.get('/')
def index(): return FileResponse(FRONTEND/'index.html')
@app.get('/api/status')
def status():
    cuda=bool(torch.cuda.is_available()); return {'online':True,'cuda':cuda,'gpu':torch.cuda.get_device_name(0) if cuda else 'CPU','models':model_status()}
@app.get('/api/state')
def get_state():
    # Never expose raw live JPEG bytes through the state JSON endpoint.
    # Jobs keep live_jpg internally for the /live MJPEG stream, but /api/state
    # must contain JSON-safe job snapshots only.
    safe_jobs = {}
    for job_id, job in state.get('jobs', {}).items():
        safe_jobs[job_id] = {k: v for k, v in job.items() if k != 'live_jpg'}
        safe_jobs[job_id]['live_frame_available'] = bool(job.get('live_jpg'))
    snapshot = {k: v for k, v in state.items() if k != 'jobs'}
    snapshot['jobs'] = safe_jobs
    snapshot['models'] = model_status()
    snapshot['cuda'] = bool(torch.cuda.is_available())
    return _json_safe(snapshot)
@app.post('/api/settings')
def settings(s:SettingsIn):
    s.theme='light' if s.theme=='light' else 'dark'; s.yolo_confidence=max(.05,min(.95,s.yolo_confidence)); s.plate_confidence=max(.05,min(.95,s.plate_confidence)); s.unattended_threshold=max(1,min(30,s.unattended_threshold))
    state.update({'theme':s.theme,'yolo_confidence':s.yolo_confidence,'plate_confidence':s.plate_confidence,'unattended_threshold':s.unattended_threshold})
    CONFIG.write_text(json.dumps({k:state[k] for k in ('theme','yolo_confidence','plate_confidence','unattended_threshold')},indent=2)); return {'ok':True,'settings':s.model_dump()}

@app.get('/api/cameras')
def cameras(): return state['cameras']
@app.post('/api/cameras/validate')
def camera_validate(c:CameraIn): validate_camera(c); return {'valid':True,'message':'IP format and stream URL format are valid. The real stream will be tested when added.'}
@app.post('/api/cameras')
def add_camera(c:CameraIn):
    validate_camera(c)
    if any(x['ip']==c.ip for x in state['cameras']): raise HTTPException(409,'This camera IP is already registered.')
    # Do not add a fake/demo camera. Test the actual URL first.
    cap=cv2.VideoCapture(c.stream_url); opened=cap.isOpened()
    ok=False
    if opened:
        ok,_frame=cap.read()
    cap.release()
    if not ok: raise HTTPException(400,'The real camera stream could not be opened. Check RTSP path, username/password, port, camera network and firewall.')
    cid=f'CAM-{len(state["cameras"])+1:02d}'; item={'id':cid,'name':c.name,'ip':c.ip,'stream_url':c.stream_url,'zone':c.zone,'status':'Online','type':c.camera_type,'scenario':c.scenario}
    state['cameras'].append(item); return item
@app.delete('/api/cameras/{cid}')
def delete_camera(cid:str):
    old=len(state['cameras']); state['cameras']=[c for c in state['cameras'] if c['id']!=cid]
    if old==len(state['cameras']): raise HTTPException(404,'Camera not found')
    return {'ok':True}

def mjpeg(source, ai=False):
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): return
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if ai:
                try:
                    frame,dets=engine.detect_annotated(frame,state['yolo_confidence'],track=True)
                    cv2.putText(frame,f"People: {sum(d['group']=='person' for d in dets)}  Vehicles: {sum(d['group']=='vehicle' for d in dets)}  Objects: {sum(d['group']=='object' for d in dets)+sum(d['group']=='bag' for d in dets)}",(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                except Exception as e:
                    cv2.putText(frame,'AI unavailable - check Tools/Models',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(60,60,255),2)
            ok,jpg=cv2.imencode('.jpg',frame,[int(cv2.IMWRITE_JPEG_QUALITY),82])
            if ok: yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg.tobytes()+b'\r\n'
    finally: cap.release()
@app.get('/api/cameras/{cid}/stream')
def camera_stream(cid:str):
    cam=next((c for c in state['cameras'] if c['id']==cid),None)
    if not cam: raise HTTPException(404,'Camera not found')
    return StreamingResponse(mjpeg(cam['stream_url']),media_type='multipart/x-mixed-replace; boundary=frame')
@app.get('/api/cameras/{cid}/ai-stream')
def camera_ai_stream(cid:str):
    cam=next((c for c in state['cameras'] if c['id']==cid),None)
    if not cam: raise HTTPException(404,'Camera not found')
    return StreamingResponse(mjpeg(cam['stream_url'],True),media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/api/videos')
def videos(): return [{'name':p.name,'url':f'/api/videos/{p.name}'} for p in sorted(VIDEOS.iterdir()) if p.suffix.lower() in {'.mp4','.mov','.avi','.mkv','.webm'}]
@app.post('/api/videos/upload')
async def upload_video(file:UploadFile=File(...)):
    if not (file.filename or '').lower().endswith(('.mp4','.mov','.avi','.mkv','.webm')): raise HTTPException(400,'Unsupported video format.')
    safe=Path(file.filename).name; dest=VIDEOS/safe; dest.write_bytes(await file.read()); return {'name':safe,'url':f'/api/videos/{safe}'}
@app.get('/api/videos/{name}')
def video(name:str):
    p=VIDEOS/Path(name).name
    if not p.exists(): raise HTTPException(404,'Video not found')
    return FileResponse(p)

async def decode_upload(file):
    raw=await file.read(); frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if frame is None: raise HTTPException(400,'Invalid image file. Use JPG, PNG or WEBP.')
    return frame

@app.post('/api/analyze/frame')
async def analyze_frame(file:UploadFile=File(...)):
    frame=await decode_upload(file)
    try:return summarize(frame,state['yolo_confidence'],state['plate_confidence'])
    except Exception as e: raise HTTPException(500,str(e))

@app.post('/api/tools/reload')
def tools_reload():
    """Reload AI models after files are added/replaced."""
    engine.load_errors.clear()
    engine.yolo = None
    engine.plate = None
    engine.face_detector = None
    engine.face_recognizer = None
    engine._face_db_cache = None
    engine._face_db_mtime = None
    engine._plate_db_cache = None
    engine._plate_db_signature_value = None
    engine._load()
    engine.configure_tesseract()
    return {"ok": True, "models": engine.status()}

@app.get('/api/tools/status')
def tools_status():
    s=engine.status()
    return {
        'tools': {
            'OpenCV': True,
            'FastAPI': True,
            'HTML/CSS/JavaScript': True,
            'YOLO26n': s['yolo_loaded'],
            'ByteTrack': s['yolo_loaded'],
            'YuNet': s['yunet_loaded'],
            'SFace': s['sface_loaded'],
            'License Plate YOLO': s['plate_loaded'],
            'Tesseract OCR': bool(engine.ocr_available),
            'Unattended Object Tracking': s['yolo_loaded'],
            'Privacy Face Masking': s['yunet_loaded'],
        },
        'models': s,
    }

@app.post('/api/tools/detect')
async def tool_detect(file:UploadFile=File(...)):
    frame=await decode_upload(file)
    try:
        annotated,dets=engine.detect_annotated(frame,state['yolo_confidence'])
        counts={'people':sum(d['group']=='person' for d in dets),'vehicles':sum(d['group']=='vehicle' for d in dets),'bags':sum(d['group']=='bag' for d in dets),'objects':sum(d['group']=='object' for d in dets)}
        return {'ok':True,**counts,'detections':dets,'image':engine_image(annotated),'models':engine.status()}
    except Exception as e: raise HTTPException(500,str(e))
@app.post('/api/tools/face-detect')
async def tool_face_detect(file:UploadFile=File(...)):
    frame=await decode_upload(file)
    try:
        annotated,faces=engine.face_annotated(frame); return {'ok':True,'faces':len(faces),'faces_data':[{k:v for k,v in f.items() if k!='raw'} for f in faces],'image':engine_image(annotated),'models':engine.status()}
    except Exception as e: raise HTTPException(500,str(e))
@app.post('/api/tools/face-recognize')
async def tool_face_recognize(file:UploadFile=File(...)):
    frame=await decode_upload(file)
    try:return engine.recognize(frame,0.45,0.05,16,20)
    except Exception as e: raise HTTPException(500,str(e))
@app.post('/api/tools/run-all')
async def tool_run_all(file: UploadFile = File(...)):
    """Run the available AI tools on one image and return all annotated results."""
    frame = await decode_upload(file)
    result = {"ok": True, "models": engine.status(), "errors": []}

    # YOLO + ByteTrack
    try:
        annotated, dets = engine.detect_annotated(frame, state['yolo_confidence'], track=True)
        result["yolo"] = {
            "detections": dets,
            "people": sum(d["group"] == "person" for d in dets),
            "vehicles": sum(d["group"] == "vehicle" for d in dets),
            "bags": sum(d["group"] == "bag" for d in dets),
            "objects": sum(d["group"] == "object" for d in dets),
            "image": engine_image(annotated),
        }
    except Exception as e:
        result["errors"].append(f"YOLO: {e}")

    # YuNet + SFace
    try:
        rec = engine.recognize(frame, 0.45, 0.05, 16, 20)
        result["faces"] = rec
    except Exception as e:
        result["errors"].append(f"Face recognition: {e}")

    # License plate YOLO + OCR
    try:
        annotated, plates, texts, ocr_error = engine.plate_annotated(
            frame, state['plate_confidence']
        )
        result["plates"] = {
            "detections": plates,
            "ocr_texts": texts,
            "ocr_error": ocr_error,
            "authorized_database": sorted(engine.authorized_plates()),
            "image": engine_image(annotated),
        }
    except Exception as e:
        result["errors"].append(f"Plate/OCR: {e}")

    return result

@app.post('/api/tools/plate-detect')
async def tool_plate_detect(file:UploadFile=File(...)):
    frame=await decode_upload(file)
    try:
        annotated,plates,texts,ocr_error=engine.plate_annotated(frame,state['plate_confidence'])
        return {'ok':True,'plates':plates,'ocr_texts':texts,'ocr_error':ocr_error,'authorized_database':sorted(engine.authorized_plates()),'image':engine_image(annotated),'models':engine.status()}
    except Exception as e: raise HTTPException(500,str(e))

def engine_image(frame):
    ok,b=cv2.imencode('.jpg',frame,[int(cv2.IMWRITE_JPEG_QUALITY),88])
    return 'data:image/jpeg;base64,'+base64.b64encode(b.tobytes()).decode() if ok else None

@app.post('/api/demo/analyze')
async def demo_analyze(payload:dict):
    name=Path(str(payload.get('name',''))).name; scenario=str(payload.get('scenario','Normal Situation')); threshold=max(1,min(30,int(payload.get('unattended_threshold',state['unattended_threshold']))))
    if scenario not in SCENARIOS: raise HTTPException(400,'Invalid scenario')
    p=VIDEOS/name
    if not p.exists(): raise HTTPException(404,'Video not found')
    job=str(uuid.uuid4()); state['jobs'][job]={
        'status':'queued',
        'progress':0,
        'scenario':scenario,
        'result':None,
        'live_jpg':None,
        'live_frame_available':False,
        'current_people':0,
        'vehicles':0,
        'bags':0,
        'unique_people':0,
        'ai_frames':0,
        'error':None,
    }
    threading.Thread(target=run_video_job,args=(job,p,scenario,threshold),daemon=True).start(); return {'job_id':job}
@app.get('/api/demo/jobs/{job_id}')
def job(job_id:str):
    j=state['jobs'].get(job_id)
    if not j:
        raise HTTPException(404,'Job not found')
    # live_jpg contains raw bytes for the separate MJPEG stream. Sending those
    # bytes through FastAPI's JSON encoder causes HTTP 500 while a job is running.
    public_job = {k: v for k, v in j.items() if k != 'live_jpg'}
    public_job['live_frame_available'] = bool(j.get('live_jpg'))
    # Force the response through FastAPI's JSON-safe encoder so NumPy/Pydantic
    # values can never cause a 500 while the browser polls analysis status.
    return _json_safe(public_job)


def demo_live_mjpeg(job_id: str):
    """Live JPEG stream of the exact frames being analysed."""
    last = None
    while True:
        j = state['jobs'].get(job_id)
        if not j:
            break
        frame = j.get('live_jpg')
        if frame and frame != last:
            last = frame
            yield b'--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n' + frame + b'\r\n'
        if j.get('status') in ('complete','error'):
            # Send the last frame one final time, then close.
            break
        time.sleep(0.08)

@app.get('/api/demo/jobs/{job_id}/live')
def demo_live(job_id: str):
    if job_id not in state['jobs']:
        raise HTTPException(404, 'Job not found')
    return StreamingResponse(demo_live_mjpeg(job_id), media_type='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control':'no-cache'})

def _draw_processed_overlay(frame, detections, scenario, progress, ai_fps_text=""):
    """Draw the latest AI detections on the current original video frame."""
    out = frame.copy()
    colors = {
        'person': (67, 211, 158),
        'vehicle': (53, 169, 255),
        'bag': (255, 174, 82),
        'object': (180, 150, 255),
    }
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d['box']]
        group = d.get('group', 'object')
        color = colors.get(group, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        tid = f" ID:{d['track_id']}" if d.get('track_id') is not None else ''
        label = f"{d.get('name','object')} {float(d.get('confidence',0)):.0%}{tid}"
        y0 = max(0, y1 - 30)
        cv2.rectangle(out, (x1, y0), (min(out.shape[1], x1 + max(150, len(label) * 9)), y1), color, -1)
        cv2.putText(out, label, (x1 + 5, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (10,20,30), 2, cv2.LINE_AA)

    # Professional video header; this is rendered into the processed video itself.
    header_h = 62
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], header_h), (5, 13, 25), -1)
    out = cv2.addWeighted(overlay, .88, out, .12, 0)
    cv2.putText(out, 'AI SECURITY ANALYSIS', (18, 25), cv2.FONT_HERSHEY_SIMPLEX, .72, (235,245,255), 2, cv2.LINE_AA)
    cv2.putText(out, f'SCENARIO: {scenario}', (18, 49), cv2.FONT_HERSHEY_SIMPLEX, .48, (170,195,215), 1, cv2.LINE_AA)
    right = f'{progress:.0f}%'
    if ai_fps_text:
        right += f'  AI {ai_fps_text}'
    tw = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)[0][0]
    cv2.putText(out, right, (out.shape[1] - tw - 18, 36), cv2.FONT_HERSHEY_SIMPLEX, .55, (67,211,158), 2, cv2.LINE_AA)
    return out


def _write_video_h264(src_mp4: Path, dst_mp4: Path):
    """Try to make a browser-friendly H.264 MP4 when ffmpeg is installed."""
    import shutil, subprocess
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return False
    tmp = dst_mp4.with_name(dst_mp4.stem + '_h264.mp4')
    cmd = [ffmpeg, '-y', '-i', str(src_mp4), '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(tmp)]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(dst_mp4)
            return True
    except Exception:
        pass
    if tmp.exists():
        try: tmp.unlink()
        except Exception: pass
    return False


def _new_face_track_state():
    return {
        'qualified': 0,
        'ambiguous': 0,
        'below_threshold': 0,
        'no_face': 0,
        'face_too_small': 0,
        'sface_errors': 0,
        'scores': {},
        'margins': {},
        'last_face': None,
    }


def _record_face_observation(track, rec):
    reason = rec.get('reason')
    if reason == 'NO_FACE':
        track['no_face'] += 1
    elif reason == 'FACE_TOO_SMALL':
        track['face_too_small'] += 1
    elif reason == 'SFACE_ERROR':
        track['sface_errors'] += 1
    elif reason == 'BELOW_THRESHOLD':
        track['below_threshold'] += 1
    elif reason == 'AMBIGUOUS':
        track['ambiguous'] += 1
    elif reason == 'QUALIFIED':
        track['qualified'] += 1
        identity = rec['best_identity']
        track['scores'].setdefault(identity, []).append(float(rec['best_score']))
        track['margins'].setdefault(identity, []).append(float(rec['margin']))


def _face_track_decision(track):
    return engine.strict_track_decision(track)


def _write_face_overlay(annotated, face_item, width, height):
    x1, y1, x2, y2 = [int(v) for v in face_item['box']]
    x1 = max(0, min(x1, width - 1)); y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width)); y2 = max(0, min(y2, height))
    status = face_item.get('status', 'VERIFYING')
    identity = face_item.get('identity', 'UNKNOWN')
    score = float(face_item.get('score', 0.0) or 0.0)
    if status == 'AUTHORIZED':
        color = (67, 211, 158)
        label = f'AUTHORIZED • {identity} {score:.2f}'
    elif status == 'AMBIGUOUS':
        color = (245, 184, 61)
        label = 'AMBIGUOUS • VERIFYING'
    elif status == 'UNVERIFIED':
        color = (60, 60, 255)
        label = 'UNVERIFIED • FACE MASKED'
    else:
        color = (245, 184, 61)
        label = f'VERIFYING • {identity}' if identity != 'UNKNOWN' else 'VERIFYING'
    if status != 'AUTHORIZED':
        roi = annotated[y1:y2, x1:x2]
        if roi.size:
            annotated[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(annotated, (x1, max(0, y1 - 28)), (min(width, x1 + max(210, len(label) * 8)), y1), color, -1)
    cv2.putText(annotated, label, (x1 + 5, max(19, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .48, (10, 20, 30), 2, cv2.LINE_AA)


def _plate_vehicle_track_id(plate_box, vehicle_detections):
    """Associate a detected plate with a vehicle by plate-center containment."""
    x1,y1,x2,y2=[int(v) for v in plate_box]
    cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
    containing=[]
    for d in vehicle_detections:
        bx1,by1,bx2,by2=[int(v) for v in d.get('box',[0,0,0,0])]
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            containing.append(d)
    if containing:
        containing.sort(key=lambda d: float(d.get('confidence',0)), reverse=True)
        return containing[0].get('track_id')
    return None


def _plate_track_finalize(observations, authorized_db):
    """Aggregate repeated OCR strings and compare them with authorized_plates."""
    rows=[]
    for normalized, obs in observations.items():
        votes=int(obs.get('votes',0))
        authorized=normalized in authorized_db
        confirmed=votes>=2
        if not confirmed:
            status='UNVERIFIED OCR'
        elif authorized:
            status='AUTHORIZED VEHICLE'
        else:
            status='UNRECOGNIZED VEHICLE'
        rows.append({
            'plate':normalized,
            'status':status,
            'authorized':bool(authorized and confirmed),
            'observations':votes,
            'first_frame':min(obs.get('frames') or [0]),
            'last_frame':max(obs.get('frames') or [0]),
            'avg_detector_confidence':round(float(np.mean(obs.get('det_conf') or [0.0])),3),
            'avg_ocr_confidence':round(float(np.mean(obs.get('ocr_conf') or [0.0])),1),
            'vehicle_track_ids':sorted({int(x) for x in obs.get('vehicle_tids',[]) if x is not None}),
        })
    rows.sort(key=lambda x: (x['status']!='UNRECOGNIZED VEHICLE', -x['observations'], x['plate']))
    return rows


def run_video_job(job,p,scenario,threshold):
    """Create a FULL-LENGTH processed video using the validated Test-14 face pipeline."""
    started=time.time()
    state['jobs'][job].update({
        'status':'running',
        'stage':'Opening video...',
        'progress':0.1,
        'started_at':time.strftime('%Y-%m-%d %H:%M:%S')
    })
    cap=cv2.VideoCapture(str(p))
    if not cap.isOpened():
        state['jobs'][job].update({
            'status':'error',
            'stage':'Opening video failed',
            'error':f'Unable to open video: {p.name}. The uploaded file may be incomplete or unsupported.'
        })
        return
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 25)
    if fps <= 0: fps = 25.0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        state['jobs'][job].update({'status':'error','stage':'Reading video metadata failed','error':'Unable to read video resolution.','finished_at':time.strftime('%Y-%m-%d %H:%M:%S')})
        return

    output_dir=VIDEOS/'processed'
    output_dir.mkdir(parents=True,exist_ok=True)
    raw_out=output_dir/f'{job}_processed.mp4'
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    writer=cv2.VideoWriter(str(raw_out), fourcc, fps, (width,height))
    if not writer.isOpened():
        cap.release()
        state['jobs'][job].update({'status':'error','stage':'Creating output writer failed','error':'Unable to create processed video writer.','finished_at':time.strftime('%Y-%m-%d %H:%M:%S')})
        return

    # Keep browser playback smooth by writing every source frame while AI runs at a lower cadence.
    ai_target_fps=min(10.0, max(1.0, fps))
    ai_interval=max(1,int(round(fps/ai_target_fps)))
    latest_detections=[]
    latest_face_overlays=[]
    latest_plate_text=[]
    latest_plate_results=[]
    plate_observations={}
    authorized_plate_db=set(engine.authorized_plates())
    people_ids=set(); vehicle_ids=set(); bag_ids=set()
    people_current=vehicles_current=bags_current=0
    frames=0; ai_frames=0; bags_seen=0
    bag_start=None; last_bag_seen=None; alert=False; grace=3.0
    last_ai_time=time.perf_counter(); measured_ai_fps=0.0

    # Face evidence is stored PER BYTE TRACK ID. Tracks are never force-merged.
    face_tracks={}
    hidden_confirmed=set()
    hidden_since={}
    hidden_tids=set()
    alert_cache=set()

    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frames += 1
            progress=frames/max(total,1)*100
            # Publish progress before AI inference so the browser does not sit
            # at 0% while the first expensive inference is running.
            state['jobs'][job].update({
                'progress':round(min(99.5,progress),1),
                'current_time':round(frames/fps,2),
                'duration_seconds':round(total/fps,2),
                'stage':'Processing video frames...'
            })

            if (frames-1) % ai_interval == 0 or frames == 1:
                try:
                    state['jobs'][job].update({
                        'stage':'AI inference (YOLO + tracking + face recognition)',
                        'progress':round(min(99.0, progress),1),
                    })
                    latest_detections=engine.detect(frame,state['yolo_confidence'],track=True)
                    ai_frames += 1
                    now=time.perf_counter(); dt=now-last_ai_time
                    if dt>0:
                        current=1.0/dt
                        measured_ai_fps=current if measured_ai_fps==0 else (0.8*measured_ai_fps+0.2*current)
                    last_ai_time=now

                    people_current=sum(d['group']=='person' for d in latest_detections)
                    vehicles_current=sum(d['group']=='vehicle' for d in latest_detections)
                    bags=[d for d in latest_detections if d['group']=='bag']
                    bags_current=len(bags)
                    current_person_tids=set()

                    for d in latest_detections:
                        tid=d.get('track_id')
                        if tid is not None and d['group']=='person':
                            tid=int(tid); people_ids.add(tid); current_person_tids.add(tid)
                            if tid not in face_tracks:
                                face_tracks[tid]=_new_face_track_state()
                        if tid is not None and d['group']=='vehicle': vehicle_ids.add(int(tid))
                        if tid is not None and d['group']=='bag': bag_ids.add(int(tid))

                    # ------------------------------------------------------
                    # FACE PIPELINE: PERSON ROI -> YUNET -> SFACE
                    # ------------------------------------------------------
                    latest_face_overlays=[]
                    face_visible_tids=set()

                    if scenario=='Company Restricted Area' and engine.face_recognizer is not None and engine.face_detector is not None:
                        for d in latest_detections:
                            if d.get('group')!='person' or d.get('track_id') is None:
                                continue
                            tid=int(d['track_id'])
                            x1,y1,x2,y2=[int(v) for v in d['box']]
                            x1=max(0,min(x1,width-1)); y1=max(0,min(y1,height-1)); x2=max(0,min(x2,width)); y2=max(0,min(y2,height))
                            if x2<=x1 or y2<=y1:
                                continue
                            roi=frame[y1:y2,x1:x2]
                            if roi.size==0:
                                continue
                            rec=engine.recognize_person_roi(roi,0.45,0.05,16,20)
                            track=face_tracks[tid]
                            _record_face_observation(track, rec)
                            if rec.get('has_face'):
                                face_visible_tids.add(tid)
                                if rec.get('face_box'):
                                    fx1,fy1,fx2,fy2=rec['face_box']
                                    global_box=[x1+fx1,y1+fy1,x1+fx2,y1+fy2]
                                    track['last_face']=global_box
                                    decision=_face_track_decision(track)
                                    latest_face_overlays.append({
                                        'track_id':tid,
                                        'box':global_box,
                                        'identity':decision.get('identity','UNKNOWN'),
                                        'status':decision.get('status','VERIFYING'),
                                        'score':float(rec.get('best_score',0.0) or 0.0),
                                        'decision':rec.get('reason'),
                                    })

                        # --------------------------------------------------
                        # HIDDEN/COVERED FACE RULE
                        # --------------------------------------------------
                        now_video_t=frames/fps
                        for tid in current_person_tids:
                            if tid in face_visible_tids:
                                hidden_since.pop(tid,None)
                                hidden_tids.discard(tid)
                            else:
                                hidden_since.setdefault(tid,now_video_t)
                                if now_video_t-hidden_since[tid] >= 1.5 and tid not in hidden_confirmed:
                                    hidden_confirmed.add(tid)
                                    hidden_tids.add(tid)
                                    alert=True
                                    if tid not in alert_cache:
                                        alert_cache.add(tid)
                                        create_alert(AlertIn(
                                            type='Face Hidden',
                                            camera='Demo Video',
                                            zone='Restricted',
                                            message=f'Face remained unavailable for Track {tid} for at least 1.5 seconds in the restricted-area demo.',
                                            severity='Critical'
                                        ))

                    if scenario=='Vehicle / Number Plate Monitoring' and engine.plate is not None:
                        try:
                            latest_plate_results=[]
                            plates=engine.plate_boxes(frame,state['plate_confidence'])
                            _, ocr_error = engine.plate_ocr(frame, plates)
                            vehicle_dets=[d for d in latest_detections if d.get('group')=='vehicle']
                            for pp in plates:
                                raw=pp.get('ocr_text','')
                                norm=engine.normalize_plate(raw)
                                vehicle_tid=_plate_vehicle_track_id(pp.get('box',[0,0,0,0]), vehicle_dets)
                                pp['vehicle_track_id']=vehicle_tid
                                pp['normalized_plate']=norm
                                pp['authorized']=bool(norm and norm in authorized_plate_db)
                                pp['status']=('AUTHORIZED VEHICLE' if pp['authorized'] else ('UNRECOGNIZED VEHICLE' if norm else 'OCR UNREADABLE'))
                                latest_plate_results.append(pp)
                                if norm:
                                    obs=plate_observations.setdefault(norm, {'votes':0,'frames':[],'det_conf':[],'ocr_conf':[],'vehicle_tids':[]})
                                    obs['votes']+=1
                                    obs['frames'].append(frames)
                                    obs['det_conf'].append(float(pp.get('confidence',0)))
                                    obs['ocr_conf'].append(float(pp.get('ocr_confidence',0)))
                                    obs['vehicle_tids'].append(vehicle_tid)
                            if ocr_error:
                                state['jobs'][job].update({'ai_warning':f'Plate/OCR: {ocr_error}'})
                        except Exception as plate_exc:
                            latest_plate_results=[]
                            state['jobs'][job].update({'ai_warning':f'Plate/OCR: {type(plate_exc).__name__}: {plate_exc}','stage':'AI warning; continuing video processing...'})

                    if bags:
                        bags_seen += len(bags)
                        last_bag_seen=frames/fps
                        if bag_start is None:
                            bag_start=frames/fps
                        if (frames/fps)-bag_start >= float(threshold):
                            alert=True
                    elif last_bag_seen is not None and (frames/fps)-last_bag_seen > grace:
                        bag_start=None
                        last_bag_seen=None

                except Exception as e:
                    # Keep the full-video job alive if one AI cadence fails; expose
                    # the warning to the UI instead of silently swallowing it.
                    state['jobs'][job].update({'ai_warning':f'{type(e).__name__}: {e}','stage':'AI warning; continuing video processing...'})

            display_dets=[d for d in latest_detections if d.get('group')!='plate']
            annotated=_draw_processed_overlay(frame,display_dets,scenario,progress,f'{measured_ai_fps:.1f}')

            # Draw latest plate/OCR verification.
            if scenario=='Vehicle / Number Plate Monitoring':
                for pp in latest_plate_results:
                    x1,y1,x2,y2=[int(v) for v in pp.get('box',[0,0,0,0])]
                    status=pp.get('status','OCR UNREADABLE')
                    color=(67,211,158) if status=='AUTHORIZED VEHICLE' else ((60,60,255) if status=='UNRECOGNIZED VEHICLE' else (245,184,61))
                    label='PLATE'
                    if pp.get('ocr_text'): label += f" • {pp['ocr_text']}"
                    label += f" • {status}"
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),color,3)
                    cv2.rectangle(annotated,(x1,max(0,y1-28)),(min(width,x1+max(250,len(label)*8)),y1),color,-1)
                    cv2.putText(annotated,label,(x1+5,max(19,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,(10,20,30),2,cv2.LINE_AA)

            # Draw latest face status/masking.
            for f in latest_face_overlays:
                _write_face_overlay(annotated,f,width,height)

            # Confirmed hidden/covered faces are explicitly treated as unauthorized for this demo rule.
            for d in display_dets:
                if d.get('group')=='person' and d.get('track_id') is not None and int(d['track_id']) in hidden_tids:
                    x1,y1,x2,y2=d['box']
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(60,60,255),3)
                    label='FACE HIDDEN • UNAUTHORIZED'
                    cv2.rectangle(annotated,(x1,max(0,y1-28)),(min(width,x1+260),y1),(60,60,255),-1)
                    cv2.putText(annotated,label,(x1+5,max(19,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),2,cv2.LINE_AA)

            # Railway timer/status banner.
            if scenario=='Railway - Unattended Object' and bag_start is not None:
                elapsed=max(0,frames/fps-bag_start)
                msg=f'BAG TIMER {elapsed:.1f}s / {threshold}s'
                c=(60,60,255) if elapsed>=threshold else (255,174,82)
                cv2.putText(annotated,msg,(18,height-22),cv2.FONT_HERSHEY_SIMPLEX,.62,c,2,cv2.LINE_AA)

            ok_live,live_buf=cv2.imencode('.jpg',annotated,[int(cv2.IMWRITE_JPEG_QUALITY),82])
            if ok_live:
                state['jobs'][job]['live_jpg']=live_buf.tobytes()
                state['jobs'][job]['live_frame_available']=True

            writer.write(annotated)
            state['jobs'][job].update({
                'stage':'Rendering processed video...',
                'progress':round(min(99.5,progress),1),
                'current_time':round(frames/fps,2),
                'duration_seconds':round(total/fps,2),
                'ai_frames':ai_frames,
                'unique_people':len(people_ids),
                'current_people':people_current,
                'vehicles':vehicles_current,
                'bags':bags_current,
                'plates':len(latest_plate_results),
                'ocr_plates':sum(1 for pp in latest_plate_results if pp.get('ocr_text')),
                'alert':bool(alert),
            })

        writer.release(); cap.release()
        final_name=f'{job}_processed.mp4'
        final_path=output_dir/final_name
        if raw_out != final_path:
            raw_out.replace(final_path)
        _write_video_h264(final_path,final_path)

        # ------------------------------------------------------------
        # FINAL STRICT FACE DECISIONS
        # ------------------------------------------------------------
        person_results=[]
        authorized_tracks=[]
        unverified_tracks=[]
        ambiguous_tracks=[]
        hidden_tracks=[]

        for tid in sorted(people_ids):
            t=face_tracks.get(tid,_new_face_track_state())
            decision=_face_track_decision(t)
            status=decision.get('status','UNVERIFIED')
            identity=decision.get('identity','UNKNOWN')
            winner=decision.get('winner') or {}
            runner=decision.get('runner_up') or {}
            # Keep the recognized identity even when a later hidden-face event
            # occurs. The hidden-face state is a SECURITY VIOLATION, not an
            # erasure of the identity that was previously verified.
            base_status=status
            current_hidden = tid in hidden_tids
            hidden_alerted = tid in hidden_confirmed
            verified_identity = identity if base_status=='AUTHORIZED' else None
            display_identity = verified_identity or 'UNKNOWN'

            if base_status=='AUTHORIZED':
                authorized_tracks.append(tid)
            elif base_status=='AMBIGUOUS':
                ambiguous_tracks.append(tid)
            else:
                unverified_tracks.append(tid)

            if current_hidden:
                display_status='FACE HIDDEN'
                hidden_tracks.append(tid)
            else:
                display_status=base_status

            person_results.append({
                'track_id':tid,
                'identity':display_identity,
                'verified_identity':verified_identity,
                'status':display_status,
                'identity_status':base_status,
                'face_hidden_current':current_hidden,
                'face_hidden_alerted':hidden_alerted,
                'auth_votes':int(winner.get('votes',0) if base_status=='AUTHORIZED' else 0),
                'unauthorized_votes':1 if current_hidden else 0,
                'qualified_votes':int(t.get('qualified',0)),
                'ambiguous_observations':int(t.get('ambiguous',0)),
                'below_threshold':int(t.get('below_threshold',0)),
                'avg_score':round(float(winner.get('avg_score',0.0)),4),
                'avg_margin':round(float(winner.get('avg_margin',0.0)),4),
                'best_score':round(float(winner.get('best_score',0.0)),4),
                'evidence':round(float(winner.get('evidence',0.0)),6),
                'evidence_gap':round(float(decision.get('evidence_gap',0.0)),6),
                'evidence_ratio':None if decision.get('evidence_ratio') is None else round(float(decision['evidence_ratio']),3),
                'vote_ratio':round(float(decision.get('vote_ratio',0.0)),4),
                'strict_checks':decision.get('strict_checks',{}),
            })

        plate_results=[]
        if scenario=='Vehicle / Number Plate Monitoring':
            plate_results=_plate_track_finalize(plate_observations, authorized_plate_db)

        # Keep identity and security-violation counts separate. A person can be
        # a previously verified authorized identity and still trigger a hidden-face
        # restricted-area alert later in the same track.
        auth_people=len(authorized_tracks)
        hidden_face_people=len(hidden_tracks)
        unauth_people=hidden_face_people
        unverified_people=len(unverified_tracks)
        ambiguous_people=len(ambiguous_tracks)

        # ------------------------------------------------------------
        # FINAL RESTRICTED-AREA ALARM DECISION
        #
        # For the demo, the audible alarm is intentionally triggered
        # ONLY AFTER the full analysis result is available.
        # A restricted-area analysis raises the alarm when the final
        # person review contains either:
        #   1) a confirmed hidden/unauthorized condition, OR
        #   2) an unverified track.
        #
        # UNVERIFIED is kept distinct from UNAUTHORIZED in the data;
        # it simply uses the demo's conservative security-alarm rule.
        # ------------------------------------------------------------
        final_alarm = bool(alert)
        alarm_reason = ''

        if scenario=='Company Restricted Area':
            if unauth_people>0:
                overall='UNAUTHORIZED'
                final_alarm = True
                alarm_reason = f'{unauth_people} restricted-area unauthorized/face-hidden condition(s) confirmed.'
            elif ambiguous_people>0:
                overall='REVIEW / AMBIGUOUS'
                if final_alarm:
                    alarm_reason = 'A critical restricted-area event occurred during analysis.'
            elif unverified_people>0:
                overall='REVIEW / UNVERIFIED'
                final_alarm = True
                alarm_reason = f'{unverified_people} restricted-area track(s) could not be verified after complete-track analysis.'
            elif auth_people>0:
                overall='AUTHORIZED'
                if final_alarm:
                    alarm_reason = 'A critical restricted-area event occurred during analysis.'
            else:
                overall='NO VERIFIED PERSONS'
                final_alarm = True
                alarm_reason = 'No person could be verified in the restricted-area analysis.'

        elif scenario=='Railway - Unattended Object':
            overall='SUSPICIOUS OBJECT' if alert else 'NORMAL'
            final_alarm = bool(alert)
            if final_alarm:
                alarm_reason = f'Bag/object remained unattended for the selected threshold ({threshold}s).'

        elif scenario=='Normal Situation':
            overall='NORMAL' if not alert else 'ALERT'
            final_alarm = bool(alert)
            if final_alarm:
                alarm_reason = 'A security event was raised during the normal-situation analysis.'

        elif scenario=='Vehicle / Number Plate Monitoring':
            confirmed_unrecognized=[x for x in plate_results if x['status']=='UNRECOGNIZED VEHICLE']
            confirmed_authorized=[x for x in plate_results if x['status']=='AUTHORIZED VEHICLE']
            unreadable=[x for x in plate_results if x['status']=='UNVERIFIED OCR']
            if confirmed_unrecognized:
                overall='UNRECOGNIZED VEHICLE'
                final_alarm=True
                alarm_reason=f'{len(confirmed_unrecognized)} OCR plate(s) were not found in the authorized plate database.'
            elif confirmed_authorized:
                overall='AUTHORIZED VEHICLE'
                final_alarm=bool(alert)
                if final_alarm:
                    alarm_reason='A security event was raised during vehicle analysis.'
            elif unreadable:
                overall='PLATE OCR UNVERIFIED'
                final_alarm=bool(alert)
                if final_alarm:
                    alarm_reason='A security event was raised during vehicle analysis.'
            elif vehicles_current>0 or vehicle_ids:
                overall='VEHICLE DETECTED / OCR NOT CONFIRMED'
                final_alarm=bool(alert)
            else:
                overall='NORMAL'
                final_alarm=bool(alert)

        else:
            overall='NORMAL' if not alert else 'ALERT'
            final_alarm = bool(alert)
            if final_alarm:
                alarm_reason = 'A security event was raised during analysis.'

        alert = final_alarm

        result={
            'scenario':scenario,
            'frames':frames,
            'duration_seconds':round(frames/fps,2),
            'source_fps':round(fps,2),
            'output_fps':round(fps,2),
            'ai_frames':ai_frames,
            'unique_people':len(people_ids),
            'current_people':people_current,
            'authorized_people':auth_people,
            'hidden_face_people':hidden_face_people,
            'unauthorized_people':unauth_people,
            'unverified_people':unverified_people,
            'ambiguous_people':ambiguous_people,
            'vehicles':max(len(vehicle_ids),vehicles_current),
            'bags':max(len(bag_ids),bags_current),
            'plate_results':plate_results,
            'authorized_plates':sorted(authorized_plate_db),
            'plate_detection_count':sum(1 for _ in plate_results),
            'unattended_threshold':threshold if scenario=='Railway - Unattended Object' else None,
            'grace_seconds':grace if scenario=='Railway - Unattended Object' else None,
            'alert':alert,
            'alarm':final_alarm,
            'alarm_reason':alarm_reason,
            'status':overall,
            'processed_video':f'/api/processed/{final_name}',
            'face_recognition':{
                'min_face':[16,20],
                'threshold':0.45,
                'identity_margin':0.05,
                'strict_min_qualified':5,
                'strict_min_votes':5,
                'strict_vote_ratio':0.70,
                'strict_avg_score':0.60,
                'strict_avg_margin':0.05,
                'strict_evidence_dominance':3.0,
            },
            'person_results':person_results,
        }

        state['jobs'][job].update({
            'status':'complete',
            'stage':'Analysis complete',
            'progress':100.0,
            'result':result,
            'live_jpg':None,
            'live_frame_available':False,
            'finished_at':time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds':round(time.time()-started,2),
            'error':None,
        })

        # Summarize strict-face review alerts once per track; hidden remains critical.
        if scenario=='Company Restricted Area':
            for tid in unverified_tracks:
                key=f'UNVERIFIED-{tid}'
                if key not in alert_cache:
                    alert_cache.add(key)
                    create_alert(AlertIn(
                        type='Identity Not Verified',
                        camera='Demo Video',
                        zone='Restricted',
                        message=f'Track {tid} did not meet the strict face-recognition evidence rules.',
                        severity='Warning'
                    ))
            for tid in ambiguous_tracks:
                key=f'AMBIGUOUS-{tid}'
                if key not in alert_cache:
                    alert_cache.add(key)
                    create_alert(AlertIn(
                        type='Ambiguous Identity',
                        camera='Demo Video',
                        zone='Restricted',
                        message=f'Track {tid} has competing identity evidence and requires review.',
                        severity='Warning'
                    ))

        if scenario=='Vehicle / Number Plate Monitoring' and plate_results:
            for pr in [x for x in plate_results if x['status']=='UNRECOGNIZED VEHICLE']:
                key=f"PLATE-{pr['plate']}"
                if key not in alert_cache:
                    alert_cache.add(key)
                    create_alert(AlertIn(type='Unrecognized Vehicle',camera='Demo Video',zone='Vehicle Gate',message=f"OCR plate {pr['plate']} was not found in the authorized plate database after {pr['observations']} observations.",severity='Critical'))

        if scenario=='Railway - Unattended Object' and alert:
            create_alert(AlertIn(type='Suspicious Object',camera='Demo Video',zone='Railway',message=f'Bag remained unattended for at least {threshold} seconds.',severity='Critical'))

    except Exception as e:
        try: writer.release()
        except Exception: pass
        cap.release()
        state['jobs'][job].update({
            'status':'error',
            'stage':'Analysis failed',
            'error':f'{type(e).__name__}: {e}',
            'live_jpg':None,
            'live_frame_available':False,
            'finished_at':time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds':round(time.time()-started,2),
        })

@app.get('/api/processed/{name}')
def processed_video(name:str):
    p=(VIDEOS/'processed'/Path(name).name)
    if not p.exists(): raise HTTPException(404,'Processed video not found')
    return FileResponse(p, media_type='video/mp4')

@app.get('/api/alarm')
def alarm_audio():
    if not ALARM.exists():
        raise HTTPException(404, 'Alarm audio file not found at alarm/alarm.mp3')
    return FileResponse(ALARM, media_type='audio/mpeg', filename='alarm.mp3')

@app.post('/api/alerts')
def create_alert(a:AlertIn):
    item={'id':str(uuid.uuid4()),'Time':time.strftime('%H:%M:%S'),'Severity':a.severity,'Type':a.type,'Camera':a.camera,'Zone':a.zone,'Confidence':'AI / Demo','Status':'Unresolved','Message':a.message}; state['alerts'].insert(0,item); state['incidents'].insert(0,item.copy()); return item
@app.get('/api/alerts')
def alerts(): return {'alerts':state['alerts'],'incidents':state['incidents']}
@app.delete('/api/alerts')
def clear_alerts(): state['alerts'].clear(); state['incidents'].clear(); return {'ok':True}
@app.get('/api/authorized-plates')
def authorized_plates():
    db=sorted(engine.authorized_plates())
    return {'plates':db,'count':len(db)}

@app.get('/api/plate-history')
def plate_history(): return state['vehicle_history']
@app.get('/api/face-database')
def face_database(): return [{'id':p.stem,'image':p.name} for p in sorted(AUTHORIZED.iterdir()) if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
@app.get('/api/map')
def security_map(): return {'zones':[{'name':'ZONE A • MAIN GATE','x':5,'y':8,'w':28,'h':35},{'name':'ZONE B • PARKING','x':39,'y':8,'w':25,'h':35},{'name':'RESTRICTED ZONE','x':70,'y':8,'w':25,'h':35},{'name':'ZONE D • CORRIDOR','x':22,'y':55,'w':35,'h':35},{'name':'RAILWAY / DEMO','x':64,'y':55,'w':27,'h':35}]}
