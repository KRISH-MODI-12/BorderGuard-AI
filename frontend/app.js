const app=document.getElementById("app"), toast=document.getElementById("toast");
let page="home", videoInfo=null, currentJob=null, poller=null, selectedScenario="Company Restricted Area";

const scenarios=["Company Restricted Area","Railway - Unattended Object","Vehicle / Number Plate Monitoring","Normal Situation"];
const $=(s)=>document.querySelector(s);
function esc(x){return String(x??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[m]))}
function fmt(s){s=Math.max(0,Number(s||0));let m=Math.floor(s/60),x=Math.floor(s%60);return `${String(m).padStart(2,"0")}:${String(x).padStart(2,"0")}`}
function notify(msg){toast.textContent=msg;toast.classList.add("show");setTimeout(()=>toast.classList.remove("show"),3000)}
async function api(url,opt={}){const r=await fetch(url,opt);if(!r.ok){let t=await r.text();throw new Error(t||`HTTP ${r.status}`)}return r.json()}
function scenarioSelect(id="scenario"){return `<select id="${id}">${scenarios.map(s=>`<option>${esc(s)}</option>`).join("")}</select>`}
function metrics(items){return `<div class="grid grid5">${items.map(x=>`<div class="card metric"><span>${esc(x[0])}</span><b>${esc(x[1])}</b><span>${esc(x[2])}</span></div>`).join("")}</div>`}
function table(rows){if(!rows.length)return `<div class="status">No records.</div>`;let keys=Object.keys(rows[0]);return `<div class="table-wrap"><table class="table"><thead><tr>${keys.map(k=>`<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>`<tr>${keys.map(k=>`<td>${esc(r[k])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`}

document.querySelectorAll(".nav button").forEach(b=>b.onclick=()=>{page=b.dataset.page;document.querySelectorAll(".nav button").forEach(x=>x.classList.remove("active"));b.classList.add("active");render()});

async function render(){
  if(page==="home") return home();
  if(page==="dashboard") return dashboard();
  if(page==="cameras") return cameras();
  if(page==="ai") return ai();
  if(page==="alerts") return alerts();
  if(page==="map") return securityMap();
  if(page==="analytics") return analytics();
  if(page==="videos") return videos();
  if(page==="settings") return settings();
}
async function home(){
 app.innerHTML=`<section class="hero"><div class="pill">● AI-POWERED CCTV • REAL-TIME EVENT ANALYSIS</div><h1>AI INTELLIGENT<br>SURVEILLANCE PLATFORM</h1><p>Existing CCTV infrastructure enhanced with AI for person, object and vehicle analysis, security-zone monitoring, incident detection and alerts.</p><span class="pill">● SYSTEM READY</span></section>
 <h2>Project Overview</h2><p class="help">HTML/CSS/JavaScript frontend + Python FastAPI AI backend.</p>
 <div class="grid grid4">${[
 ["🏢 Restricted Area","Person → face recognition → authorized database → restricted zone → unauthorized alert."],
 ["🚉 Unattended Bag","Person + bag detection → person leaves → stationary bag → threshold → alert."],
 ["🚗 Vehicle Monitoring","Vehicle detection → plate/OCR stage → authorized database → IN/OUT history."],
 ["🟢 Normal Situation","Normal activity → no suspicious rule → no alert."]
 ].map(x=>`<div class="scenario"><h3>${x[0]}</h3><p>${x[1]}</p></div>`).join("")}</div>
 <div class="flow"><b>Existing CCTV</b> → PERSON / OBJECT / VEHICLE → FACE / TRACK / NUMBER PLATE OCR → <b>EVENT ANALYSIS</b> → NORMAL or <span class="bad">SUSPICIOUS</span> → Alert</div>`;
}
async function dashboard(){
 let d=await api("/api/dashboard"),c=await api("/api/cameras");
 app.innerHTML=`<div class="hero"><h1>AI SECURITY COMMAND CENTER</h1><p>Real-Time Intelligent Surveillance & Threat Detection</p></div>${metrics([
 ["CAMERAS",d.cameras,"Registered"],["PEOPLE",d.people,"Latest AI result"],["VEHICLES",d.vehicles,"Latest AI result"],["BAGS",d.bags,"Latest AI result"],["ACTIVE ALERTS",d.alerts,"Current session"]])}
 <div class="section-title"><h2>Registered Cameras</h2></div>${table(c.cameras)}`;
}
async function cameras(){
 let d=await api("/api/cameras");
 app.innerHTML=`<h2>📹 LIVE CCTV CAMERAS</h2><p class="help">Browser pages cannot directly play RTSP URLs. The Python backend can connect to RTSP/HTTP sources; this page registers the source and provides the configuration.</p>
 <div class="panel"><h3>➕ Add Camera</h3><div class="form">
 <div class="field"><label>Camera Name</label><input id="camName" placeholder="Main Gate Camera"></div>
 <div class="field"><label>Real Camera IP</label><input id="camIp" placeholder="192.168.1.100"></div>
 <div class="field full"><label>RTSP / HTTP Stream URL</label><input id="camUrl" placeholder="rtsp://192.168.1.100:554/stream"></div>
 <div class="field"><label>Security Zone</label><select id="camZone"><option>Zone A</option><option>Zone B</option><option>Zone C</option><option>Zone D</option><option>Restricted</option><option>Railway Station</option></select></div>
 <div class="field"><label>Camera Type</label><select id="camType"><option>Normal</option><option>Night Vision</option><option>Thermal</option></select></div>
 <div class="field"><label>Scenario</label>${scenarioSelect("camScenario")}</div>
 <div class="full"><button class="btn primary" id="addCam">Register Real IP Camera</button></div></div></div>
 <h3>Camera Configuration</h3>${table(d.cameras)}`;
 $("#addCam").onclick=async()=>{try{let body={name:$("#camName").value.trim(),ip:$("#camIp").value.trim(),stream_url:$("#camUrl").value.trim(),zone:$("#camZone").value,type:$("#camType").value,scenario:$("#camScenario").value};if(!body.name||!body.ip||!body.stream_url)throw Error("Fill camera name, IP and stream URL.");await api("/api/cameras",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});notify("Camera registered");cameras()}catch(e){notify(e.message)}};
}
async function ai(){
 app.innerHTML=`<h2>🤖 AI DETECTION CENTER</h2><div class="panel"><div class="field"><label>Select Scenario</label>${scenarioSelect("aiScenario")}</div><div id="scenarioPanel"></div></div>`;
 $("#aiScenario").value=selectedScenario;$("#aiScenario").onchange=()=>{selectedScenario=$("#aiScenario").value;drawAiPanel()};drawAiPanel();
}
function drawAiPanel(){
 let s=$("#aiScenario").value,p=$("#scenarioPanel");
 if(s==="Company Restricted Area") p.innerHTML=`<div class="flow">CCTV → Person Detection → Face Detection → Face Recognition → Authorized Face Database → Restricted Zone → 🟩 Authorized / 🟥 Unauthorized → Alarm</div><button class="btn danger" id="testAlarm">🔊 Test Unauthorized Alarm</button><div id="faceList"></div>`;
 else if(s==="Railway - Unattended Object") p.innerHTML=`<div class="flow">Person + Bag → Detection → Person moves away → Bag remains → Threshold → 🚨 Suspicious Object</div><p class="help">This manual demo keeps the original 30-second security threshold.</p><input id="railDur" type="number" min="0" max="300" value="30"><button class="btn primary" id="railBtn">Evaluate Railway Event</button><div id="railOut"></div>`;
 else if(s==="Vehicle / Number Plate Monitoring") p.innerHTML=`<div class="flow">CCTV → Vehicle Detection → Number Plate Detection → OCR → Authorized Plate Database → IN / OUT → Alert</div><p class="help">The current Python source does not contain a plate-detection model, so the UI does not fake OCR. Demo plate checking uses synthetic records.</p><div class="form"><div class="field"><label>Demo Plate</label><input id="plate" value="GJXX1234"></div><div class="field"><label>Direction</label><select id="dir"><option>IN</option><option>OUT</option></select></div><div class="field"><label>Camera</label><select id="vcam"><option>Gate 1</option><option>Gate 2</option><option>Parking Camera</option></select></div></div><button class="btn primary" id="plateBtn">Check Demo Plate</button><div id="plateOut"></div>`;
 else p.innerHTML=`<div class="grid grid4">${metrics([["PEOPLE",8,"Demo"],["VEHICLES",3,"Demo"],["SUSPICIOUS OBJECTS",0,"Demo"],["ALERTS",0,"Demo"]])}</div><div class="success">🟢 SYSTEM STATUS: NORMAL</div>`;
 if($("#testAlarm"))$("#testAlarm").onclick=async()=>{await api("/api/event/test-unauthorized",{method:"POST"});notify("Unauthorized demo alert created");};
 if($("#railBtn"))$("#railBtn").onclick=async()=>{let r=await api("/api/railway/evaluate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({duration:Number($("#railDur").value)})});$("#railOut").innerHTML=r.alert===false?`<div class="success">🟢 Threshold not reached.</div>`:`<div class="alert">🚨 Suspicious Object — 30-second threshold reached.</div>`};
 if($("#plateBtn"))$("#plateBtn").onclick=async()=>{let r=await api("/api/vehicle/check",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({plate:$("#plate").value,direction:$("#dir").value,camera:$("#vcam").value})});$("#plateOut").innerHTML=r.status==="Authorized"?`<div class="success">🟩 ${esc(r.row.Plate)} — AUTHORIZED</div>`:`<div class="alert">🟥 ${esc(r.row.Plate)} — UNAUTHORIZED / UNKNOWN</div>`};
 if(s==="Company Restricted Area")loadFaces();
}
async function loadFaces(){try{let r=await api("/api/faces");$("#faceList").innerHTML=`<h3>Authorized Face Database</h3>${r.authorized.length?table(r.authorized.map(x=>({"Authorized ID":x}))):`<div class="status">No authorized face images found.</div>`}`}catch(e){}}
async function alerts(){
 let d=await api("/api/alerts");app.innerHTML=`<div class="section-title"><h2>🚨 ALERTS & INCIDENTS</h2><button class="btn danger" id="clearAlerts">Clear Demo Alerts</button></div>${metrics([["ACTIVE ALERTS",d.alerts.length,"Current"],["INCIDENTS",d.incidents.length,"Recorded"],["CRITICAL",d.alerts.filter(x=>x.Severity==="Critical").length,"Severity"]])}<h3>Active Alerts</h3>${d.alerts.map(x=>`<div class="alert"><b>${esc(x.Type)}</b> • ${esc(x.Time)} • ${esc(x.Camera)}<br>${esc(x.Message)}</div>`).join("")||`<div class="success">🟢 No active alerts. System is normal.</div>`}<h3>Incident Log</h3>${table(d.incidents)}`;$("#clearAlerts").onclick=async()=>{await api("/api/alerts/clear",{method:"POST"});alerts()};
}
async function securityMap(){
 let d=await api("/api/alerts");let restricted=d.alerts.some(x=>String(x.Zone).includes("Restricted"));app.innerHTML=`<h2>🗺️ 2D SECURITY MAP</h2><div class="map">
 <div class="mz" style="left:5%;top:8%;width:28%;height:35%">ZONE A • MAIN GATE</div><div class="mz" style="left:39%;top:8%;width:25%;height:35%">ZONE B • PARKING</div><div class="mz" style="left:70%;top:8%;width:25%;height:35%">RESTRICTED ZONE</div><div class="mz" style="left:22%;top:55%;width:35%;height:35%">ZONE D • CORRIDOR</div><div class="mz" style="left:64%;top:55%;width:27%;height:35%">RAILWAY / DEMO</div>
 <div class="mc" style="left:18%;top:25%"></div><div class="ml" style="left:18%;top:25%">CAM-01</div><div class="mc" style="left:51%;top:25%"></div><div class="ml" style="left:51%;top:25%">CAM-02</div><div class="mc ${restricted?"ma":""}" style="left:82%;top:25%"></div><div class="ml" style="left:82%;top:25%">CAM-03</div><div class="mc" style="left:39%;top:72%"></div><div class="ml" style="left:39%;top:72%">CAM-04</div></div>`;
}
async function analytics(){
 let d=await api("/api/dashboard");app.innerHTML=`<h2>📈 SECURITY ANALYTICS</h2>${metrics([["PEOPLE",d.people,"Latest AI"],["VEHICLES",d.vehicles,"Latest AI"],["ALERTS",d.alerts,"Current"],["INCIDENTS",d.incidents,"Recorded"]])}<div class="panel"><h3>Current Security Counts</h3><div class="flow">People: <b>${d.people}</b> &nbsp; Vehicles: <b>${d.vehicles}</b> &nbsp; Alerts: <b>${d.alerts}</b> &nbsp; Incidents: <b>${d.incidents}</b></div></div>`;
}
async function videos(){
 app.innerHTML=`<h2>🎬 VIDEO ANALYSIS CENTER</h2><div class="panel"><div class="field"><label>Security Scenario</label>${scenarioSelect("videoScenario")}</div>
 <div class="drop"><p>📁 Upload CCTV test video</p><input id="videoFile" type="file" accept=".mp4,.mov,.avi,.mkv,.webm"></div><div id="videoInfo"></div></div><div id="analysisPanel"></div>`;
 $("#videoFile").onchange=uploadVideo;
}
async function uploadVideo(){
 let f=$("#videoFile").files[0];if(!f)return;$("#videoInfo").innerHTML=`<div class="status">Uploading ${esc(f.name)}...</div>`;
 try{let fd=new FormData();fd.append("file",f);videoInfo=await api("/api/upload-video",{method:"POST",body:fd});$("#videoInfo").innerHTML=`<div class="grid grid4">${metrics([["DURATION",fmt(videoInfo.duration),"Video"],["FPS",videoInfo.fps.toFixed(1),"Original"],["RESOLUTION",`${videoInfo.width} × ${videoInfo.height}`,"Video"],["FRAMES",videoInfo.frames.toLocaleString(),"Total"]])}</div><div class="video-wrap"><video controls src="${URL.createObjectURL(f)}"></video></div><div class="success">🟢 Video uploaded. <b>YOLO has NOT started yet.</b> Click START ANALYSIS.</div>`;
 drawAnalysisPanel();
 }catch(e){$("#videoInfo").innerHTML=`<div class="alert">${esc(e.message)}</div>`}
}
function drawAnalysisPanel(){
 if(!videoInfo)return;let d=videoInfo;
 $("#analysisPanel").innerHTML=`<div class="panel"><h3>⚙️ AI Analysis Settings</h3><div class="form">
 <div class="field"><label>Analysis Range</label><select id="range"><option value="all">Entire Video</option><option value="custom">Custom Range</option></select></div>
 <div class="field"><label>AI Processing FPS</label><select id="aiFps">${[2,3,5,8,10,15].map(x=>`<option ${x===5?"selected":""}>${x}</option>`).join("")}</select></div>
 <div class="field"><label>YOLO Confidence</label><input id="conf" type="number" min=".1" max=".95" step=".05" value=".40"></div>
 <div class="field"><label>Unattended Object Threshold (seconds)</label><input id="unattended" type="number" min="1" max="300" value="30"></div>
 <div id="rangeFields" class="full"></div><div class="full"><button class="btn primary" id="startAnalysis">▶ START ANALYSIS</button></div></div><div id="jobStatus"></div></div>`;
 $("#range").onchange=()=>{$("#rangeFields").innerHTML=$("#range").value==="custom"?`<label class="help">Start / End seconds</label><div class="form"><input id="start" type="number" min="0" max="${d.duration}" value="0"><input id="end" type="number" min="0" max="${d.duration}" value="${d.duration.toFixed(1)}"></div>`:""};
 $("#startAnalysis").onclick=startAnalysis;
}
async function startAnalysis(){
 if(!videoInfo)return;let scenario=$("#videoScenario").value;
 let start=0,end=videoInfo.duration;if($("#range").value==="custom"){start=Number($("#start").value);end=Number($("#end").value)}
 if(end<=start)return notify("Invalid video range.");
 $("#startAnalysis").disabled=true;$("#jobStatus").innerHTML=`<div class="status">Starting AI analysis...</div>`;
 try{let r=await api("/api/analysis/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({video_id:videoInfo.video_id,scenario,start_seconds:start,end_seconds:end,ai_fps:Number($("#aiFps").value),confidence:Number($("#conf").value),unattended_seconds:Number($("#unattended").value)})});currentJob=r.job_id;poller=setInterval(pollJob,700);pollJob()}catch(e){$("#jobStatus").innerHTML=`<div class="alert">${esc(e.message)}</div>`;$("#startAnalysis").disabled=false}
}
async function pollJob(){
 if(!currentJob)return;try{let j=await api("/api/analysis/"+currentJob);let p=(j.progress||0)*100;
 $("#jobStatus").innerHTML=`<div class="panel"><b>${j.status==="complete"?"✅ ANALYSIS COMPLETE":j.status==="error"?"❌ ANALYSIS ERROR":"🤖 YOLO ANALYSIS RUNNING"}</b><div class="progress"><div style="width:${p}%"></div></div><p class="help">${p.toFixed(1)}% • Video ${fmt(j.video_time)} / ${fmt(j.duration)} • AI frames ${j.ai_frames||0} • People ${j.people||0} • Vehicles ${j.vehicles||0} • Bags ${j.bags||0}</p>${j.error?`<div class="alert">${esc(j.error)}</div>`:""}</div>`;
 if(j.status==="complete"){clearInterval(poller);poller=null;$("#jobStatus").innerHTML+=`<div class="video-wrap"><h3>🎥 YOLO PROCESSED VIDEO</h3><video controls autoplay src="${j.video_url}?t=${Date.now()}"></video></div>${j.last_faces?.length?`<div class="panel"><h3>Face Recognition</h3>${j.last_faces.map(f=>f.authorized?`<div class="success">🟩 AUTHORIZED — ${esc(f.id)} (${(f.score*100).toFixed(1)}%)</div>`:`<div class="alert">🟥 UNAUTHORIZED / UNKNOWN — FACE MASKED</div>`).join("")}</div>`:""}`;$("#startAnalysis").disabled=false}
 if(j.status==="error"){clearInterval(poller);poller=null;$("#startAnalysis").disabled=false}
 }catch(e){clearInterval(poller);poller=null;notify(e.message)}
}
async function settings(){
 let s=await api("/api/status");app.innerHTML=`<h2>⚙️ SYSTEM SETTINGS</h2><div class="panel"><h3>Backend</h3><div class="success">FastAPI: ONLINE • http://127.0.0.1:8000</div><h3>AI Model Status</h3>${table(Object.entries(s.models).map(([k,v])=>({"Component":k,"Status":v?"READY":"MISSING"})))}</div><div class="panel"><h3>Important</h3><p class="help">The current source has YOLO vehicle detection and synthetic plate checks. A real plate detector/OCR model is not included, so the HTML frontend does not pretend that OCR is working.</p></div>`;
}
async function checkBackend(){try{let s=await api("/api/status");$("#backendPill").textContent=s.online?"● BACKEND ONLINE":"● BACKEND OFFLINE";$("#backendPill").className="pill"}catch(e){$("#backendPill").textContent="● BACKEND OFFLINE"}}
setInterval(()=>$("#clock").textContent=new Date().toLocaleString(),1000);checkBackend();render();
