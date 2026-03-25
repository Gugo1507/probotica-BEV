import av
import cv2
import numpy as np
import json
import os
import threading
import asyncio
import fractions
import time
import signal
import sys
import logging

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

# ─── Load intrinsics ──────────────────────────────────────────────────────────
with open("camera_intrinsics.json") as f:
    cal = json.load(f)

K     = np.array(cal["camera_matrix"])
dist  = np.array(cal["dist_coefficients"])
cal_w, cal_h = cal["calibration_info"]["image_size_wh"]

map1, map2 = None, None
last_size  = None
scaled_roi = None

def build_maps(w, h, alpha=0.1):
    sx, sy = w / cal_w, h / cal_h
    sK = K.copy()
    sK[0] *= sx
    sK[1] *= sy
    new_K, roi = cv2.getOptimalNewCameraMatrix(sK, dist, (w, h), alpha=alpha, newImgSize=(w, h))
    m1, m2 = cv2.initUndistortRectifyMap(sK, dist, None, new_K, (w, h), cv2.CV_16SC2)
    return m1, m2, roi

def inpaint_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray == 0).astype(np.uint8) * 255
    if mask.any():
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return img

# ─── Shared state ─────────────────────────────────────────────────────────────
rtsp_frame   = None
usb_frame    = None
display_out  = None
display_lock = threading.Lock()
running      = True
stop_signal  = False

# ─── Controls (toggled via browser) ──────────────────────────────────────────
rectify     = True
use_inpaint = False
alpha       = 0.1
show_usb    = True

# ─── Stats ────────────────────────────────────────────────────────────────────
fps_val      = 0.0
rtsp_ok      = False
usb_ok       = False

DEVICE_IP   = "0.0.0.0"       # bind address
DISPLAY_IP  = "192.168.88.251" # shown in browser UI
WEBRTC_PORT = 8080
STREAM_FPS  = 30

# ─── RTSP Thread ─────────────────────────────────────────────────────────────
def rtsp_worker():
    global rtsp_frame, running, rtsp_ok
    try:
        container = av.open(
            "rtsp://admin:123456@192.168.88.225:554/stream1",
            options={"rtsp_transport": "tcp", "fflags": "nobuffer"}
        )
        for frame in container.decode(video=0):
            if not running:
                break
            try:
                img = frame.to_ndarray(format='bgr24')
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                rtsp_frame = img
                rtsp_ok = True
            except:
                continue
    except Exception as e:
        print(f"[RTSP ERROR] {e}")
        rtsp_ok = False

# ─── USB Thread ──────────────────────────────────────────────────────────────
def usb_worker():
    global usb_frame, running, usb_ok
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print("[WARN] USB camera not found")
        usb_ok = False
        return
    usb_ok = True
    while running:
        ret, frame = cap.read()
        if ret:
            usb_frame = frame
    cap.release()

# ─── Compositor Thread ────────────────────────────────────────────────────────
def compositor_thread():
    global display_out, map1, map2, last_size, scaled_roi
    global fps_val

    fps_count = 0
    fps_timer = time.time()

    while not stop_signal:
        if rtsp_frame is None:
            time.sleep(0.005)
            continue

        img = rtsp_frame.copy()
        h, w = img.shape[:2]

        # Rebuild maps if needed
        if (w, h, alpha) != last_size:
            map1, map2, scaled_roi = build_maps(w, h, alpha)
            last_size = (w, h, alpha)

        # Rectify RTSP
        if rectify and map1 is not None:
            img = cv2.remap(img, map1, map2,
                            interpolation=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
            if scaled_roi is not None:
                x, y, rw, rh = scaled_roi
                if rw > 10 and rh > 10:
                    img = img[y:y+rh, x:x+rw]
            if use_inpaint:
                img = inpaint_borders(img)

        # USB panel
        usb = None
        if show_usb and usb_frame is not None:
            usb = usb_frame.copy()
            uh, uw = usb.shape[:2]
            scale = img.shape[0] / uh
            usb = cv2.resize(usb, (int(uw * scale), img.shape[0]))

        # Compose side by side
        if usb is not None:
            combined = np.hstack((img, usb))
        else:
            combined = img

        # Force even dimensions for H.264
        ch, cw = combined.shape[:2]
        combined = combined[:ch - ch%2, :cw - cw%2]

        # FPS
        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_val   = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

        with display_lock:
            display_out = combined

        time.sleep(0.001)

# ─── WebRTC Track ─────────────────────────────────────────────────────────────
class CameraTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self._pts   = 0
        self._start = None

    async def recv(self):
        if self._start is None:
            self._start = time.time()
        target = self._start + self._pts / STREAM_FPS
        wait   = target - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        with display_lock:
            frame = display_out

        if frame is None:
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        av_frame           = VideoFrame.from_ndarray(rgb, format="rgb24")
        av_frame.pts       = self._pts
        av_frame.time_base = fractions.Fraction(1, STREAM_FPS)
        self._pts         += 1
        return av_frame

# ─── HTML UI ─────────────────────────────────────────────────────────────────
_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Camera Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:       #090b0f;
    --surface:  #0e1219;
    --panel:    #111620;
    --border:   #1e2840;
    --accent:   #00c8a0;
    --accent2:  #0088ff;
    --warn:     #ff6b35;
    --ok:       #00e87a;
    --off:      #2a3040;
    --text:     #c8d4e8;
    --muted:    #445060;
    --font-ui:  'Rajdhani', sans-serif;
    --font-mono:'Share Tech Mono', monospace;
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-ui);
    font-size: 14px;
    letter-spacing: 0.03em;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }}

  /* ── Header ── */
  header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    height: 48px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }}

  .logo {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
  }}

  .logo-icon {{
    width: 28px; height: 28px;
    border: 2px solid var(--accent);
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    position: relative;
  }}
  .logo-icon::before {{
    content: '';
    width: 10px; height: 10px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%,100% {{ opacity: 1; transform: scale(1); }}
    50%      {{ opacity: 0.5; transform: scale(0.7); }}
  }}

  .header-right {{
    display: flex;
    align-items: center;
    gap: 20px;
  }}

  .stat-group {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 12px;
  }}

  .stat-label {{ color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }}
  .stat-value {{ color: var(--accent); }}

  #conn-badge {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 2px;
    text-transform: uppercase;
    background: var(--off);
    color: var(--muted);
    border: 1px solid var(--border);
    transition: all 0.3s;
  }}
  #conn-badge.ok  {{ background: rgba(0,232,122,0.1); color: var(--ok);   border-color: var(--ok);   }}
  #conn-badge.err {{ background: rgba(255,107,53,0.1); color: var(--warn); border-color: var(--warn); }}

  /* ── Main layout ── */
  .main {{
    display: grid;
    grid-template-columns: 1fr 240px;
    grid-template-rows: 1fr;
    flex: 1;
    gap: 0;
    overflow: hidden;
  }}

  /* ── Video panel ── */
  .video-wrap {{
    position: relative;
    background: #000;
    overflow: hidden;
    border-right: 1px solid var(--border);
  }}

  video {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    background: #000;
  }}

  /* Scan-line overlay */
  .video-wrap::after {{
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.05) 2px,
      rgba(0,0,0,0.05) 4px
    );
    pointer-events: none;
  }}

  /* Corner brackets */
  .corner {{
    position: absolute;
    width: 20px; height: 20px;
    border-color: var(--accent);
    border-style: solid;
    opacity: 0.6;
    z-index: 2;
  }}
  .corner.tl {{ top:12px; left:12px;  border-width: 2px 0 0 2px; }}
  .corner.tr {{ top:12px; right:12px; border-width: 2px 2px 0 0; }}
  .corner.bl {{ bottom:12px; left:12px;  border-width: 0 0 2px 2px; }}
  .corner.br {{ bottom:12px; right:12px; border-width: 0 2px 2px 0; }}

  /* Live dot */
  .live-dot {{
    position: absolute;
    top: 18px; left: 50%;
    transform: translateX(-50%);
    display: flex; align-items: center; gap: 6px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--warn);
    letter-spacing: 0.12em;
    z-index: 2;
    opacity: 0;
    transition: opacity 0.5s;
  }}
  .live-dot.visible {{ opacity: 1; }}
  .live-dot::before {{
    content: '';
    width: 7px; height: 7px;
    background: var(--warn);
    border-radius: 50%;
    animation: blink 1s step-end infinite;
  }}
  @keyframes blink {{ 50% {{ opacity: 0; }} }}

  /* No-signal overlay */
  #no-signal {{
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    background: var(--bg);
    z-index: 3;
    transition: opacity 0.4s;
  }}
  #no-signal.hidden {{ opacity: 0; pointer-events: none; }}
  .ns-icon {{
    width: 64px; height: 64px;
    border: 2px solid var(--muted);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: var(--muted);
    font-size: 28px;
  }}
  #no-signal p {{
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }}

  /* ── Sidebar ── */
  .sidebar {{
    display: flex;
    flex-direction: column;
    background: var(--surface);
    overflow-y: auto;
  }}

  .sidebar::-webkit-scrollbar {{ width: 4px; }}
  .sidebar::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}

  .section {{
    border-bottom: 1px solid var(--border);
    padding: 14px 16px;
  }}

  .section-title {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }}

  /* Indicators */
  .indicator-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 13px;
  }}
  .indicator-label {{ color: var(--text); }}
  .dot-status {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--off);
    transition: background 0.3s;
  }}
  .dot-status.active {{ background: var(--ok); box-shadow: 0 0 6px var(--ok); }}
  .dot-status.error  {{ background: var(--warn); box-shadow: 0 0 6px var(--warn); }}

  /* Toggle buttons */
  .toggle-btn {{
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 10px;
    border-radius: 3px;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--text);
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.05em;
    cursor: pointer;
    margin-bottom: 7px;
    transition: all 0.2s;
  }}
  .toggle-btn:hover {{ border-color: var(--accent); }}
  .toggle-btn.active {{
    border-color: var(--accent);
    background: rgba(0,200,160,0.08);
    color: var(--accent);
  }}
  .toggle-btn .pill-state {{
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 2px 6px;
    border-radius: 2px;
    background: var(--off);
    color: var(--muted);
  }}
  .toggle-btn.active .pill-state {{
    background: rgba(0,200,160,0.15);
    color: var(--accent);
  }}

  /* Alpha slider */
  .slider-row {{
    margin-bottom: 6px;
  }}
  .slider-header {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
    font-size: 12px;
  }}
  .slider-name {{ color: var(--text); }}
  .slider-val  {{ font-family: var(--font-mono); color: var(--accent); font-size: 12px; }}

  input[type=range] {{
    -webkit-appearance: none;
    width: 100%;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    outline: none;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    box-shadow: 0 0 6px rgba(0,200,160,0.4);
  }}

  /* Action buttons */
  .action-btn {{
    width: 100%;
    padding: 9px 12px;
    border-radius: 3px;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--text);
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    margin-bottom: 7px;
    transition: all 0.2s;
    text-align: left;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .action-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
  .action-btn.primary {{
    border-color: var(--accent);
    background: rgba(0,200,160,0.1);
    color: var(--accent);
  }}
  .action-btn.primary:hover {{ background: rgba(0,200,160,0.2); }}
  .action-btn.danger {{ border-color: var(--warn); color: var(--warn); background: rgba(255,107,53,0.06); }}
  .action-btn.danger:hover {{ background: rgba(255,107,53,0.15); }}
  .action-btn:disabled {{ opacity: 0.35; cursor: default; border-color: var(--border); color: var(--muted); background: transparent; }}

  /* Log */
  #log {{
    flex: 1;
    padding: 10px 12px;
    overflow-y: auto;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
    min-height: 80px;
  }}
  #log::-webkit-scrollbar {{ width: 3px; }}
  #log::-webkit-scrollbar-thumb {{ background: var(--border); }}
  .log-line {{ padding: 1px 0; line-height: 1.6; }}
  .log-line.info  {{ color: var(--text); }}
  .log-line.ok    {{ color: var(--ok); }}
  .log-line.error {{ color: var(--warn); }}

  /* FPS bar */
  .fps-bar {{
    height: 2px;
    background: var(--border);
    border-radius: 1px;
    margin-top: 6px;
    overflow: hidden;
  }}
  .fps-fill {{
    height: 100%;
    background: var(--accent);
    border-radius: 1px;
    transition: width 0.5s ease;
    width: 0%;
  }}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon"></div>
    Camera Monitor — {DISPLAY_IP}
  </div>
  <div class="header-right">
    <div class="stat-group">
      <span class="stat-label">FPS</span>
      <span class="stat-value" id="fps-hdr">--.-</span>
    </div>
    <div class="stat-group">
      <span class="stat-label">LATENCY</span>
      <span class="stat-value" id="lat-hdr">-- ms</span>
    </div>
    <span id="conn-badge">OFFLINE</span>
  </div>
</header>

<div class="main">

  <!-- Video -->
  <div class="video-wrap">
    <div class="corner tl"></div>
    <div class="corner tr"></div>
    <div class="corner bl"></div>
    <div class="corner br"></div>
    <div class="live-dot" id="live-dot">LIVE</div>
    <div id="no-signal">
      <div class="ns-icon">⊘</div>
      <p>No signal — connect to stream</p>
    </div>
    <video id="v" autoplay playsinline muted></video>
  </div>

  <!-- Sidebar -->
  <div class="sidebar">

    <!-- Sources -->
    <div class="section">
      <div class="section-title">Sources</div>
      <div class="indicator-row">
        <span class="indicator-label">RTSP / Ethernet</span>
        <div class="dot-status" id="dot-rtsp"></div>
      </div>
      <div class="indicator-row">
        <span class="indicator-label">USB Camera</span>
        <div class="dot-status" id="dot-usb"></div>
      </div>
    </div>

    <!-- Processing -->
    <div class="section">
      <div class="section-title">Processing</div>

      <button class="toggle-btn active" id="btn-rectify" onclick="toggle('rectify')">
        Lens Rectify <span class="pill-state">ON</span>
      </button>
      <button class="toggle-btn" id="btn-inpaint" onclick="toggle('inpaint')">
        Inpaint Borders <span class="pill-state">OFF</span>
      </button>
      <button class="toggle-btn active" id="btn-usb" onclick="toggle('usb')">
        Show USB Feed <span class="pill-state">ON</span>
      </button>

      <div class="slider-row" style="margin-top:12px">
        <div class="slider-header">
          <span class="slider-name">Crop Alpha</span>
          <span class="slider-val" id="alpha-val">0.10</span>
        </div>
        <input type="range" id="alpha-slider" min="0" max="100" value="10"
               oninput="setAlpha(this.value)">
        <div class="fps-bar"><div class="fps-fill" id="alpha-fill" style="width:10%"></div></div>
      </div>
    </div>

    <!-- Connection -->
    <div class="section">
      <div class="section-title">Connection</div>
      <button class="action-btn primary" id="btn-connect" onclick="connect()">▶ Connect</button>
      <button class="action-btn danger"  id="btn-disc"    onclick="disconnect()" disabled>■ Disconnect</button>
      <button class="action-btn"                          onclick="fullscreen()">⛶ Fullscreen</button>
      <button class="action-btn"                          onclick="snapshot()">⊙ Snapshot</button>
    </div>

    <!-- Log -->
    <div class="section-title" style="padding:10px 16px 4px; color:var(--muted); font-size:10px; letter-spacing:0.2em; text-transform:uppercase;">Event Log</div>
    <div id="log"></div>

  </div>
</div>

<script>
let pc = null;
let statsTimer = null;
let t0stream = null;

const $ = id => document.getElementById(id);

function log(msg, type='') {{
  const d = $('log');
  const ts = new Date().toTimeString().slice(0,8);
  d.innerHTML += `<div class="log-line ${{type}}">[${{ts}}] ${{msg}}</div>`;
  d.scrollTop = d.scrollHeight;
  if (d.children.length > 80) d.removeChild(d.firstChild);
}}

function setConnBadge(state) {{
  const b = $('conn-badge');
  b.textContent = state;
  b.className = state === 'LIVE' ? 'ok' : state === 'ERROR' ? 'err' : '';
}}

function setLive(on) {{
  $('live-dot').classList.toggle('visible', on);
  $('no-signal').classList.toggle('hidden', on);
}}

// Poll server state
async function pollState() {{
  try {{
    const j = await (await fetch('/state')).json();
    $('fps-hdr').textContent = j.fps.toFixed(1);
    $('fps-fill') && ($('fps-fill').style.width = Math.min(j.fps/60*100,100)+'%');
    $('dot-rtsp').className = 'dot-status' + (j.rtsp ? ' active' : ' error');
    $('dot-usb').className  = 'dot-status' + (j.usb  ? ' active' : ' error');
    _syncBtn('btn-rectify', j.rectify,     'Lens Rectify');
    _syncBtn('btn-inpaint', j.inpaint,     'Inpaint Borders');
    _syncBtn('btn-usb',     j.show_usb,    'Show USB Feed');
    $('alpha-val').textContent  = j.alpha.toFixed(2);
    $('alpha-slider').value     = Math.round(j.alpha * 100);
    $('alpha-fill').style.width = Math.round(j.alpha * 100) + '%';
  }} catch(e) {{}}
}}
setInterval(pollState, 1000);
pollState();

function _syncBtn(id, state, label) {{
  const b = $(id);
  b.classList.toggle('active', state);
  b.querySelector('.pill-state').textContent = state ? 'ON' : 'OFF';
}}

async function toggle(what) {{
  try {{
    const r = await fetch('/toggle/' + what, {{ method: 'POST' }});
    const j = await r.json();
    log(what.toUpperCase() + ' → ' + (j.value ? 'ON' : 'OFF'), j.value ? 'ok' : '');
  }} catch(e) {{ log('Toggle error: ' + e, 'error'); }}
}}

async function setAlpha(v) {{
  $('alpha-val').textContent  = (v/100).toFixed(2);
  $('alpha-fill').style.width = v + '%';
  try {{
    await fetch('/alpha/' + v, {{ method: 'POST' }});
  }} catch(e) {{}}
}}

function _strip_mdns(sdp) {{
  return sdp.split('\\n').filter(l => !(l.startsWith('a=candidate:') && l.includes('.local'))).join('\\n');
}}

async function connect() {{
  $('btn-connect').disabled = true;
  $('btn-disc').disabled    = false;
  setConnBadge('CONNECTING');
  log('Creating RTCPeerConnection…');

  pc = new RTCPeerConnection({{ iceServers: [{{ urls: 'stun:stun.l.google.com:19302' }}] }});

  pc.onicecandidate = e => {{
    if (e.candidate) log('ICE: ' + e.candidate.type);
  }};

  pc.onconnectionstatechange = () => {{
    const s = pc.connectionState;
    log('Connection: ' + s, s === 'connected' ? 'ok' : s === 'failed' ? 'error' : '');
    if (s === 'connected')  {{ setConnBadge('LIVE');   setLive(true);  t0stream = performance.now(); }}
    else if (s === 'failed') {{ setConnBadge('ERROR'); setLive(false); disconnect(); }}
    else if (s === 'closed') {{ setConnBadge('OFFLINE'); setLive(false); }}
  }};

  pc.ontrack = e => {{
    log('Video track received ✓', 'ok');
    $('v').srcObject = e.streams[0];
    startLatencyPoll();
  }};

  pc.addTransceiver('video', {{ direction: 'recvonly' }});
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  log('Gathering ICE candidates…');

  await new Promise(resolve => {{
    if (pc.iceGatheringState === 'complete') {{ resolve(); return; }}
    const t = setTimeout(resolve, 3000);
    pc.onicegatheringstatechange = () => {{
      if (pc.iceGatheringState === 'complete') {{ clearTimeout(t); resolve(); }}
    }};
  }});

  try {{
    const r = await fetch('/offer', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ sdp: _strip_mdns(pc.localDescription.sdp), type: pc.localDescription.type }})
    }});
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const ans = await r.json();
    await pc.setRemoteDescription(ans);
    log('Remote description set — awaiting media…', 'info');
  }} catch(e) {{
    log('Offer error: ' + e, 'error');
    setConnBadge('ERROR');
    disconnect();
  }}
}}

function startLatencyPoll() {{
  if (statsTimer) clearInterval(statsTimer);
  statsTimer = setInterval(async () => {{
    if (!pc) return;
    try {{
      const stats = await pc.getStats();
      stats.forEach(r => {{
        if (r.type === 'inbound-rtp' && r.kind === 'video') {{
          const jitterMs = (r.jitter * 1000).toFixed(0);
          $('lat-hdr').textContent = jitterMs + ' ms';
        }}
      }});
    }} catch(e) {{}}
  }}, 1000);
}}

function disconnect() {{
  if (statsTimer) {{ clearInterval(statsTimer); statsTimer = null; }}
  if (pc) {{ pc.close(); pc = null; }}
  $('v').srcObject = null;
  $('btn-connect').disabled = false;
  $('btn-disc').disabled    = true;
  setConnBadge('OFFLINE');
  setLive(false);
  log('Disconnected');
}}

function fullscreen() {{
  document.querySelector('.video-wrap').requestFullscreen?.();
}}

function snapshot() {{
  window.open('/snapshot');
  log('Snapshot requested');
}}
</script>
</body>
</html>"""

# ─── WebRTC Server ────────────────────────────────────────────────────────────
pcs = set()

async def handle_index(request):
    return web.Response(content_type="text/html", text=_HTML)

async def handle_offer(request):
    params = await request.json()
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"[WebRTC] {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close(); pcs.discard(pc)

    pc.addTrack(CameraTrack())
    await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def handle_toggle(request):
    global rectify, use_inpaint, show_usb
    what = request.match_info["what"]
    if   what == "rectify": rectify     = not rectify;     val = rectify
    elif what == "inpaint": use_inpaint = not use_inpaint; val = use_inpaint
    elif what == "usb":     show_usb    = not show_usb;    val = show_usb
    else: return web.Response(status=400)
    print(f"  [{what}] {'ON' if val else 'OFF'}")
    return web.json_response({"value": val})

async def handle_alpha(request):
    global alpha, last_size
    v = int(request.match_info["v"])
    alpha = round(v / 100, 2)
    last_size = None   # force remap rebuild
    return web.json_response({"alpha": alpha})

async def handle_state(request):
    return web.json_response({
        "rectify":  rectify,
        "inpaint":  use_inpaint,
        "show_usb": show_usb,
        "alpha":    alpha,
        "fps":      fps_val,
        "rtsp":     rtsp_ok,
        "usb":      usb_ok,
    })

async def handle_snapshot(request):
    with display_lock:
        frame = display_out
    if frame is None:
        return web.Response(status=503, text="No frame yet")
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return web.Response(body=jpg.tobytes(), content_type="image/jpeg")

async def webrtc_main():
    app = web.Application()
    app.router.add_get("/",                handle_index)
    app.router.add_post("/offer",          handle_offer)
    app.router.add_post("/toggle/{what}",  handle_toggle)
    app.router.add_post("/alpha/{v}",      handle_alpha)
    app.router.add_get("/state",           handle_state)
    app.router.add_get("/snapshot",        handle_snapshot)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, DEVICE_IP, WEBRTC_PORT).start()
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  Stream ready → http://{DISPLAY_IP}:{WEBRTC_PORT}  │")
    print(f"  │  /snapshot → JPEG grab                      │")
    print(f"  └─────────────────────────────────────────────┘\n")
    while not stop_signal:
        await asyncio.sleep(0.5)

def run_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:    loop.run_until_complete(webrtc_main())
    finally: loop.close()

def signal_handler(sig, frame):
    global running, stop_signal
    running = stop_signal = True
    time.sleep(0.3)
    sys.exit(0)

# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    threading.Thread(target=rtsp_worker,      daemon=True).start()
    threading.Thread(target=usb_worker,       daemon=True).start()
    threading.Thread(target=compositor_thread, daemon=True).start()
    threading.Thread(target=run_server,       daemon=True).start()

    print("Camera Monitor starting…  Ctrl+C to quit")
    while not stop_signal:
        time.sleep(5)
        print(f"  FPS={fps_val:.1f}  peers={len(pcs)}  "
              f"RTSP={'OK' if rtsp_ok else 'ERR'}  "
              f"USB={'OK' if usb_ok else 'ERR'}")