"""
WebRTC BEV surround-view stream — 4 cameras, remap warp, blend, stream only.
No cv2.imshow. Browser at http://192.168.88.251:8080
Layout: [BEV] | [FRONT RAW / BACK RAW]
Controls via browser UI: BLEND  COLORBAL  SNAPSHOT
"""
import pyzed.sl as sl
import cv2
import numpy as np
import threading, time, signal, asyncio, fractions, logging, sys, os

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

stop_signal = False

# ── Scale / layout ─────────────────────────────────────────────────────
SCALE = 0.15

SQ      = 45;  COLS = 6;  ROWS = 4
BOARD_W = SQ * COLS
BOARD_H = SQ * ROWS
GAP_FB  = 270;  GAP_LR = 270
SHIFT   = 2000
VEH_W   = 490;  VEH_L  = 740
BW_LR   = BOARD_H

BEV_W_FULL = SHIFT + BW_LR + GAP_LR + VEH_W + GAP_LR + BW_LR + SHIFT
BEV_H_FULL = SHIFT + BOARD_H + GAP_FB + VEH_L + GAP_FB + BOARD_H + SHIFT

BEV_W = int(BEV_W_FULL * SCALE); BEV_W += BEV_W % 2
BEV_H = int(BEV_H_FULL * SCALE); BEV_H += BEV_H % 2

VEH_X   = int((SHIFT + BW_LR + GAP_LR) * SCALE)
VEH_Y   = int((SHIFT + BOARD_H + GAP_FB) * SCALE)
VEH_W_S = int(VEH_W * SCALE)
VEH_L_S = int(VEH_L * SCALE)

xl = VEH_X;         xr = VEH_X + VEH_W_S
yt = VEH_Y;         yb = VEH_Y + VEH_L_S

CAMERAS = [
    {"side": "front", "serial": 50542844,  "type": "stereo", "name": "ZED X Mini"},
    {"side": "back",  "serial": 40685614,  "type": "stereo", "name": "ZED X"},
    {"side": "right", "serial": 308745927, "type": "mono",   "name": "ZED XOne GS #1"},
    {"side": "left",  "serial": 304788437, "type": "mono",   "name": "ZED XOne GS #2"},
]

VALID = {
    "front": (0,  0,    BEV_W, yt),
    "back":  (0,  yb,   BEV_W, BEV_H),
    "left":  (0,  0,    xl,    BEV_H),
    "right": (xr, 0,    BEV_W, BEV_H),
}

INPUT_SCALE  = 1.0
SRC_W, SRC_H = 1920, 1200
STREAM_FPS   = 20

# ── Display canvas layout constants ───────────────────────────────────
# Same visual layout as the OpenCV window:
#  ┌──────────────────────┬──────────────────┐
#  │                      │  FRONT CAM RAW   │
#  │   BEV 360°           ├──────────────────┤
#  │                      │  BACK CAM  RAW   │
#  └──────────────────────┴──────────────────┘
C_BG         = (18,  18,  18)
C_BORDER     = (0,   180, 160)
C_BORDER_DIM = (0,   80,  70)
C_LABEL_BG   = (0,   0,   0)
C_ACCENT     = (0,   220, 180)
C_WARN       = (30,  150, 255)
C_ON         = (60,  220, 100)
C_OFF        = (80,  80,  80)

BORDER   = 2
LABEL_H  = 22
GUTTER   = 4
FONT     = cv2.FONT_HERSHEY_PLAIN

def _draw_label(canvas, x, y, w, text, color, bg=None):
    if bg:
        cv2.rectangle(canvas, (x, y), (x+w-1, y+LABEL_H-1), bg, -1)
    cv2.rectangle(canvas, (x, y), (x+w-1, y+LABEL_H-1), color, 1)
    sz = cv2.getTextSize(text, FONT, 1.0, 1)[0]
    cv2.putText(canvas, text,
                (x + (w - sz[0])//2, y + LABEL_H - 6),
                FONT, 1.0, color, 1, cv2.LINE_AA)

def _pill(canvas, x, y, label, state):
    color = C_ON if state else C_OFF
    text  = f"{label}: {'ON' if state else 'OFF'}"
    sz    = cv2.getTextSize(text, FONT, 0.85, 1)[0]
    pw, ph = sz[0]+14, sz[1]+8
    cv2.rectangle(canvas, (x, y), (x+pw, y+ph), color, 1)
    cv2.putText(canvas, text, (x+7, y+ph-4), FONT, 0.85, color, 1, cv2.LINE_AA)
    return pw + 8

def compose_display(bev, front_raw, back_raw, fps,
                    enable_blend, enable_color_balance):
    """
    Build a single BGR numpy frame with the layout:
      HUD bar (fps + status pills)
      Left col : BEV composite
      Right col: FRONT raw (top) / BACK raw (bottom)
    All dimensions are forced even for H.264.
    """
    BEV_CONTENT_H = BEV_H
    BEV_PH  = LABEL_H + BEV_CONTENT_H
    BEV_PW  = BEV_W

    CAM_PH  = BEV_PH // 2
    # Camera content area keeps 16:10 aspect of 1920×1200
    CAM_CONTENT_H = CAM_PH - LABEL_H
    CAM_PW  = int(CAM_CONTENT_H * SRC_W / SRC_H)
    CAM_PW += CAM_PW % 2   # keep even

    HUD_H   = 38
    TOTAL_W = BORDER + BEV_PW + GUTTER + CAM_PW + BORDER
    TOTAL_H = HUD_H  + BORDER + BEV_PH + BORDER
    # Force even
    TOTAL_W += TOTAL_W % 2
    TOTAL_H += TOTAL_H % 2

    canvas = np.full((TOTAL_H, TOTAL_W, 3), C_BG, dtype=np.uint8)

    # ── HUD ────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0,0), (TOTAL_W-1, HUD_H-1), (25,25,25), -1)
    cv2.line(canvas, (0, HUD_H-1), (TOTAL_W-1, HUD_H-1), C_BORDER, 1)
    cv2.putText(canvas, f"FPS {fps:05.1f}", (10, HUD_H-10),
                FONT, 1.1, C_ACCENT, 1, cv2.LINE_AA)
    adv = _pill(canvas, 120, 8, "BLEND",    enable_blend)
    _pill(canvas,        120+adv, 8, "COLORBAL", enable_color_balance)
    hint = "Controls in browser"
    hsz  = cv2.getTextSize(hint, FONT, 0.85, 1)[0]
    cv2.putText(canvas, hint, (TOTAL_W-hsz[0]-10, HUD_H-10),
                FONT, 0.85, (80,80,80), 1, cv2.LINE_AA)

    # ── BEV panel ──────────────────────────────────────────────────────
    bx = BORDER
    by = HUD_H + BORDER
    _draw_label(canvas, bx, by, BEV_PW, "BIRD'S-EYE VIEW  360°",
                C_ACCENT, C_LABEL_BG)
    bcy = by + LABEL_H
    if bev is not None:
        b = cv2.resize(bev, (BEV_PW, BEV_CONTENT_H), interpolation=cv2.INTER_LINEAR)
        canvas[bcy:bcy+BEV_CONTENT_H, bx:bx+BEV_PW] = b
    cv2.rectangle(canvas, (bx, by), (bx+BEV_PW-1, by+BEV_PH-1), C_BORDER, BORDER)

    # ── Camera panels ──────────────────────────────────────────────────
    cx = BORDER + BEV_PW + GUTTER
    for i, (label, raw) in enumerate([("FRONT CAM — RAW", front_raw),
                                       ("BACK  CAM — RAW", back_raw)]):
        cy  = HUD_H + BORDER + i * (CAM_PH + GUTTER)
        col = C_WARN if raw is not None else C_BORDER_DIM
        _draw_label(canvas, cx, cy, CAM_PW, label, col, C_LABEL_BG)
        ccy = cy + LABEL_H
        avh = min(CAM_CONTENT_H, TOTAL_H - ccy)
        avw = min(CAM_PW, TOTAL_W - cx)
        if raw is not None:
            r = cv2.resize(raw, (avw, avh), interpolation=cv2.INTER_LINEAR)
            canvas[ccy:ccy+avh, cx:cx+avw] = r
        else:
            cv2.putText(canvas, "NO SIGNAL",
                        (cx + CAM_PW//2 - 40, ccy + avh//2),
                        FONT, 1.0, (60,60,60), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (cx, cy), (cx+CAM_PW-1, cy+CAM_PH-1), col, BORDER)

    return canvas

# ── Runtime toggles ────────────────────────────────────────────────────
enable_blend         = True
enable_color_balance = False

DEVICE_IP   = "192.168.88.251"
WEBRTC_PORT = 8080

# ── Shared display output ──────────────────────────────────────────────
display_out      = None
display_out_lock = threading.Lock()

# ── Shared BEV output (for snapshot endpoint) ─────────────────────────
bev_out      = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
bev_out_lock = threading.Lock()

# ── Per-camera shared state ────────────────────────────────────────────
bev_frames  = {}
bev_lock    = {}
bev_ts      = {}
raw_frames  = {}          # front/back raw BGR for side panels
raw_lock    = {}
cam_handles = []
cam_frames  = []
cam_lock    = []
cam_gains   = {s: np.ones(3, dtype=np.float32)
               for s in ("front","back","left","right")}


def signal_handler(sig, frame):
    global stop_signal
    stop_signal = True
    time.sleep(0.5)
    sys.exit(0)


# ── Remap ──────────────────────────────────────────────────────────────
def build_remap(H_full, input_scale, output_scale, src_w, src_h):
    S_in  = np.diag([input_scale,  input_scale,  1.0])
    S_out = np.diag([output_scale, output_scale, 1.0])
    H_s   = S_out @ H_full @ np.linalg.inv(S_in)
    H_inv = np.linalg.inv(H_s)
    ys, xs = np.mgrid[0:BEV_H, 0:BEV_W]
    coords = np.stack([xs, ys, np.ones_like(xs)], 0).reshape(3,-1).astype(np.float64)
    sc = H_inv @ coords;  sc /= sc[2:3]
    mx = sc[0].reshape(BEV_H, BEV_W).astype(np.float32)
    my = sc[1].reshape(BEV_H, BEV_W).astype(np.float32)
    sw, sh = int(src_w*input_scale), int(src_h*input_scale)
    oob = (mx<0)|(mx>=sw)|(my<0)|(my>=sh)
    mx[oob] = -1;  my[oob] = -1
    return mx, my

def bake_clip(mx, my, valid_rect):
    x0,y0,x1,y1 = valid_rect
    m = np.zeros((BEV_H, BEV_W), dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    out = m == 0
    mx[out] = -1;  my[out] = -1


# ── Colour balance ─────────────────────────────────────────────────────
def compute_gains(frames):
    means = {}
    for side, f in frames.items():
        if f is None: continue
        mask = f.sum(axis=2) > 15
        if mask.sum() < 100: continue
        means[side] = f[mask].astype(np.float32).mean(axis=0)
    if len(means) < 4: return
    gm = np.mean(list(means.values()), axis=0)
    for side, m in means.items():
        cam_gains[side] = np.clip(
            np.where(m > 1.0, gm / m, 1.0).astype(np.float32), 0.5, 2.0)

def apply_gain(bgr, gain):
    out = bgr.astype(np.float32)
    out[...,0]*=gain[0]; out[...,1]*=gain[1]; out[...,2]*=gain[2]
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Grab + warp threads ────────────────────────────────────────────────
def _warp_store(bgra, mx, my, side, ts, input_scale):
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    # Store raw full-res frame for front/back side panels
    if side in ("front", "back"):
        with raw_lock[side]:
            raw_frames[side] = bgr
    if input_scale != 1.0:
        bgr = cv2.resize(bgr,
            (int(bgr.shape[1]*input_scale), int(bgr.shape[0]*input_scale)),
            interpolation=cv2.INTER_LINEAR)
    w = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    with bev_lock[side]:
        bev_frames[side] = w
        bev_ts[side]     = ts

def grab_stereo(idx, side, mx, my):
    runtime = sl.RuntimeParameters()
    cam     = cam_handles[idx]
    last_ts = 0
    while not stop_signal:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            ts = cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
            if ts <= last_ts: time.sleep(0.001); continue
            last_ts = ts
            with cam_lock[idx]:
                cam.retrieve_image(cam_frames[idx], sl.VIEW.LEFT)
                raw = cam_frames[idx].get_data().copy()
            _warp_store(raw, mx, my, side, ts, INPUT_SCALE)
        else:
            time.sleep(0.001)
    cam.close()

def grab_mono(idx, side, mx, my):
    cam     = cam_handles[idx]
    last_ts = 0
    use_arg = None
    try:    rp = sl.RuntimeParametersOne(); has_rp = True
    except: has_rp = False
    while not stop_signal:
        if use_arg is None:
            try:
                err = cam.grab(rp) if has_rp else cam.grab(sl.RuntimeParameters())
                use_arg = True
            except TypeError:
                err = cam.grab(); use_arg = False
        elif use_arg:
            err = cam.grab(rp) if has_rp else cam.grab(sl.RuntimeParameters())
        else:
            err = cam.grab()
        if err == sl.ERROR_CODE.SUCCESS:
            ts = cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
            if ts <= last_ts: time.sleep(0.001); continue
            last_ts = ts
            with cam_lock[idx]:
                cam.retrieve_image(cam_frames[idx], sl.VIEW.LEFT)
                raw = cam_frames[idx].get_data().copy()
            _warp_store(raw, mx, my, side, ts, INPUT_SCALE)
        else:
            time.sleep(0.001)
    cam.close()


# ── Blend ──────────────────────────────────────────────────────────────
def compute_weight_matrix(imA, imB):
    def gmask(img):
        _, m = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                             0, 255, cv2.THRESH_BINARY)
        return m
    ov  = cv2.bitwise_and(imA, imB)
    ovm = cv2.dilate(gmask(ov), np.ones((2,2),np.uint8), iterations=2)
    inv = cv2.bitwise_not(ovm)
    dA  = gmask(cv2.bitwise_and(imA, imA, mask=inv))
    dB  = gmask(cv2.bitwise_and(imB, imB, mask=inv))
    distA = cv2.distanceTransform(255-dA, cv2.DIST_L2, 5).astype(np.float32)
    distB = cv2.distanceTransform(255-dB, cv2.DIST_L2, 5).astype(np.float32)
    mx = max(distA.max(), distB.max())
    if mx > 0: distA/=mx; distB/=mx
    distA **= 2; distB **= 2
    G = distB / (distA + distB + 1e-6)
    fg = np.zeros_like(G)
    mA = gmask(imA).astype(bool)
    fg[mA] = G[mA]
    return np.nan_to_num(fg, nan=0.5).astype(np.float32)

def blend(imA, imB, G):
    G3 = G[:,:,np.newaxis]
    return np.clip(imA.astype(np.float32)*G3 +
                   imB.astype(np.float32)*(1-G3), 0, 255).astype(np.uint8)


# ── Compositor thread ──────────────────────────────────────────────────
def compositor_thread(G0, G1, G2, G3):
    """
    Composites 4 BEV frames + front/back raw panels into display_out.
    Runs as fast as possible; WebRTC track reads display_out independently.
    """
    colour_bal_timer = 0.0
    bev  = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    fps_count = 0
    fps_timer = time.time()
    fps_val   = 0.0

    while not stop_signal:
        # ── Grab latest warped BEV frames ─────────────────────────────
        with bev_lock["front"]:  front = bev_frames["front"]
        with bev_lock["back"]:   back  = bev_frames["back"]
        with bev_lock["left"]:   left  = bev_frames["left"]
        with bev_lock["right"]:  right = bev_frames["right"]
        with raw_lock["front"]:  front_raw = raw_frames.get("front")
        with raw_lock["back"]:   back_raw  = raw_frames.get("back")

        if any(x is None for x in (front, back, left, right)):
            time.sleep(0.005); continue

        front = front.copy(); back  = back.copy()
        left  = left.copy();  right = right.copy()

        # ── Colour balance ─────────────────────────────────────────────
        now = time.time()
        if enable_color_balance:
            if now - colour_bal_timer > 2.0:
                compute_gains({"front":front,"back":back,
                               "left":left,"right":right})
                colour_bal_timer = now
            front = apply_gain(front, cam_gains["front"])
            back  = apply_gain(back,  cam_gains["back"])
            left  = apply_gain(left,  cam_gains["left"])
            right = apply_gain(right, cam_gains["right"])

        # ── BEV composite ──────────────────────────────────────────────
        if enable_blend:
            bev[:yt, :xl]  = blend(front[:yt,:xl], left[:yt,:xl],  G0)
            bev[:yt, xr:]  = blend(front[:yt,xr:], right[:yt,xr:], G1)
            bev[yb:, :xl]  = blend(back[yb:,:xl],  left[yb:,:xl],  G2)
            bev[yb:, xr:]  = blend(back[yb:,xr:],  right[yb:,xr:], G3)
        else:
            bev[:yt, :xl]  = front[:yt,:xl]
            bev[:yt, xr:]  = front[:yt,xr:]
            bev[yb:, :xl]  = back[yb:,:xl]
            bev[yb:, xr:]  = back[yb:,xr:]

        bev[:yt,  xl:xr] = front[:yt, xl:xr]
        bev[yb:,  xl:xr] = back[yb:,  xl:xr]
        bev[yt:yb, :xl]  = left[yt:yb, :xl]
        bev[yt:yb, xr:]  = right[yt:yb, xr:]

        # Update snapshot-only bev_out
        with bev_out_lock:
            np.copyto(bev_out, bev)

        # ── FPS ────────────────────────────────────────────────────────
        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_val   = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

        # ── Compose full display frame (BEV + raw panels) ──────────────
        disp = compose_display(bev, front_raw, back_raw,
                               fps_val, enable_blend, enable_color_balance)

        with display_out_lock:
            global display_out
            display_out = disp


# ── WebRTC track ───────────────────────────────────────────────────────
class DisplayTrack(VideoStreamTrack):
    """Streams the full composed display frame (BEV + camera panels)."""
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

        with display_out_lock:
            frame = display_out

        if frame is None:
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Guarantee even dims for H.264
            h, w = rgb.shape[:2]
            rgb  = rgb[:h-(h%2), :w-(w%2)]

        av_frame           = VideoFrame.from_ndarray(rgb, format="rgb24")
        av_frame.pts       = self._pts
        av_frame.time_base = fractions.Fraction(1, STREAM_FPS)
        self._pts         += 1
        return av_frame


# ── WebRTC server ──────────────────────────────────────────────────────
pcs = set()

def _strip_mdns(sdp):
    return "\r\n".join(l for l in sdp.splitlines()
                       if not (l.startswith("a=candidate:") and ".local" in l))

_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BEV Surround View</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d0d0d;color:#ccc;font-family:monospace;
        display:flex;flex-direction:column;align-items:center;padding:16px;gap:8px}}
  header{{width:100%;max-width:1920px;display:flex;justify-content:space-between;
           align-items:center;border-bottom:1px solid #00b4a0;padding-bottom:8px}}
  h1{{font-size:.9rem;color:#00dcc0;letter-spacing:.15em}}
  #st{{font-size:.75rem;padding:4px 10px;border:1px solid #555;
       border-radius:3px;color:#888;white-space:nowrap}}
  #st.ok{{border-color:#3cdc64;color:#3cdc64}}
  #st.err{{border-color:#f44;color:#f44}}
  #wrap{{width:100%;max-width:1920px;background:#000;border:2px solid #00b4a0;
          line-height:0}}
  video{{width:100%;display:block;background:#000}}
  footer{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
  button{{background:#1a1a1a;color:#00dcc0;border:1px solid #00b4a0;
           padding:5px 16px;cursor:pointer;font-family:inherit;font-size:.75rem;
           letter-spacing:.05em}}
  button:hover{{background:#003830}}
  button:disabled{{opacity:.35;cursor:default}}
  .pill{{font-size:.7rem;padding:3px 10px;border:1px solid #555;
          border-radius:3px;color:#555;cursor:pointer;user-select:none;
          transition:all .15s}}
  .pill.on{{border-color:#3cdc64;color:#3cdc64}}
  #log{{width:100%;max-width:1920px;height:64px;overflow-y:auto;
        border:1px solid #1e1e1e;padding:4px 8px;
        font-size:.62rem;color:#444;background:#0a0a0a}}
</style>
</head>
<body>
<header>
  <h1>⬡ BEV SURROUND VIEW — ZED Box {DEVICE_IP}</h1>
  <span id="st">DISCONNECTED</span>
</header>

<div id="wrap">
  <video id="v" autoplay playsinline muted></video>
</div>

<footer>
  <button id="bc" onclick="connect()">CONNECT</button>
  <button id="bd" onclick="disconnect()" disabled>DISCONNECT</button>
  <button onclick="document.getElementById('wrap').requestFullscreen()">FULLSCREEN</button>
  <button onclick="window.open('/snapshot')">SNAPSHOT</button>
  <span id="pBlend" class="pill on" onclick="toggle('blend')">BLEND: ON</span>
  <span id="pColor" class="pill"    onclick="toggle('color')">COLORBAL: OFF</span>
</footer>
<div id="log"></div>

<script>
let pc = null;

const log = m => {{
  const d = document.getElementById('log');
  d.innerHTML += `<div>[${{new Date().toTimeString().slice(0,8)}}] ${{m}}</div>`;
  d.scrollTop = d.scrollHeight;
}};
const st = (t, c) => {{
  const e = document.getElementById('st');
  e.textContent = t; e.className = c || '';
}};

async function toggle(what) {{
  try {{
    const r = await fetch('/toggle/' + what, {{method: 'POST'}});
    const j = await r.json();
    const id  = what === 'blend' ? 'pBlend' : 'pColor';
    const lbl = what === 'blend' ? 'BLEND'  : 'COLORBAL';
    const el  = document.getElementById(id);
    el.textContent = lbl + ': ' + (j.value ? 'ON' : 'OFF');
    el.className   = 'pill' + (j.value ? ' on' : '');
    log(lbl + ' → ' + (j.value ? 'ON' : 'OFF'));
  }} catch(e) {{ log('Toggle error: ' + e); }}
}}

async function connect() {{
  document.getElementById('bc').disabled = true;
  document.getElementById('bd').disabled = false;
  st('CONNECTING…');
  log('Creating RTCPeerConnection…');

  pc = new RTCPeerConnection({{
    iceServers: [{{urls: 'stun:stun.l.google.com:19302'}}]
  }});

  pc.onicecandidate = e =>
    log('ICE: ' + (e.candidate ? e.candidate.type + ' ' + e.candidate.address : 'gathering done'));

  pc.onconnectionstatechange = () => {{
    log('Connection: ' + pc.connectionState);
    if (pc.connectionState === 'connected')  st('CONNECTED', 'ok');
    else if (pc.connectionState === 'failed') {{ st('FAILED', 'err'); disconnect(); }}
    else if (pc.connectionState === 'closed') st('DISCONNECTED');
    else st(pc.connectionState.toUpperCase());
  }};

  pc.ontrack = e => {{
    log('Track received ✓');
    document.getElementById('v').srcObject = e.streams[0];
  }};

  pc.addTransceiver('video', {{direction: 'recvonly'}});

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  log('Waiting for ICE candidates…');

  await new Promise(resolve => {{
    if (pc.iceGatheringState === 'complete') {{ resolve(); return; }}
    const t = setTimeout(resolve, 3000);
    pc.onicegatheringstatechange = () => {{
      if (pc.iceGatheringState === 'complete') {{ clearTimeout(t); resolve(); }}
    }};
  }});

  log('Sending offer (' + pc.localDescription.sdp.split('\\n').filter(l=>l.startsWith('a=candidate')).length + ' candidates)…');
  try {{
    const r = await fetch('/offer', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{sdp: pc.localDescription.sdp, type: pc.localDescription.type}})
    }});
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const ans = await r.json();
    await pc.setRemoteDescription(ans);
    log('Remote description set — waiting for media…');
  }} catch(e) {{
    log('ERROR: ' + e); st('ERROR', 'err'); disconnect();
  }}
}}

function disconnect() {{
  if (pc) {{ pc.close(); pc = null; }}
  document.getElementById('v').srcObject = null;
  document.getElementById('bc').disabled = false;
  document.getElementById('bd').disabled = true;
  st('DISCONNECTED'); log('Disconnected');
}}

// Sync pill state with server on load
window.onload = async () => {{
  try {{
    const j = await (await fetch('/state')).json();
    const pb = document.getElementById('pBlend');
    pb.textContent = 'BLEND: '    + (j.blend ? 'ON' : 'OFF');
    pb.className   = 'pill' + (j.blend ? ' on' : '');
    const pc2 = document.getElementById('pColor');
    pc2.textContent = 'COLORBAL: ' + (j.color ? 'ON' : 'OFF');
    pc2.className   = 'pill' + (j.color ? ' on' : '');
  }} catch {{}}
}};
</script>
</body>
</html>"""


async def handle_index(request):
    return web.Response(content_type="text/html", text=_HTML)

async def handle_offer(request):
    params = await request.json()
    clean  = _strip_mdns(params["sdp"])
    n_cand = clean.count("a=candidate")
    print(f"  [WebRTC] offer received — {n_cand} candidates after mDNS strip")
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"  [WebRTC] {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close(); pcs.discard(pc)

    pc.addTrack(DisplayTrack())
    await pc.setRemoteDescription(RTCSessionDescription(sdp=clean, type=params["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp,
                               "type": pc.localDescription.type})

async def handle_toggle(request):
    global enable_blend, enable_color_balance
    what = request.match_info["what"]
    if what == "blend":
        enable_blend = not enable_blend
        val = enable_blend
        print(f"  Blend: {'ON' if val else 'OFF'}")
    elif what == "color":
        enable_color_balance = not enable_color_balance
        val = enable_color_balance
        print(f"  Colour balance: {'ON' if val else 'OFF'}")
    else:
        return web.Response(status=400)
    return web.json_response({"value": val})

async def handle_state(request):
    return web.json_response({"blend": enable_blend, "color": enable_color_balance})

async def handle_snapshot(request):
    """Returns the full composed display frame as JPEG."""
    with display_out_lock:
        frame = display_out
    if frame is None:
        return web.Response(status=503, text="No frame yet")
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return web.Response(body=jpg.tobytes(), content_type="image/jpeg")

async def handle_bev_snapshot(request):
    """Returns just the BEV composite as JPEG."""
    with bev_out_lock:
        bgr = bev_out.copy()
    _, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return web.Response(body=jpg.tobytes(), content_type="image/jpeg")

async def webrtc_main():
    app = web.Application()
    app.router.add_get("/",               handle_index)
    app.router.add_post("/offer",         handle_offer)
    app.router.add_post("/toggle/{what}", handle_toggle)
    app.router.add_get("/state",          handle_state)
    app.router.add_get("/snapshot",       handle_snapshot)
    app.router.add_get("/bev",            handle_bev_snapshot)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", WEBRTC_PORT).start()
    print(f"\n  ┌──────────────────────────────────────────────────┐")
    print(f"  │  WebRTC stream ready                             │")
    print(f"  │  http://{DEVICE_IP}:{WEBRTC_PORT}                       │")
    print(f"  │  /snapshot  — full display JPEG                 │")
    print(f"  │  /bev       — BEV-only JPEG                     │")
    print(f"  └──────────────────────────────────────────────────┘\n")
    while not stop_signal:
        await asyncio.sleep(0.5)
    await runner.cleanup()

def run_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:    loop.run_until_complete(webrtc_main())
    finally: loop.close()


# ── Main ───────────────────────────────────────────────────────────────
def main():
    global stop_signal, display_out
    signal.signal(signal.SIGINT, signal_handler)

    print(f"BEV canvas {BEV_W}×{BEV_H}  SCALE={SCALE}  INPUT_SCALE={INPUT_SCALE}")
    print("Loading homographies & building remap tables…")

    remap_maps = {}
    for cfg in CAMERAS:
        side = cfg["side"]
        path = f"homography_{side}_wider.npy"
        if not os.path.isfile(path):
            print(f"  [ERROR] {path} not found"); sys.exit(1)
        H = np.load(path, allow_pickle=True).item()["H"]
        mx, my = build_remap(H, INPUT_SCALE, SCALE, SRC_W, SRC_H)
        bake_clip(mx, my, VALID[side])
        remap_maps[side] = (mx, my)
        print(f"  ✓ {path}")

    for cfg in CAMERAS:
        side = cfg["side"]
        bev_frames[side] = None
        bev_ts[side]     = 0
        bev_lock[side]   = threading.Lock()

    for side in ("front", "back"):
        raw_frames[side] = None
        raw_lock[side]   = threading.Lock()

    print("\nOpening cameras…")
    threads = []
    for i, cfg in enumerate(CAMERAS):
        cam_lock.append(threading.Lock())
        serial = cfg["serial"]; kind = cfg["type"]; side = cfg["side"]
        print(f"  {cfg['name']}  S/N {serial}")
        if kind == "stereo":
            init = sl.InitParameters()
            init.set_from_serial_number(serial)
            init.camera_resolution = sl.RESOLUTION.HD1200
            init.camera_fps        = 30
            init.depth_mode        = sl.DEPTH_MODE.NONE
            cam = sl.Camera()
        else:
            init = sl.InitParametersOne()
            init.set_from_serial_number(serial)
            init.camera_resolution = sl.RESOLUTION.HD1200
            init.camera_fps        = 30
            cam = sl.CameraOne()
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"    !! Failed: {repr(status)}")
            cam_handles.append(None); cam_frames.append(sl.Mat())
            threads.append(None); continue
        cam_handles.append(cam); cam_frames.append(sl.Mat())
        mx, my = remap_maps[side]
        fn = grab_stereo if kind == "stereo" else grab_mono
        t  = threading.Thread(target=fn, args=(i, side, mx, my), daemon=True)
        t.start(); threads.append(t)
        print(f"    ✓ streaming")

    opened = sum(1 for h in cam_handles if h is not None)
    if opened == 0:
        print("No cameras opened."); return

    print(f"\n{opened}/4 cameras streaming. Waiting for first frames…")
    timeout = time.time() + 15.0
    while time.time() < timeout:
        if all(bev_frames[s] is not None for s in ("front","back","left","right")):
            break
        time.sleep(0.05)
    else:
        missing = [s for s in ("front","back","left","right") if bev_frames[s] is None]
        print(f"[ERROR] Timed out: {missing}"); stop_signal = True; return

    print("Computing corner blend weights…")
    with bev_lock["front"]: f0 = bev_frames["front"].copy()
    with bev_lock["back"]:  b0 = bev_frames["back"].copy()
    with bev_lock["left"]:  l0 = bev_frames["left"].copy()
    with bev_lock["right"]: r0 = bev_frames["right"].copy()

    G0 = compute_weight_matrix(f0[:yt,:xl], l0[:yt,:xl])
    G1 = compute_weight_matrix(f0[:yt,xr:], r0[:yt,xr:])
    G2 = compute_weight_matrix(b0[yb:,:xl], l0[yb:,:xl])
    G3 = compute_weight_matrix(b0[yb:,xr:], r0[yb:,xr:])
    print("  ✓ Blend weights ready")

    # Start compositor thread
    ct = threading.Thread(target=compositor_thread,
                          args=(G0, G1, G2, G3), daemon=True)
    ct.start()

    # Wait for first composed frame before starting WebRTC
    print("  Waiting for first composed frame…")
    t0 = time.time()
    while display_out is None and time.time() - t0 < 5.0:
        time.sleep(0.05)
    if display_out is None:
        print("  [WARN] No composed frame yet — starting server anyway")
    else:
        h, w = display_out.shape[:2]
        print(f"  ✓ First frame ready: {w}×{h}")

    # Start WebRTC server thread
    threading.Thread(target=run_server, daemon=True).start()

    print("\nPress Ctrl+C to quit\n")

    # Main thread — heartbeat stats every 5 s
    while not stop_signal:
        time.sleep(5)
        with bev_out_lock:
            filled = int((bev_out.sum(axis=2) > 0).sum())
        total = BEV_W * BEV_H
        print(f"  BEV {100*filled//total}% filled  "
              f"peers={len(pcs)}  "
              f"blend={'ON' if enable_blend else 'OFF'}  "
              f"colorbal={'ON' if enable_color_balance else 'OFF'}")

if __name__ == "__main__":
    main()