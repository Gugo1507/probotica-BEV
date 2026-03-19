import pyzed.sl as sl
import cv2
import numpy as np
import threading, time, signal, os, sys

stop_signal = False

# ── Scale factor ───────────────────────────────────────────────────────
SCALE = 0.35

# ── Canvas / layout ────────────────────────────────────────────────────
SQ      = 45;  COLS = 6;  ROWS = 4
BOARD_W = SQ * COLS
BOARD_H = SQ * ROWS
GAP_FB  = 270
GAP_LR  = 270
SHIFT   = 2000
VEH_W   = 490
VEH_L   = 740
BW_LR   = BOARD_H
BH_LR   = BOARD_W

BEV_W_FULL = SHIFT + BW_LR + GAP_LR + VEH_W + GAP_LR + BW_LR + SHIFT
BEV_H_FULL = SHIFT + BOARD_H + GAP_FB + VEH_L + GAP_FB + BOARD_H + SHIFT

BEV_W = int(BEV_W_FULL * SCALE)
BEV_H = int(BEV_H_FULL * SCALE)

VEH_X_FULL = SHIFT + BW_LR + GAP_LR
VEH_Y_FULL = SHIFT + BOARD_H + GAP_FB

VEH_X   = int(VEH_X_FULL * SCALE)
VEH_Y   = int(VEH_Y_FULL * SCALE)
VEH_W_S = int(VEH_W * SCALE)
VEH_L_S = int(VEH_L * SCALE)

xl = VEH_X
xr = VEH_X + VEH_W_S
yt = VEH_Y
yb = VEH_Y + VEH_L_S

# ── Camera registry ────────────────────────────────────────────────────
CAMERAS = [
    {"side": "front", "serial": 50542844,  "type": "stereo", "name": "ZED X Mini"},
    {"side": "back",  "serial": 40685614,  "type": "stereo", "name": "ZED X"},
    {"side": "right", "serial": 308745927, "type": "mono",   "name": "ZED XOne GS #1"},
    {"side": "left",  "serial": 304788437, "type": "mono",   "name": "ZED XOne GS #2"},
]

VALID = {
    "front": (0,   0,    BEV_W, yt),
    "back":  (0,   yb,   BEV_W, BEV_H),
    "left":  (0,   0,    xl,    BEV_H),
    "right": (xr,  0,    BEV_W, BEV_H),
}

# ── Runtime toggles ────────────────────────────────────────────────────
enable_blend         = True   # seamless corner blending
enable_color_balance = False  # colour balance (gains per camera)

# ── Scale homography ───────────────────────────────────────────────────
def scale_homography(H, input_scale, output_scale):
    S_in  = np.diag([input_scale,  input_scale,  1.0])
    S_out = np.diag([output_scale, output_scale, 1.0])
    return S_out @ H @ np.linalg.inv(S_in)

# ── Shared state ───────────────────────────────────────────────────────
cam_handles  = []
cam_frames   = []
cam_ts       = []
cam_lock     = []

bev_frames   = {}   # side → warped BEV  (scaled)
raw_frames   = {}   # side → latest raw BGR  (full-res)
bev_ts       = {}
bev_lock     = {}
raw_lock     = {}

# Per-camera colour-balance gain (BGR, float32)
cam_gains = {s: np.ones(3, dtype=np.float32) for s in ("front","back","left","right")}

def signal_handler(sig, frame):
    global stop_signal
    stop_signal = True
    time.sleep(0.5)
    exit()

# ── Colour balance ─────────────────────────────────────────────────────
def compute_gains(frames):
    """
    Grey-world colour balance across all 4 cameras.
    Computes mean only over non-black (visible) pixels in each BEV frame,
    then sets per-camera gains so every camera matches the global mean.
    """
    means = {}
    for side, f in frames.items():
        if f is None:
            continue
        # Mask out black/empty pixels (sum of channels < 15)
        mask = f.sum(axis=2) > 15
        if mask.sum() < 100:
            continue
        pixels = f[mask].astype(np.float32)   # shape (N, 3)
        means[side] = pixels.mean(axis=0)      # (B, G, R)

    if len(means) < 4:
        return  # not enough valid frames yet

    global_mean = np.mean(list(means.values()), axis=0)

    for side, m in means.items():
        gains = np.where(m > 1.0, global_mean / m, 1.0).astype(np.float32)
        cam_gains[side] = np.clip(gains, 0.5, 2.0)

def apply_gain(bgr, gain):
    out = bgr.astype(np.float32)
    out[..., 0] *= gain[0]
    out[..., 1] *= gain[1]
    out[..., 2] *= gain[2]
    return np.clip(out, 0, 255).astype(np.uint8)

# ── Process and store ──────────────────────────────────────────────────
def _process_and_store(raw_bgra, H_scaled, clip_mask, side, ts, input_scale):
    bgr = cv2.cvtColor(raw_bgra, cv2.COLOR_BGRA2BGR)

    # Store full-res raw frame for camera panel
    with raw_lock[side]:
        raw_frames[side] = bgr

    if input_scale != 1.0:
        bgr = cv2.resize(bgr,
                         (int(bgr.shape[1]*input_scale), int(bgr.shape[0]*input_scale)),
                         interpolation=cv2.INTER_LINEAR)

    warped = cv2.warpPerspective(bgr, H_scaled, (BEV_W, BEV_H),
                                 flags=cv2.INTER_LINEAR)
    warped = cv2.bitwise_and(warped, warped, mask=clip_mask)

    with bev_lock[side]:
        bev_frames[side] = warped
        bev_ts[side]     = ts

def grab_stereo(idx, side, H_scaled, clip_mask, input_scale):
    runtime = sl.RuntimeParameters()
    cam     = cam_handles[idx]
    last_ts = 0
    while not stop_signal:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            ts = cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
            if ts <= last_ts:
                time.sleep(0.001); continue
            last_ts = ts
            with cam_lock[idx]:
                cam.retrieve_image(cam_frames[idx], sl.VIEW.LEFT)
                raw = cam_frames[idx].get_data().copy()
            _process_and_store(raw, H_scaled, clip_mask, side, ts, input_scale)
        else:
            time.sleep(0.001)
    cam.close()

def grab_mono(idx, side, H_scaled, clip_mask, input_scale):
    cam     = cam_handles[idx]
    last_ts = 0
    use_arg = None
    try:
        rp_one     = sl.RuntimeParametersOne()
        has_rp_one = True
    except AttributeError:
        has_rp_one = False

    while not stop_signal:
        if use_arg is None:
            try:
                err = cam.grab(rp_one) if has_rp_one else cam.grab(sl.RuntimeParameters())
                use_arg = True
            except TypeError:
                err = cam.grab(); use_arg = False
        elif use_arg:
            err = cam.grab(rp_one) if has_rp_one else cam.grab(sl.RuntimeParameters())
        else:
            err = cam.grab()

        if err == sl.ERROR_CODE.SUCCESS:
            ts = cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
            if ts <= last_ts:
                time.sleep(0.001); continue
            last_ts = ts
            with cam_lock[idx]:
                cam.retrieve_image(cam_frames[idx], sl.VIEW.LEFT)
                raw = cam_frames[idx].get_data().copy()
            _process_and_store(raw, H_scaled, clip_mask, side, ts, input_scale)
        else:
            time.sleep(0.001)
    cam.close()

# ── Blend helpers ──────────────────────────────────────────────────────
def compute_weight_matrix(imA, imB):
    def get_mask(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        return m
    overlap   = cv2.bitwise_and(imA, imB)
    ovMask    = get_mask(overlap)
    ovMask    = cv2.dilate(ovMask, np.ones((2,2), np.uint8), iterations=2)
    ovMaskInv = cv2.bitwise_not(ovMask)
    dA = get_mask(cv2.bitwise_and(imA, imA, mask=ovMaskInv))
    dB = get_mask(cv2.bitwise_and(imB, imB, mask=ovMaskInv))
    distA = cv2.distanceTransform(255-dA, cv2.DIST_L2, 5).astype(np.float32)
    distB = cv2.distanceTransform(255-dB, cv2.DIST_L2, 5).astype(np.float32)
    mx = max(distA.max(), distB.max())
    if mx > 0: distA /= mx; distB /= mx
    distA **= 2; distB **= 2
    G = distB / (distA + distB + 1e-6)
    maskA = get_mask(imA).astype(bool)
    finalG = np.zeros_like(G)
    finalG[maskA] = G[maskA]
    return np.nan_to_num(finalG, nan=0.5).astype(np.float32)

def blend(imA, imB, G):
    G3  = G[:, :, np.newaxis]
    res = imA.astype(np.float32) * G3 + imB.astype(np.float32) * (1.0 - G3)
    return np.clip(res, 0, 255).astype(np.uint8)

def make_vehicle_patch():
    p  = np.full((VEH_L_S, VEH_W_S, 3), 30, dtype=np.uint8)
    cv2.rectangle(p, (0,0), (VEH_W_S-1, VEH_L_S-1), (80,80,80), 1)
    cx, cy = VEH_W_S//2, VEH_L_S//2
    cv2.putText(p, "VEH", (cx-18, cy+5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (120,120,120), 1, cv2.LINE_AA)
    cv2.arrowedLine(p, (cx, cy-8), (cx, cy-22), (100,100,100), 1, tipLength=0.4)
    return p

# ── HUD / UI drawing ───────────────────────────────────────────────────
# Design: dark industrial theme, monospaced labels, cyan accents

C_BG        = (18,  18,  18)   # near-black background
C_BORDER    = (0,   180, 160)  # teal border
C_BORDER_DIM= (0,   80,  70)   # dimmed border
C_LABEL_BG  = (0,   0,   0)
C_TEXT      = (220, 220, 220)
C_ACCENT    = (0,   220, 180)  # cyan-teal accent
C_ON        = (60,  220, 100)  # green for ON
C_OFF       = (80,  80,  80)   # grey for OFF
C_WARN      = (30,  150, 255)  # orange-ish for labels

BORDER      = 2    # px border thickness
LABEL_H     = 22   # height of label bar above each panel
GUTTER      = 4    # gap between panels

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO   = cv2.FONT_HERSHEY_PLAIN


def draw_panel_border(canvas, x, y, w, h, color, thickness=BORDER):
    cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), color, thickness)


def draw_label(canvas, x, y, w, text, color=C_ACCENT, bg=C_LABEL_BG):
    """Draw a label bar of height LABEL_H above the panel content."""
    cv2.rectangle(canvas, (x, y), (x+w-1, y+LABEL_H-1), bg, -1)
    cv2.rectangle(canvas, (x, y), (x+w-1, y+LABEL_H-1), color, 1)
    text_size = cv2.getTextSize(text, FONT_MONO, 1.0, 1)[0]
    tx = x + (w - text_size[0]) // 2
    ty = y + LABEL_H - 6
    cv2.putText(canvas, text, (tx, ty), FONT_MONO, 1.0, color, 1, cv2.LINE_AA)


def status_pill(canvas, x, y, label, state):
    """Draw a small ON/OFF indicator pill."""
    color = C_ON if state else C_OFF
    text  = f"{label}: {'ON' if state else 'OFF'}"
    sz    = cv2.getTextSize(text, FONT_MONO, 0.85, 1)[0]
    pw, ph = sz[0]+14, sz[1]+8
    cv2.rectangle(canvas, (x, y), (x+pw, y+ph), color, 1)
    cv2.putText(canvas, text, (x+7, y+ph-4), FONT_MONO, 0.85, color, 1, cv2.LINE_AA)
    return pw + 8  # advance


def compose_display(bev_composite, front_raw, back_raw, fps,
                    G0, G1, G2, G3,
                    front_bev, back_bev, left_bev, right_bev):
    """
    Build the full display frame.

    Layout (two columns):
    ┌─────────────────────────┬───────────────────────────┐
    │      BEV composite      │   FRONT camera (raw)      │
    │                         ├───────────────────────────┤
    │                         │   BACK  camera (raw)      │
    └─────────────────────────┴───────────────────────────┘
    HUD bar at top spanning full width.
    """
    # ── Target panel sizes ─────────────────────────────────────────────
    # BEV panel: label bar + BEV content
    BEV_CONTENT_H = BEV_H        # the actual BEV image height
    BEV_PH  = LABEL_H + BEV_CONTENT_H   # full panel height incl label
    BEV_PW  = BEV_W

    # Right column: two camera panels stacked, each = half of BEV_PH
    CAM_PH  = BEV_PH // 2
    CAM_PW  = int((CAM_PH - LABEL_H) * 1920 / 1200)  # keep 16:10 on content area

    HUD_H   = 38
    TOTAL_W = BORDER + BEV_PW + GUTTER + CAM_PW + BORDER
    TOTAL_H = HUD_H + BORDER + BEV_PH + BORDER

    canvas = np.full((TOTAL_H, TOTAL_W, 3), C_BG, dtype=np.uint8)

    # ── HUD bar ────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0,0), (TOTAL_W-1, HUD_H-1), (25,25,25), -1)
    cv2.line(canvas, (0, HUD_H-1), (TOTAL_W-1, HUD_H-1), C_BORDER, 1)

    # FPS
    fps_txt = f"FPS {fps:05.1f}"
    cv2.putText(canvas, fps_txt, (10, HUD_H-10), FONT_MONO, 1.1, C_ACCENT, 1, cv2.LINE_AA)

    # Toggle pills
    adv = status_pill(canvas, 120, 8, "BLEND",   enable_blend)
    status_pill(canvas, 120+adv, 8, "COLORBAL", enable_color_balance)

    # Keybinds hint (right-aligned)
    hint = "B=blend  C=colorbal  S=save  Q=quit"
    hsz  = cv2.getTextSize(hint, FONT_MONO, 0.85, 1)[0]
    cv2.putText(canvas, hint, (TOTAL_W - hsz[0] - 10, HUD_H-10),
                FONT_MONO, 0.85, (100,100,100), 1, cv2.LINE_AA)

    # ── BEV panel (left column) ────────────────────────────────────────
    bev_x = BORDER
    bev_y = HUD_H + BORDER

    # Label
    draw_label(canvas, bev_x, bev_y, BEV_PW, "BIRD'S-EYE VIEW  360°", C_ACCENT)
    bev_content_y = bev_y + LABEL_H

    if bev_composite is not None:
        # Resize BEV to exactly fill the content area (handles rounding edge cases)
        bev_resized = cv2.resize(bev_composite, (BEV_PW, BEV_CONTENT_H),
                                 interpolation=cv2.INTER_LINEAR)
        canvas[bev_content_y:bev_content_y+BEV_CONTENT_H,
               bev_x:bev_x+BEV_PW] = bev_resized

    draw_panel_border(canvas, bev_x, bev_y, BEV_PW, BEV_PH, C_BORDER)

    # ── Right column ───────────────────────────────────────────────────
    col2_x = BORDER + BEV_PW + GUTTER

    for cam_idx, (label, raw_frame) in enumerate([
            ("FRONT CAM — RAW", front_raw),
            ("BACK CAM  — RAW", back_raw)]):

        cam_y = HUD_H + BORDER + cam_idx * (CAM_PH + GUTTER)
        draw_label(canvas, col2_x, cam_y, CAM_PW, label, C_WARN)
        content_y = cam_y + LABEL_H
        content_h = CAM_PH - LABEL_H
        # Clamp to avoid overrun if rounding makes content_h off by 1
        avail_h = min(content_h, TOTAL_H - content_y)
        avail_w = min(CAM_PW, TOTAL_W - col2_x)

        if raw_frame is not None:
            resized = cv2.resize(raw_frame, (avail_w, avail_h),
                                 interpolation=cv2.INTER_LINEAR)
            canvas[content_y:content_y+avail_h,
                   col2_x:col2_x+avail_w] = resized
        else:
            # Placeholder
            cx = col2_x + CAM_PW//2
            cy = content_y + content_h//2
            cv2.putText(canvas, "NO SIGNAL", (cx-40, cy),
                        FONT_MONO, 1.0, (60,60,60), 1, cv2.LINE_AA)

        draw_panel_border(canvas, col2_x, cam_y, CAM_PW, CAM_PH,
                          C_BORDER if raw_frame is not None else C_BORDER_DIM)

    return canvas


# ── Main ───────────────────────────────────────────────────────────────
def main():
    global stop_signal, enable_blend, enable_color_balance

    signal.signal(signal.SIGINT, signal_handler)

    INPUT_SCALE = SCALE

    print(f"\nSCALE={SCALE}  BEV canvas {BEV_W}×{BEV_H} (full {BEV_W_FULL}×{BEV_H_FULL})")
    print("Loading & scaling homography matrices…")

    Hs_scaled = {}
    for cfg in CAMERAS:
        side = cfg["side"]
        path = f"homography_{side}_wider.npy"
        if not os.path.isfile(path):
            print(f"  [ERROR] {path} not found"); sys.exit(1)
        H_full = np.load(path, allow_pickle=True).item()["H"]
        Hs_scaled[side] = scale_homography(H_full, INPUT_SCALE, SCALE)
        print(f"  ✓ {path}")

    clip_masks = {}
    for side, (x0,y0,x1,y1) in VALID.items():
        m = np.zeros((BEV_H, BEV_W), dtype=np.uint8)
        m[y0:y1, x0:x1] = 255
        clip_masks[side] = m

    for cfg in CAMERAS:
        side = cfg["side"]
        bev_frames[side] = None
        raw_frames[side] = None
        bev_ts[side]     = 0
        bev_lock[side]   = threading.Lock()
        raw_lock[side]   = threading.Lock()

    print("\nOpening cameras…")
    threads = []
    for i, cfg in enumerate(CAMERAS):
        cam_lock.append(threading.Lock())
        cam_ts.append(0)
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
        args = (i, side, Hs_scaled[side], clip_masks[side], INPUT_SCALE)
        fn   = grab_stereo if kind == "stereo" else grab_mono
        t    = threading.Thread(target=fn, args=args, daemon=True)
        t.start(); threads.append(t)
        print(f"    ✓ streaming")

    opened = sum(1 for h in cam_handles if h is not None)
    if opened == 0:
        print("No cameras opened."); return

    print(f"\n{opened}/4 cameras streaming.")
    print("Waiting for first frames…")

    timeout = time.time() + 15.0
    while time.time() < timeout:
        if all(bev_frames[s] is not None for s in ("front","back","left","right")):
            break
        time.sleep(0.05)
    else:
        missing = [s for s in ("front","back","left","right") if bev_frames[s] is None]
        print(f"[ERROR] Timed out waiting for: {missing}")
        stop_signal = True; return

    # Compute colour balance gains from first frames
    first = {s: bev_frames[s] for s in ("front","back","left","right")}
    compute_gains(first)

    print("Computing corner blend weights…")
    with bev_lock["front"]: f0 = bev_frames["front"].copy()
    with bev_lock["back"]:  b0 = bev_frames["back"].copy()
    with bev_lock["left"]:  l0 = bev_frames["left"].copy()
    with bev_lock["right"]: r0 = bev_frames["right"].copy()

    G0 = compute_weight_matrix(f0[:yt, :xl],  l0[:yt, :xl])
    G1 = compute_weight_matrix(f0[:yt, xr:],  r0[:yt, xr:])
    G2 = compute_weight_matrix(b0[yb:, :xl],  l0[yb:, :xl])
    G3 = compute_weight_matrix(b0[yb:, xr:],  r0[yb:, xr:])
    print("  ✓ Ready")
    print("\nControls:")
    print("  B  — toggle seamless blending")
    print("  C  — toggle colour balance")
    print("  S  — save snapshot")
    print("  Q  — quit\n")

    #vehicle_patch = make_vehicle_patch()
    bev_out       = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

    WIN = "SURROUND VIEW SYSTEM"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # Estimate display window size after first compose
    fps_count = 0; fps_timer = time.time(); fps_val = 0.0
    colour_bal_timer = 0.0   # recompute gains every 2 s

    while True:
        # ── Snapshot all BEV frames ────────────────────────────────────
        with bev_lock["front"]:  front = bev_frames["front"]
        with bev_lock["back"]:   back  = bev_frames["back"]
        with bev_lock["left"]:   left  = bev_frames["left"]
        with bev_lock["right"]:  right = bev_frames["right"]
        with raw_lock["front"]:  front_raw = raw_frames["front"]
        with raw_lock["back"]:   back_raw  = raw_frames["back"]

        if any(x is None for x in (front, back, left, right)):
            cv2.waitKey(1); continue

        front = front.copy(); back  = back.copy()
        left  = left.copy();  right = right.copy()

        # ── Colour balance: recompute gains every 2 s, apply to all 4 simultaneously ─
        now = time.time()
        if enable_color_balance:
            if now - colour_bal_timer > 2.0:
                compute_gains({"front": front, "back": back,
                               "left": left,  "right": right})
                colour_bal_timer = now
            # Apply gains to all 4 frames in the main thread — always in sync
            front = apply_gain(front, cam_gains["front"])
            back  = apply_gain(back,  cam_gains["back"])
            left  = apply_gain(left,  cam_gains["left"])
            right = apply_gain(right, cam_gains["right"])

        # ── BEV composite ──────────────────────────────────────────────
        if enable_blend:
            bev_out[:yt,  :xl]  = blend(front[:yt, :xl], left[:yt,  :xl], G0)
            bev_out[:yt,  xr:]  = blend(front[:yt, xr:], right[:yt, xr:], G1)
            bev_out[yb:,  :xl]  = blend(back[yb:,  :xl], left[yb:,  :xl], G2)
            bev_out[yb:,  xr:]  = blend(back[yb:,  xr:], right[yb:, xr:], G3)
        else:
            # Hard cut — just paste each region directly
            bev_out[:yt,  :xl]  = front[:yt, :xl]
            bev_out[:yt,  xr:]  = front[:yt, xr:]
            bev_out[yb:,  :xl]  = back[yb:,  :xl]
            bev_out[yb:,  xr:]  = back[yb:,  xr:]

        bev_out[:yt,  xl:xr]  = front[:yt, xl:xr]
        bev_out[yb:,  xl:xr]  = back[yb:,  xl:xr]
        bev_out[yt:yb, :xl]   = left[yt:yb, :xl]
        bev_out[yt:yb, xr:]   = right[yt:yb, xr:]
        #bev_out[yt:yb, xl:xr] = vehicle_patch

        # ── FPS ────────────────────────────────────────────────────────
        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_val   = fps_count / elapsed
            fps_count = 0; fps_timer = time.time()

        # ── Compose full display ───────────────────────────────────────
        display = compose_display(
            bev_out, front_raw, back_raw, fps_val,
            G0, G1, G2, G3,
            front, back, left, right
        )

        # Auto-size window on first frame
        if fps_count == 1:
            dh, dw = display.shape[:2]
            cv2.resizeWindow(WIN, dw, dh)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('b'), ord('B')):
            enable_blend = not enable_blend
            print(f"  Blend: {'ON' if enable_blend else 'OFF'}")
        elif key in (ord('c'), ord('C')):
            enable_color_balance = not enable_color_balance
            print(f"  Colour balance: {'ON' if enable_color_balance else 'OFF'}")
        elif key in (ord('s'), ord('S')):
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            fname  = f"surround_{ts_str}.png"
            cv2.imwrite(fname, display.copy())
            print(f"  Saved → {fname}")

    stop_signal = True
    cv2.destroyAllWindows()
    print("FINISH")


if __name__ == "__main__":
    main()