"""
calibrate_bev.py
================
Bird's-Eye-View calibration for all 4 surround-view cameras.
Layout constants match generate_masks.py exactly.

Camera mapping:
  front  → ZED X Mini  (stereo, sl.Camera)    S/N 50542844
  back   → ZED X       (stereo, sl.Camera)    S/N 40685614
  right  → ZED XOne GS (mono,   sl.CameraOne) S/N 308745927
  left   → USB fisheye camera (cv2.VideoCapture, index 0)
             Undistorted via fisheye_calibration.npz before BEV warp.

Flow:
  0. Terminal asks which side to calibrate
  1. Live feed → press 'S' to snapshot
  2a. Press 'A' to AUTO-detect chessboard corners → corner-order check shown
  2b. OR click 4 outer corners manually (TL TR BR BL) → press 'N'
      Scroll wheel = zoom in/out on snapshot
      Right-drag = pan
      Right-click (no drag) = undo last point
  3. Press 'U' on mask image to use auto-computed corners → 'C' to compute
  4. Live BEV preview → 'S' to save  |  'Q' to quit
"""

import cv2
import numpy as np
import os, sys

# ══════════════════════════════════════════════════════════════════════
# USB fisheye camera settings
# ══════════════════════════════════════════════════════════════════════
USB_CAM_INDEX  = 0
USB_CAM_WIDTH  = 1920
USB_CAM_HEIGHT = 1080
USB_CAM_FPS    = 30
USB_CALIB_FILE = "fisheye_calibration.npz"

# ══════════════════════════════════════════════════════════════════════
# Layout constants — must match generate_masks.py exactly
# ══════════════════════════════════════════════════════════════════════
SQ        = 45
COLS      = 6
ROWS      = 4
BOARD_W   = SQ * COLS
BOARD_H   = SQ * ROWS
GAP_FB    = 650
GAP_LR    = 70
SHIFT     = 1500
VEHICLE_W = 500
VEHICLE_L = 150
BW_LR     = BOARD_H
BH_LR     = BOARD_W

CANVAS_W  = SHIFT + BW_LR + GAP_LR + VEHICLE_W + GAP_LR + BW_LR + SHIFT
CANVAS_H  = SHIFT + BOARD_H + GAP_FB + VEHICLE_L + GAP_FB + BOARD_H + SHIFT
VEH_X     = SHIFT + BW_LR + GAP_LR
VEH_Y     = SHIFT + BOARD_H + GAP_FB


def board_info(side):
    if side in ("front", "back"):
        cols, rows, bw, bh, gap = COLS, ROWS, BOARD_W, BOARD_H, GAP_FB
    else:
        cols, rows, bw, bh, gap = ROWS, COLS, BW_LR, BH_LR, GAP_LR
    if   side == "front": bx = VEH_X + (VEHICLE_W - bw) // 2;  by = VEH_Y - gap - bh
    elif side == "back":  bx = VEH_X + (VEHICLE_W - bw) // 2;  by = VEH_Y + VEHICLE_L + gap
    elif side == "left":  bx = VEH_X - gap - bw;                by = VEH_Y + (VEHICLE_L - bh) // 2
    else:                 bx = VEH_X + VEHICLE_W + gap;         by = VEH_Y + (VEHICLE_L - bh) // 2
    return cols, rows, bw, bh, gap, bx, by


def world_corners(side):
    _, _, bw, bh, _, bx, by = board_info(side)
    return np.float32([
        [bx,      by     ],
        [bx + bw, by     ],
        [bx + bw, by + bh],
        [bx,      by + bh],
    ])


def inner_pattern(side):
    if side in ("front", "back"):
        return (COLS - 1, ROWS - 1)   # (5, 3)
    else:
        return (ROWS - 1, COLS - 1)   # (3, 5)


# ══════════════════════════════════════════════════════════════════════
# Camera registry
# ══════════════════════════════════════════════════════════════════════
CAMERAS = {
    "front": {"serial": 50542844,  "type": "stereo",  "name": "ZED X Mini"},
    "back":  {"serial": 40685614,  "type": "stereo",  "name": "ZED X"},
    "right": {"serial": 308745927, "type": "mono",    "name": "ZED XOne GS #1"},
    "left":  {"serial": None,      "type": "usb",     "name": "USB Fisheye (cv2)"},
}

MASK_DIR = "calibration_masks"

CORNER_LABELS = ["TL", "TR", "BR", "BL"]
CORNER_COLORS = [(0, 0, 255), (0, 128, 255), (0, 220, 0), (255, 0, 0)]


# ══════════════════════════════════════════════════════════════════════
# USB fisheye helpers
# ══════════════════════════════════════════════════════════════════════
def load_fisheye_maps():
    if not os.path.isfile(USB_CALIB_FILE):
        print(f"[ERROR] Fisheye calibration not found: '{USB_CALIB_FILE}'")
        sys.exit(1)
    data = np.load(USB_CALIB_FILE)
    print(f"  Loaded fisheye maps from '{USB_CALIB_FILE}'")
    return data["map1"], data["map2"]


def open_usb_camera():
    cap = cv2.VideoCapture(USB_CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  USB_CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          USB_CAM_FPS)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open USB camera at index {USB_CAM_INDEX}")
        sys.exit(1)
    for _ in range(10):
        cap.read()
    map1, map2 = load_fisheye_maps()
    print(f"  Opened USB Fisheye  index={USB_CAM_INDEX}  {USB_CAM_WIDTH}x{USB_CAM_HEIGHT} @ {USB_CAM_FPS} fps")
    return cap, map1, map2


def get_usb_frame(cap, map1, map2):
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            return cv2.remap(frame, map1, map2,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
    return None


# ══════════════════════════════════════════════════════════════════════
# ZED helpers
# ══════════════════════════════════════════════════════════════════════
def open_zed_camera(side):
    import pyzed.sl as sl
    cfg = CAMERAS[side]; serial = cfg["serial"]; kind = cfg["type"]
    if kind == "stereo":
        init = sl.InitParameters()
        init.set_from_serial_number(serial)
        init.camera_resolution = sl.RESOLUTION.HD1200
        init.camera_fps = 30
        init.depth_mode = sl.DEPTH_MODE.NONE
        cam = sl.Camera()
        if cam.open(init) != sl.ERROR_CODE.SUCCESS:
            print(f"[ERROR] Cannot open {cfg['name']} S/N {serial}"); sys.exit(1)
        runtime = sl.RuntimeParameters()
        def grab(): return cam.grab(runtime)
    else:
        init = sl.InitParametersOne()
        init.set_from_serial_number(serial)
        init.camera_resolution = sl.RESOLUTION.HD1200
        init.camera_fps = 30
        cam = sl.CameraOne()
        if cam.open(init) != sl.ERROR_CODE.SUCCESS:
            print(f"[ERROR] Cannot open {cfg['name']} S/N {serial}"); sys.exit(1)
        try:
            rp = sl.RuntimeParametersOne(); cam.grab(rp)
            def grab(): return cam.grab(sl.RuntimeParametersOne())
        except (AttributeError, TypeError):
            def grab(): return cam.grab()
    print(f"  Opened {cfg['name']}  S/N {serial}")
    return cam, grab


def get_zed_frame(cam, grab):
    import pyzed.sl as sl
    mat = sl.Mat()
    for _ in range(10):
        if grab() == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            return cv2.cvtColor(mat.get_data().copy(), cv2.COLOR_BGRA2BGR)
    return None


def open_camera(side):
    if side == "left":
        cap, map1, map2 = open_usb_camera()
        return {"type": "usb", "cap": cap, "map1": map1, "map2": map2}
    else:
        cam, grab = open_zed_camera(side)
        return {"type": "zed", "cam": cam, "grab": grab}


def get_frame(handle):
    if handle["type"] == "usb":
        return get_usb_frame(handle["cap"], handle["map1"], handle["map2"])
    return get_zed_frame(handle["cam"], handle["grab"])


def close_camera(handle):
    if handle["type"] == "usb":
        handle["cap"].release()
    else:
        handle["cam"].close()


# ══════════════════════════════════════════════════════════════════════
# Corner order validation
# ══════════════════════════════════════════════════════════════════════
def validate_corner_order(pts, img_w, img_h):
    """
    Checks that pts[0..3] are in TL, TR, BR, BL order.
    Returns (ok: bool, issues: list[str]).
    """
    TL, TR, BR, BL = pts
    issues = []

    if TL[0] >= TR[0]:
        issues.append(f"TL.x ({TL[0]:.0f}) should be < TR.x ({TR[0]:.0f})")
    if BL[0] >= BR[0]:
        issues.append(f"BL.x ({BL[0]:.0f}) should be < BR.x ({BR[0]:.0f})")
    if TL[1] >= BL[1]:
        issues.append(f"TL.y ({TL[1]:.0f}) should be < BL.y ({BL[1]:.0f})")
    if TR[1] >= BR[1]:
        issues.append(f"TR.y ({TR[1]:.0f}) should be < BR.y ({BR[1]:.0f})")

    cx, cy = img_w / 2, img_h / 2
    if TL[0] > cx or TL[1] > cy:
        issues.append(f"TL ({TL[0]:.0f},{TL[1]:.0f}) not in top-left quadrant")
    if BR[0] < cx or BR[1] < cy:
        issues.append(f"BR ({BR[0]:.0f},{BR[1]:.0f}) not in bottom-right quadrant")

    # Convexity: all cross products same sign
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    quad = [TL, TR, BR, BL]
    crosses = [cross(quad[i], quad[(i+1)%4], quad[(i+2)%4]) for i in range(4)]
    if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
        issues.append("Quad is not convex — corners may be in wrong order")

    return len(issues) == 0, issues


def draw_corner_order_vis(img, pts, ok, issues):
    """
    Annotates img with numbered corners, sequence arrows, and PASS/FAIL banner.
    """
    vis = img.copy()

    # Sequence arrows TL->TR->BR->BL->TL
    for i in range(4):
        p1 = tuple(pts[i].astype(int))
        p2 = tuple(pts[(i+1) % 4].astype(int))
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        cv2.arrowedLine(vis, p1, mid, (255, 255, 0), 2, tipLength=0.15)
        cv2.line(vis, mid, p2, (255, 255, 0), 2)

    # Numbered corner circles
    for i, (pt, lbl, col) in enumerate(zip(pts, CORNER_LABELS, CORNER_COLORS)):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x, y), 18, (0, 0, 0), -1)
        cv2.circle(vis, (x, y), 16, col, -1)
        cv2.circle(vis, (x, y), 18, (255, 255, 255), 1)
        cv2.putText(vis, str(i+1), (x-6, y+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, lbl, (x+22, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(vis, lbl, (x+22, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

    # PASS / FAIL banner at bottom
    h, w = vis.shape[:2]
    n_lines = max(1, len(issues))
    banner_h = 36 + 28 * (n_lines - 1) if not ok else 36
    cv2.rectangle(vis, (0, h - banner_h), (w, h), (20, 20, 20), -1)
    if ok:
        cv2.putText(vis, "  CORNER ORDER OK  --  press N to accept",
                    (4, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 60), 2, cv2.LINE_AA)
    else:
        cv2.putText(vis, "  CORNER ORDER WRONG  --  press R to reset or fix manually",
                    (4, h - banner_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 255), 2, cv2.LINE_AA)
        for j, issue in enumerate(issues):
            cv2.putText(vis, f"    * {issue}",
                        (4, h - banner_h + 24 + 28*(j+1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1, cv2.LINE_AA)
    return vis


# ══════════════════════════════════════════════════════════════════════
# Zoomable / pannable image viewer
# ══════════════════════════════════════════════════════════════════════
class ZoomView:
    """
    Static-image viewer with scroll-wheel zoom and right-drag pan.

    on_click(img_x, img_y)  -- called on left-button click (image coords)
    on_right_click()        -- called on right-button click with <5px motion (undo)
    Right-button drag       -- pan
    Scroll wheel            -- zoom centred on cursor
    """
    ZOOM_STEP = 1.15
    ZOOM_MIN  = 0.05
    ZOOM_MAX  = 30.0

    def __init__(self, win_name, img, win_w=960, win_h=600):
        self.win  = win_name
        self.img  = img.copy()
        self.ih, self.iw = img.shape[:2]
        self.scale = min(win_w / self.iw, win_h / self.ih)
        self.ox = (win_w - self.iw * self.scale) / 2
        self.oy = (win_h - self.ih * self.scale) / 2
        self.win_w = win_w
        self.win_h = win_h
        self.overlay = None

        self.on_click       = None
        self.on_right_click = None

        self._rb_active   = False
        self._rb_start    = (0, 0)
        self._pan_start_m = (0, 0)
        self._pan_start_o = (0.0, 0.0)

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, win_w, win_h)
        cv2.setMouseCallback(self.win, self._mouse)

    def win_to_img(self, wx, wy):
        return (wx - self.ox) / self.scale, (wy - self.oy) / self.scale

    def render(self):
        src = self.overlay if self.overlay is not None else self.img
        canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)

        x0 = max(0, int((0           - self.ox) / self.scale))
        y0 = max(0, int((0           - self.oy) / self.scale))
        x1 = min(self.iw, int((self.win_w - self.ox) / self.scale) + 1)
        y1 = min(self.ih, int((self.win_h - self.oy) / self.scale) + 1)

        if x1 <= x0 or y1 <= y0:
            cv2.imshow(self.win, canvas); return canvas

        crop = src[y0:y1, x0:x1]
        dx0 = int(x0 * self.scale + self.ox)
        dy0 = int(y0 * self.scale + self.oy)
        dw  = max(1, int((x1 - x0) * self.scale))
        dh  = max(1, int((y1 - y0) * self.scale))
        dx0c = max(0, dx0); dy0c = max(0, dy0)
        dx1c = min(self.win_w, dx0 + dw)
        dy1c = min(self.win_h, dy0 + dh)
        if dx1c <= dx0c or dy1c <= dy0c:
            cv2.imshow(self.win, canvas); return canvas

        crop_w = dx1c - dx0c; crop_h = dy1c - dy0c
        interp = cv2.INTER_LINEAR if self.scale < 1 else cv2.INTER_NEAREST
        resized = cv2.resize(crop, (dw, dh), interpolation=interp)
        rx0 = dx0c - dx0; ry0 = dy0c - dy0
        canvas[dy0c:dy1c, dx0c:dx1c] = resized[ry0:ry0+crop_h, rx0:rx0+crop_w]

        cv2.putText(canvas, f"  zoom {self.scale:.2f}x  |  scroll=zoom  right-drag=pan  right-click=undo",
                    (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

        cv2.imshow(self.win, canvas)
        return canvas

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            factor = self.ZOOM_STEP if flags > 0 else 1.0 / self.ZOOM_STEP
            new_scale = float(np.clip(self.scale * factor, self.ZOOM_MIN, self.ZOOM_MAX))
            ix, iy = self.win_to_img(x, y)
            self.scale = new_scale
            self.ox = x - ix * self.scale
            self.oy = y - iy * self.scale
            self.render()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self._rb_active   = True
            self._rb_start    = (x, y)
            self._pan_start_m = (x, y)
            self._pan_start_o = (self.ox, self.oy)

        elif event == cv2.EVENT_RBUTTONUP:
            if self._rb_active:
                dist = ((x - self._rb_start[0])**2 + (y - self._rb_start[1])**2) ** 0.5
                if dist < 5 and self.on_right_click:
                    self.on_right_click()
                self._rb_active = False
            self.render()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._rb_active:
                dx = x - self._pan_start_m[0]
                dy = y - self._pan_start_m[1]
                self.ox = self._pan_start_o[0] + dx
                self.oy = self._pan_start_o[1] + dy
                self.render()

        elif event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = self.win_to_img(x, y)
            if 0 <= ix < self.iw and 0 <= iy < self.ih:
                if self.on_click:
                    self.on_click(ix, iy)


# ══════════════════════════════════════════════════════════════════════
# Auto chessboard detection
# ══════════════════════════════════════════════════════════════════════
def auto_detect_corners(img, side):
    """
    Detect inner chessboard corners, extrapolate 4 outer corners.
    Returns (cam_pts, vis_img, order_ok, order_issues).
    """
    pattern = inner_pattern(side)
    cols_i, rows_i = pattern
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)

    vis = img.copy()
    cv2.drawChessboardCorners(vis, pattern, corners, found)

    if not found:
        print(f"  [AUTO] Board not found (pattern {cols_i}x{rows_i}).")
        return None, vis, False, ["Board not detected"]

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    corners  = corners.reshape(-1, 2)

    grid_pts = np.array(
        [[c, r] for r in range(rows_i) for c in range(cols_i)],
        dtype=np.float32
    )
    H_grid, _ = cv2.findHomography(grid_pts, corners, cv2.RANSAC, 3.0)

    def grid_to_px(gx, gy):
        p = H_grid @ np.array([gx, gy, 1.0])
        return p[:2] / p[2]

    TL = grid_to_px(-1,     -1)
    TR = grid_to_px(cols_i, -1)
    BR = grid_to_px(cols_i, rows_i)
    BL = grid_to_px(-1,     rows_i)
    cam_pts = np.float32([TL, TR, BR, BL])

    if side in ("back", "right", "left"):
        cam_pts = np.float32([BR, BL, TL, TR])
        print(f"  [AUTO] Applied 180 deg corner flip for {side} camera")

    h, w = img.shape[:2]
    order_ok, order_issues = validate_corner_order(cam_pts, w, h)
    vis = draw_corner_order_vis(vis, cam_pts, order_ok, order_issues)

    print(f"  [AUTO] Board detected  pattern={pattern}")
    for lbl, pt in zip(CORNER_LABELS, cam_pts):
        print(f"         {lbl}: ({pt[0]:.1f}, {pt[1]:.1f})")
    if order_ok:
        print("  [AUTO] Corner order: PASS")
    else:
        print("  [AUTO] Corner order: FAIL")
        for issue in order_issues:
            print(f"         * {issue}")

    return cam_pts, vis, order_ok, order_issues


# ══════════════════════════════════════════════════════════════════════
# Side selection
# ══════════════════════════════════════════════════════════════════════
def ask_side():
    print("\n" + "="*58)
    print("  Surround-View BEV Calibration")
    print("="*58)
    print("\n  Which camera do you want to calibrate?\n")
    labels = {
        "front": f"ZED X Mini  (S/N {CAMERAS['front']['serial']})",
        "back":  f"ZED X       (S/N {CAMERAS['back']['serial']})",
        "right": f"ZED XOne GS (S/N {CAMERAS['right']['serial']})",
        "left":  f"USB Fisheye (cv2.VideoCapture index {USB_CAM_INDEX})",
    }
    for key, desc in labels.items():
        print(f"    [{key[0].upper()}]  {key:6s} -- {desc}")
    print("    [Q]  Quit\n")
    while True:
        choice = input("  Enter choice (F/B/L/R/Q): ").strip().lower()
        if choice in ('f', 'front'):  return 'front'
        if choice in ('b', 'back'):   return 'back'
        if choice in ('l', 'left'):   return 'left'
        if choice in ('r', 'right'):  return 'right'
        if choice in ('q', 'quit'):   sys.exit(0)
        print("  Please enter F, B, L, R, or Q.")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    side = ask_side()
    cfg  = CAMERAS[side]
    cols, rows, bw, bh, gap, bx, by = board_info(side)

    serial_str = str(cfg["serial"]) if cfg["serial"] else f"USB index {USB_CAM_INDEX}"
    print(f"\n{'='*58}")
    print(f"  Calibrating: {side.upper()}  --  {cfg['name']}  ({serial_str})")
    print(f"  Board: {cols}x{rows} squares ({bw}x{bh}mm)  gap={gap}mm")
    print(f"  Board TL in mask: ({bx}, {by})")
    print(f"  Auto-detect pattern (inner corners): {inner_pattern(side)}")
    print(f"{'='*58}")

    # ── Load mask ─────────────────────────────────────────────────────
    mask_path = os.path.join(MASK_DIR, f"mask_{side}.png")
    if not os.path.isfile(mask_path):
        cands = [f for f in os.listdir(MASK_DIR)
                 if f.startswith(f"mask_{side}") and f.endswith(".png")]
        if not cands:
            print(f"[ERROR] No mask for '{side}' in '{MASK_DIR}'"); sys.exit(1)
        mask_path = os.path.join(MASK_DIR, cands[0])
    mask_img = cv2.imread(mask_path)
    BEV_H_px, BEV_W_px = mask_img.shape[:2]
    print(f"  Mask: {mask_path}  ({BEV_W_px}x{BEV_H_px} px)")

    mask_corners = world_corners(side)
    print(f"  Mask corners (TL TR BR BL): {mask_corners.tolist()}\n")

    # ── Open camera ───────────────────────────────────────────────────
    handle = open_camera(side)
    if handle["type"] == "zed":
        print("  Warming up (30 frames)...")
        for _ in range(30):
            handle["grab"]()

    # ═════════════════════════════════════════════════════════════════
    # STEP 1 — Live snapshot
    # ═════════════════════════════════════════════════════════════════
    print("\nSTEP 1 -- Live feed.")
    print("  S = snapshot   Q = quit")

    preview_win = "Live Feed -- S=snapshot  Q=quit"
    cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview_win, 960, 600)

    cam_frame = None
    while True:
        f = get_frame(handle)
        if f is None:
            continue
        disp = f.copy()
        label = f"  [{side.upper()}] {cfg['name']}  |  S=snapshot  Q=quit"
        if side == "left":
            label += "  (undistorted)"
        cv2.rectangle(disp, (0,0), (disp.shape[1], 36), (20,20,20), -1)
        cv2.putText(disp, label, (4, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(preview_win, cv2.resize(disp, (0,0), fx=0.5, fy=0.5))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cam_frame = f.copy()
            print("  Snapshot captured.")
            break
        elif key == ord('q'):
            close_camera(handle); cv2.destroyAllWindows(); return

    cv2.destroyWindow(preview_win)

    # ═════════════════════════════════════════════════════════════════
    # STEP 2 — Pick corners on snapshot (zoomable)
    # ═════════════════════════════════════════════════════════════════
    print("\nSTEP 2 -- Camera corners on snapshot.")
    print("  A             = auto-detect chessboard (with order check)")
    print("  Left-click    = place corner (TL -> TR -> BR -> BL)")
    print("  N             = accept current 4 points")
    print("  R             = reset all points")
    print("  Scroll wheel  = zoom in/out centred on cursor")
    print("  Right-drag    = pan image")
    print("  Right-click   = undo last point")
    print("  Q             = quit")

    pts_img = []   # corners in full image coords

    def make_snap_overlay():
        out = cam_frame.copy()
        for i, (px, py) in enumerate(pts_img):
            x, y = int(px), int(py)
            cv2.circle(out, (x, y), 10, (0, 0, 0), -1)
            cv2.circle(out, (x, y), 8, CORNER_COLORS[i], -1)
            cv2.circle(out, (x, y), 10, (255, 255, 255), 1)
            cv2.putText(out, CORNER_LABELS[i], (x+14, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(out, CORNER_LABELS[i], (x+14, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CORNER_COLORS[i], 1, cv2.LINE_AA)
        if len(pts_img) == 4:
            poly = np.array([(int(p[0]), int(p[1])) for p in pts_img], np.int32)
            cv2.polylines(out, [poly], True, (0, 255, 255), 2)
            arr = np.float32(pts_img)
            h, w = cam_frame.shape[:2]
            ok, issues = validate_corner_order(arr, w, h)
            out = draw_corner_order_vis(out, arr, ok, issues)
        n    = len(pts_img)
        hint = CORNER_LABELS[n] if n < 4 else "4/4 ready -- N=accept"
        info = (f"  {n}/4  next: {hint}  |  "
                "A=auto  N=accept  R=reset  scroll=zoom  right-drag=pan  Q=quit")
        cv2.rectangle(out, (0, 0), (out.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(out, info, (4, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return out

    zv = ZoomView("Step 2 -- Snapshot  (scroll=zoom  right-drag=pan  right-click=undo)",
                  cam_frame, win_w=960, win_h=600)

    def on_click(ix, iy):
        if len(pts_img) < 4:
            pts_img.append((ix, iy))
            zv.overlay = make_snap_overlay()
            zv.render()

    def on_undo():
        if pts_img:
            pts_img.pop()
            zv.overlay = make_snap_overlay()
            zv.render()

    zv.on_click       = on_click
    zv.on_right_click = on_undo
    zv.overlay = make_snap_overlay()
    zv.render()

    cam_pts = None
    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('a'):
            detected, vis_auto, order_ok, order_issues = auto_detect_corners(cam_frame, side)
            if detected is not None:
                pts_img.clear()
                pts_img.extend([tuple(p) for p in detected])
                zv.overlay = vis_auto
                zv.render()
                if order_ok:
                    print("  Corner order OK -- press N to accept or R to redo.")
                else:
                    print("  Corner order WRONG -- press R to reset or fix manually.")
            else:
                zv.overlay = vis_auto
                zv.render()
                print("  Auto-detect failed -- try manual clicking.")

        elif key == ord('r'):
            pts_img.clear()
            cam_pts = None
            zv.overlay = make_snap_overlay()
            zv.render()

        elif key == ord('n'):
            if len(pts_img) == 4:
                arr = np.float32(pts_img)
                h, w = cam_frame.shape[:2]
                ok, issues = validate_corner_order(arr, w, h)
                if not ok:
                    print("  WARNING: accepting despite order issues:")
                    for iss in issues:
                        print(f"     * {iss}")
                cam_pts = arr
                break
            else:
                print(f"  Need 4 points ({len(pts_img)}/4).")

        elif key == ord('q'):
            close_camera(handle); cv2.destroyAllWindows(); return

    cv2.destroyWindow(zv.win)
    print(f"  Camera pts accepted: {cam_pts.tolist()}")

    # ═════════════════════════════════════════════════════════════════
    # STEP 3 — Pick corners on mask (zoomable)
    # ═════════════════════════════════════════════════════════════════
    print("\nSTEP 3 -- Mask corners.")
    print("  U          = use auto-computed mask corners (recommended)")
    print("  Left-click = place corners manually")
    print("  C          = compute homography")
    print("  R          = reset   |   scroll=zoom   right-drag=pan   right-click=undo")

    mask_pts_list = []

    def make_mask_overlay():
        out = mask_img.copy()
        for i, (px, py) in enumerate(mask_pts_list):
            x, y = int(px), int(py)
            cv2.circle(out, (x, y), 12, CORNER_COLORS[i], -1)
            cv2.putText(out, CORNER_LABELS[i], (x+14, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3)
            cv2.putText(out, CORNER_LABELS[i], (x+14, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CORNER_COLORS[i], 2)
        if len(mask_pts_list) == 4:
            poly = np.array([(int(p[0]),int(p[1])) for p in mask_pts_list], np.int32)
            cv2.polylines(out, [poly], True, (0,255,255), 2)
        n    = len(mask_pts_list)
        hint = CORNER_LABELS[n] if n < 4 else "4/4 ready -- C=compute"
        info = (f"  {n}/4  next: {hint}  |  "
                "U=auto  C=compute  R=reset  scroll=zoom  right-drag=pan  Q=quit")
        cv2.rectangle(out, (0, 0), (out.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(out, info, (4, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return out

    zv2 = ZoomView("Step 3 -- Mask  (scroll=zoom  right-drag=pan  right-click=undo)",
                   mask_img, win_w=960, win_h=600)

    def on_mask_click(ix, iy):
        if len(mask_pts_list) < 4:
            mask_pts_list.append((ix, iy))
            zv2.overlay = make_mask_overlay()
            zv2.render()

    def on_mask_undo():
        if mask_pts_list:
            mask_pts_list.pop()
            zv2.overlay = make_mask_overlay()
            zv2.render()

    zv2.on_click       = on_mask_click
    zv2.on_right_click = on_mask_undo
    zv2.overlay = make_mask_overlay()
    zv2.render()

    mask_pts = None
    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u'):
            mask_pts_list.clear()
            mask_pts_list.extend([tuple(p) for p in mask_corners])
            vis_m = mask_img.copy()
            for i, pt in enumerate(mask_corners):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(vis_m, (x,y), 16, CORNER_COLORS[i], -1)
                cv2.putText(vis_m, CORNER_LABELS[i], (x+18, y-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 3)
                cv2.putText(vis_m, CORNER_LABELS[i], (x+18, y-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, CORNER_COLORS[i], 2)
            poly = mask_corners.astype(np.int32)
            cv2.polylines(vis_m, [poly], True, (0,255,255), 2)
            cv2.rectangle(vis_m, (0,0), (vis_m.shape[1], 36), (20,20,20), -1)
            cv2.putText(vis_m, "  Auto mask corners -- press C to compute homography",
                        (4, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,180), 1, cv2.LINE_AA)
            zv2.overlay = vis_m
            zv2.render()
            print(f"  Auto mask corners: {mask_corners.tolist()}")
            print("  Press C to compute homography.")

        elif key == ord('r'):
            mask_pts_list.clear()
            mask_pts = None
            zv2.overlay = make_mask_overlay()
            zv2.render()

        elif key == ord('c'):
            if len(mask_pts_list) == 4:
                mask_pts = np.float32(mask_pts_list)
                break
            else:
                print(f"  Need 4 points ({len(mask_pts_list)}/4) -- or press U for auto.")

        elif key == ord('q'):
            close_camera(handle); cv2.destroyAllWindows(); return

    cv2.destroyWindow(zv2.win)
    print(f"  Mask pts accepted: {mask_pts.tolist()}")

    # ═════════════════════════════════════════════════════════════════
    # STEP 4 — Compute homography
    # ═════════════════════════════════════════════════════════════════
    H, _ = cv2.findHomography(cam_pts, mask_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("[ERROR] findHomography failed."); close_camera(handle); return

    print(f"\n  Homography H:\n{H}")
    print("\n  Reprojection errors (camera -> mask):")
    full_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    for i in range(4):
        ph = H @ [cam_pts[i,0], cam_pts[i,1], 1.0]; ph /= ph[2]
        err = np.linalg.norm(ph[:2] - mask_pts[i])
        print(f"    {full_labels[i]:14s}  {err:.2f} px")

    # ═════════════════════════════════════════════════════════════════
    # STEP 5 — Live BEV preview
    # ═════════════════════════════════════════════════════════════════
    print("\nSTEP 5 -- Live BEV preview.")
    print("  S = save homography   Q = quit")

    WIN_BEV  = f"BEV [{side.upper()}]  |  S=save  Q=quit"
    WIN_CAM  = "Camera feed"
    cv2.namedWindow(WIN_BEV, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_BEV, int(BEV_W_px * 0.2), int(BEV_H_px * 0.2))
    cv2.namedWindow(WIN_CAM, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CAM, 960, 600)

    while True:
        f = get_frame(handle)
        if f is None:
            continue
        bev = cv2.warpPerspective(f, H, (BEV_W_px, BEV_H_px))
        cv2.imshow(WIN_CAM, cv2.resize(f,   (0,0), fx=0.5, fy=0.5))
        cv2.imshow(WIN_BEV, cv2.resize(bev, (0,0), fx=0.2, fy=0.2))
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            fname_npy = f"homography_{side}.npy"
            fname_png = f"bev_{side}.png"
            serial_val = cfg["serial"] if cfg["serial"] else f"usb:{USB_CAM_INDEX}"
            np.save(fname_npy,
                    {"H":             H,
                     "board_side":    side,
                     "bev_size":      (BEV_W_px, BEV_H_px),
                     "camera_serial": serial_val,
                     "camera_name":   cfg["name"],
                     "camera_pts":    cam_pts.tolist(),
                     "mask_pts":      mask_pts.tolist()},
                    allow_pickle=True)
            cv2.imwrite(fname_png, bev)
            print(f"  Saved {fname_npy}  +  {fname_png}")
        elif key == ord('q'):
            break

    close_camera(handle)
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()