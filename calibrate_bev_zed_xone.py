"""
calibrate_bev.py
================
Bird's-Eye-View calibration for all 4 surround-view cameras.
Layout constants match generate_masks.py exactly.

Camera mapping:
  front  → ZED X Mini  (stereo, sl.Camera)    S/N 50542844
  back   → ZED X       (stereo, sl.Camera)    S/N 40685614
  right  → ZED XOne GS (mono,   sl.CameraOne) S/N 308745927
  left   → ZED XOne GS (mono,   sl.CameraOne) S/N 304788437

Flow:
  0. Terminal asks which side to calibrate
  1. Live feed → press 'S' to snapshot
  2a. Press 'A' to AUTO-detect chessboard corners (uses cv2.findChessboardCorners)
  2b. OR click 4 outer corners manually (TL TR BR BL) → press 'N'
  3. Click same 4 corners on MASK image → 'C' to compute homography
  4. Live BEV preview → 'S' to save  |  'Q' to quit

  Right-click = undo last point.  'R' = reset points.
"""

import cv2
import numpy as np
import os, sys

# ══════════════════════════════════════════════════════════════════════
# Layout constants — must match generate_masks.py exactly
# ══════════════════════════════════════════════════════════════════════
SQ        = 45
COLS      = 6
ROWS      = 4
BOARD_W   = SQ * COLS   # 270 mm  (front/back)
BOARD_H   = SQ * ROWS   # 180 mm
GAP_FB    = 270          # front/back vehicle-edge → board
GAP_LR    = 270          # left/right vehicle-edge → board
SHIFT     = 1000
VEHICLE_W = 490
VEHICLE_L = 740
BW_LR     = BOARD_H     # 180  (rotated board width  for left/right)
BH_LR     = BOARD_W     # 270  (rotated board height for left/right)

CANVAS_W  = SHIFT + BW_LR + GAP_LR + VEHICLE_W + GAP_LR + BW_LR + SHIFT  # 5390
CANVAS_H  = SHIFT + BOARD_H + GAP_FB + VEHICLE_L + GAP_FB + BOARD_H + SHIFT  # 5640
VEH_X     = SHIFT + BW_LR + GAP_LR   # 2450
VEH_Y     = SHIFT + BOARD_H + GAP_FB  # 2450

# ── Per-side derived geometry ─────────────────────────────────────────
def board_info(side):
    """Return (cols, rows, bw, bh, gap, bx, by) for a given side."""
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
    """4 outer corners of the chessboard in mask/world coordinates (TL TR BR BL)."""
    _, _, bw, bh, _, bx, by = board_info(side)
    return np.float32([
        [bx,      by     ],   # TL
        [bx + bw, by     ],   # TR
        [bx + bw, by + bh],   # BR
        [bx,      by + bh],   # BL
    ])


def inner_pattern(side):
    """
    Inner-corner count for cv2.findChessboardCorners.
    cv2 counts intersections, not squares:
      front/back: 6×4 squares  → 5×3 inner corners
      left/right: 4×6 squares  → 3×5 inner corners
    """
    if side in ("front", "back"):
        return (COLS - 1, ROWS - 1)   # (5, 3)
    else:
        return (ROWS - 1, COLS - 1)   # (3, 5)


# ══════════════════════════════════════════════════════════════════════
# Camera registry
# ══════════════════════════════════════════════════════════════════════
CAMERAS = {
    "front": {"serial": 50542844,  "type": "stereo", "name": "ZED X Mini"},
    "back":  {"serial": 40685614,  "type": "stereo", "name": "ZED X"},
    "right": {"serial": 308745927, "type": "mono",   "name": "ZED XOne GS #1"},
    "left":  {"serial": 304788437, "type": "mono",   "name": "ZED XOne GS #2"},
}

MASK_DIR = "calibration_masks"


# ══════════════════════════════════════════════════════════════════════
# Camera helpers
# ══════════════════════════════════════════════════════════════════════
def open_camera(side):
    import pyzed.sl as sl
    cfg    = CAMERAS[side]
    serial = cfg["serial"]
    kind   = cfg["type"]

    if kind == "stereo":
        init = sl.InitParameters()
        init.set_from_serial_number(serial)
        init.camera_resolution = sl.RESOLUTION.HD1200
        init.camera_fps        = 30
        init.depth_mode        = sl.DEPTH_MODE.NONE
        cam = sl.Camera()
        if cam.open(init) != sl.ERROR_CODE.SUCCESS:
            print(f"[ERROR] Cannot open {cfg['name']} S/N {serial}"); sys.exit(1)
        runtime = sl.RuntimeParameters()
        def grab(): return cam.grab(runtime)

    else:
        init = sl.InitParametersOne()
        init.set_from_serial_number(serial)
        init.camera_resolution = sl.RESOLUTION.HD1200
        init.camera_fps        = 30
        cam = sl.CameraOne()
        if cam.open(init) != sl.ERROR_CODE.SUCCESS:
            print(f"[ERROR] Cannot open {cfg['name']} S/N {serial}"); sys.exit(1)
        try:
            rp = sl.RuntimeParametersOne(); cam.grab(rp)
            def grab(): return cam.grab(sl.RuntimeParametersOne())
        except (AttributeError, TypeError):
            def grab(): return cam.grab()

    print(f"  Opened {cfg['name']}  S/N {serial}")
    return cam, grab, kind


def get_frame(cam, grab, kind):
    import pyzed.sl as sl
    mat = sl.Mat()
    for _ in range(10):
        if grab() == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            return cv2.cvtColor(mat.get_data().copy(), cv2.COLOR_BGRA2BGR)
    return None


# ══════════════════════════════════════════════════════════════════════
# Auto chessboard detection
# ══════════════════════════════════════════════════════════════════════
def auto_detect_corners(img, side):
    """
    Detect the chessboard inner corners and derive the 4 outer board corners.

    Returns (cam_pts, vis_img) where cam_pts is np.float32 shape (4,2)
    in TL TR BR BL order, or (None, vis_img) on failure.
    """
    pattern = inner_pattern(side)    # e.g. (5,3) for front/back
    cols_i, rows_i = pattern
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)

    found, corners = cv2.findChessboardCorners(gray, pattern, flags)

    vis = img.copy()
    cv2.drawChessboardCorners(vis, pattern, corners, found)

    if not found:
        print(f"  [AUTO] Board not found (pattern {cols_i}×{rows_i}). "
              "Try better lighting or use manual mode.")
        return None, vis

    # Sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    corners  = corners.reshape(-1, 2)   # (N, 2)

    # corners are ordered row-by-row from findChessboardCorners.
    # The board may be detected in any orientation, so we derive the
    # 4 outer corners from the extent of inner corners + one half-square offset.
    half = SQ / 2.0

    # Fit a homography from inner-corner grid indices to pixel positions
    # so we can robustly extrapolate the outer corners regardless of rotation.
    grid_pts = np.array(
        [[c, r] for r in range(rows_i) for c in range(cols_i)],
        dtype=np.float32
    )
    H_grid, _ = cv2.findHomography(grid_pts, corners, cv2.RANSAC, 3.0)

    def grid_to_px(gx, gy):
        p = H_grid @ np.array([gx, gy, 1.0])
        return p[:2] / p[2]

    # Outer corner grid coords (one half-square beyond the inner corners)
    # Inner corners span [0 .. cols_i-1] × [0 .. rows_i-1] in grid space.
    # One square = 1.0 in grid space (inner corners are 1 apart).
    TL = grid_to_px(-1,        -1)
    TR = grid_to_px(cols_i,    -1)
    BR = grid_to_px(cols_i,    rows_i)
    BL = grid_to_px(-1,        rows_i)

    cam_pts = np.float32([TL, TR, BR, BL])

    # back, right and left cameras are mounted so that findChessboardCorners
    # returns corners in reverse order — rotate 180° by reversing TL↔BR TR↔BL
    if side in ("back", "right", "left"):
        cam_pts = np.float32([BR, BL, TL, TR])  # 180° rotation = swap diagonals
        print(f"  [AUTO] Applied 180° corner flip for {side} camera")

    # Draw outer quad on vis
    poly = cam_pts.astype(np.int32)
    cv2.polylines(vis, [poly], True, (0, 255, 0), 3)
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0,0,255),(0,128,255),(0,220,0),(255,0,0)]
    for i, (pt, lbl, col) in enumerate(zip(cam_pts, labels, colors)):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x, y), 10, col, -1)
        cv2.putText(vis, lbl, (x+12, y-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(vis, lbl, (x+12, y-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, col, 2, cv2.LINE_AA)

    print(f"  [AUTO] Board detected  pattern={pattern}  outer corners:")
    for lbl, pt in zip(labels, cam_pts):
        print(f"         {lbl}: ({pt[0]:.1f}, {pt[1]:.1f})")

    return cam_pts, vis


# ══════════════════════════════════════════════════════════════════════
# GUI overlay
# ══════════════════════════════════════════════════════════════════════
LABELS = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
COLORS = [(0,0,255), (0,128,255), (0,220,0), (255,0,0)]

g_pts   = []
g_img   = None
g_scale = 1.0
g_win   = "Calibration"


def draw_overlay(extra_text=""):
    out = cv2.resize(g_img, (0,0), fx=g_scale, fy=g_scale)
    for i, (ox, oy) in enumerate(g_pts):
        x, y = int(ox * g_scale), int(oy * g_scale)
        cv2.circle(out, (x,y), 8, (0,0,0), -1)
        cv2.circle(out, (x,y), 7, COLORS[i], -1)
        cv2.circle(out, (x,y), 9, (255,255,255), 1)
        cv2.putText(out, LABELS[i], (x+10,y-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(out, LABELS[i], (x+10,y-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, COLORS[i], 1, cv2.LINE_AA)
    if len(g_pts) == 4:
        poly = np.array([(int(ox*g_scale), int(oy*g_scale)) for ox,oy in g_pts], np.int32)
        cv2.polylines(out, [poly], True, (0,255,255), 2)
    n    = len(g_pts)
    hint = LABELS[n] if n < 4 else "4/4 — press N or C"
    status = f"  {n}/4  |  next: {hint}  |  RClick=undo  R=reset  Q=quit"
    if extra_text:
        status += f"  |  {extra_text}"
    cv2.rectangle(out, (0,0), (out.shape[1], 36), (20,20,20), -1)
    cv2.putText(out, status, (4,25), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200,200,200), 1, cv2.LINE_AA)
    return out


def mouse_cb(event, x, y, flags, param):
    global g_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(g_pts) < 4:
        g_pts.append((x / g_scale, y / g_scale))
        cv2.imshow(g_win, draw_overlay())
    elif event == cv2.EVENT_RBUTTONDOWN and g_pts:
        g_pts.pop()
        cv2.imshow(g_win, draw_overlay())


# ══════════════════════════════════════════════════════════════════════
# Side selection
# ══════════════════════════════════════════════════════════════════════
def ask_side():
    print("\n" + "="*58)
    print("  Surround-View BEV Calibration")
    print("="*58)
    print("\n  Which camera do you want to calibrate?\n")
    for key, cfg in CAMERAS.items():
        print(f"    [{key[0].upper()}]  {key:6s} — {cfg['name']}  (S/N {cfg['serial']})")
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
    global g_pts, g_img, g_scale

    side = ask_side()
    cfg  = CAMERAS[side]
    cols, rows, bw, bh, gap, bx, by = board_info(side)

    print(f"\n{'='*58}")
    print(f"  Calibrating: {side.upper()}  —  {cfg['name']}  S/N {cfg['serial']}")
    print(f"  Board: {cols}×{rows} squares ({bw}×{bh}mm)  gap={gap}mm")
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
    print(f"  Mask: {mask_path}  ({BEV_W_px}×{BEV_H_px} px)")

    # Mask outer corners (computed from layout — same as world_corners())
    mask_corners = world_corners(side)
    print(f"  Mask corners (TL TR BR BL): {mask_corners.tolist()}\n")

    # ── Open camera ───────────────────────────────────────────────────
    cam, grab, kind = open_camera(side)
    print("  Warming up (30 frames)…")
    for _ in range(30):
        grab()

    # Display at half resolution for comfort — mouse coords scaled back automatically
    g_scale = 0.5

    cv2.namedWindow(g_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(g_win, 960, 600)   # half of 1920×1200 — fits on most screens
    cv2.setMouseCallback(g_win, mouse_cb)

    # ═════════════════════════════════════════════════════════════════
    # STEP 1 — Live snapshot
    # ═════════════════════════════════════════════════════════════════
    print("\nSTEP 1 — Live feed.")
    print("  S = snapshot   Q = quit")
    cam_frame = None
    while True:
        f = get_frame(cam, grab, kind)
        if f is None:
            continue
        disp = f.copy()
        cv2.rectangle(disp, (0,0), (disp.shape[1], 36), (20,20,20), -1)
        cv2.putText(disp,
                    f"  [{side.upper()}] {cfg['name']}  |  S=snapshot   Q=quit",
                    (4,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
        disp_half = cv2.resize(disp, (0,0), fx=0.5, fy=0.5)
        cv2.imshow(g_win, disp_half)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cam_frame = f.copy()
            print("  Snapshot captured.")
            break
        elif key == ord('q'):
            cam.close(); cv2.destroyAllWindows(); return

    # ═════════════════════════════════════════════════════════════════
    # STEP 2 — Pick corners on camera image (auto or manual)
    # ═════════════════════════════════════════════════════════════════
    print("\nSTEP 2 — Camera corners.")
    print("  A = auto-detect chessboard")
    print("  Or click the 4 outer corners manually (TL→TR→BR→BL) then press N")
    print("  Right-click=undo  |  R=reset  |  Q=quit")

    g_img = cam_frame
    g_pts = []
    cv2.imshow(g_win, draw_overlay("A=auto  N=accept(manual)"))

    cam_pts = None
    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('a'):
            # ── Auto detection ──────────────────────────────────────
            detected, vis = auto_detect_corners(cam_frame, side)
            if detected is not None:
                cam_pts = detected
                g_pts   = [tuple(p) for p in cam_pts]
                cv2.imshow(g_win, vis)
                print("  Auto-detect succeeded.  Press N to accept or R to reset.")
            else:
                cv2.imshow(g_win, vis)
                print("  Auto-detect failed — use manual clicking.")

        elif key == ord('r'):
            cam_pts = None
            g_pts   = []
            cv2.imshow(g_win, draw_overlay("A=auto  N=accept(manual)"))

        elif key == ord('n'):
            if len(g_pts) == 4:
                cam_pts = np.float32(g_pts)
                break
            else:
                print(f"  Need 4 points ({len(g_pts)}/4).")

        elif key == ord('q'):
            cam.close(); cv2.destroyAllWindows(); return

        # If auto already gave us 4 pts and user presses nothing, still show overlay
        if cam_pts is None:
            cv2.imshow(g_win, draw_overlay("A=auto  N=accept(manual)"))

    print(f"  Camera pts accepted: {cam_pts.tolist()}")

    # ═════════════════════════════════════════════════════════════════
    # STEP 3 — Pick corners on mask image (or use auto-computed ones)
    # ═════════════════════════════════════════════════════════════════
    print("\nSTEP 3 — Mask corners.")
    print("  U = use auto-computed mask corners (recommended)")
    print("  Or click the same 4 corners manually then press C")
    print("  Right-click=undo  |  R=reset  |  Q=quit")

    g_img = mask_img
    g_pts = []
    cv2.imshow(g_win, draw_overlay("U=auto-mask  C=compute(manual)"))

    mask_pts = None
    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u'):
            # Use the layout-computed corners directly
            mask_pts = mask_corners.copy()
            g_pts    = [tuple(p) for p in mask_pts]
            # Draw them on mask preview
            vis = mask_img.copy()
            labels = ["TL","TR","BR","BL"]
            colors = [(0,0,255),(0,128,255),(0,220,0),(255,0,0)]
            for i, pt in enumerate(mask_pts):
                x,y = int(pt[0]), int(pt[1])
                cv2.circle(vis, (x,y), 12, colors[i], -1)
                cv2.putText(vis, labels[i], (x+14,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3)
                cv2.putText(vis, labels[i], (x+14,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
            poly = mask_pts.astype(np.int32)
            cv2.polylines(vis, [poly], True, (0,255,255), 2)
            cv2.imshow(g_win, vis)
            print(f"  Auto mask corners: {mask_pts.tolist()}")
            print("  Press C to compute homography.")

        elif key == ord('r'):
            mask_pts = None
            g_pts    = []
            cv2.imshow(g_win, draw_overlay("U=auto-mask  C=compute(manual)"))

        elif key == ord('c'):
            if mask_pts is not None:
                break
            elif len(g_pts) == 4:
                mask_pts = np.float32(g_pts)
                break
            else:
                print(f"  Need 4 points ({len(g_pts)}/4).")

        elif key == ord('q'):
            cam.close(); cv2.destroyAllWindows(); return

        if mask_pts is None:
            cv2.imshow(g_win, draw_overlay("U=auto-mask  C=compute(manual)"))

    print(f"  Mask pts accepted: {mask_pts.tolist()}")

    # ═════════════════════════════════════════════════════════════════
    # STEP 4 — Compute homography
    # ═════════════════════════════════════════════════════════════════
    H, status_h = cv2.findHomography(cam_pts, mask_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("[ERROR] findHomography failed."); cam.close(); return

    print(f"\n  Homography H:\n{H}")
    print("\n  Reprojection errors (camera → mask):")
    for i in range(4):
        ph = H @ [cam_pts[i,0], cam_pts[i,1], 1.0]; ph /= ph[2]
        err = np.linalg.norm(ph[:2] - mask_pts[i])
        print(f"    {LABELS[i]:14s}  {err:.2f} px")

    # ═════════════════════════════════════════════════════════════════
    # STEP 5 — Live BEV preview
    # ═════════════════════════════════════════════════════════════════
    print("\nSTEP 5 — Live BEV preview.")
    print("  S = save homography   Q = quit")

    WIN_BEV = f"BEV [{side.upper()}]  |  S=save  Q=quit"
    cv2.namedWindow(WIN_BEV, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_BEV, int(BEV_W_px * 0.2), int(BEV_H_px * 0.2))

    while True:
        f = get_frame(cam, grab, kind)
        if f is None:
            continue
        bev = cv2.warpPerspective(f, H, (BEV_W_px, BEV_H_px))
        cam_disp = cv2.resize(f,   (0,0), fx=0.5, fy=0.5)
        bev_disp = cv2.resize(bev, (0,0), fx=0.2, fy=0.2)
        cv2.imshow(g_win,   cam_disp)
        cv2.imshow(WIN_BEV, bev_disp)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            fname_npy = f"homography_{side}.npy"
            fname_png = f"bev_{side}.png"
            np.save(fname_npy,
                    {"H":             H,
                     "board_side":    side,
                     "bev_size":      (BEV_W_px, BEV_H_px),
                     "camera_serial": cfg["serial"],
                     "camera_name":   cfg["name"],
                     "camera_pts":    cam_pts.tolist(),
                     "mask_pts":      mask_pts.tolist()},
                    allow_pickle=True)
            cv2.imwrite(fname_png, bev)
            print(f"  Saved {fname_npy}  +  {fname_png}")
        elif key == ord('q'):
            break

    cam.close()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()