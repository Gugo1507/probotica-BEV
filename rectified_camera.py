import av
import cv2
import numpy as np
import json
import os

# ─── Load intrinsics ──────────────────────────────────────────────────────────
with open("camera_intrinsics.json") as f:
    cal = json.load(f)

K     = np.array(cal["camera_matrix"])
dist  = np.array(cal["dist_coefficients"])
cal_w, cal_h = cal["calibration_info"]["image_size_wh"]

map1, map2   = None, None
last_size    = None
scaled_new_K = None
scaled_roi   = None

def build_maps(w, h, alpha=0.1):
    """
    alpha=0   → full crop, no black borders, loses most FOV
    alpha=0.1 → tiny crop, almost no FOV loss, minimal black edge
    alpha=1   → no crop at all, full black borders visible
    """
    sx, sy = w / cal_w, h / cal_h

    sK = K.copy()
    sK[0] *= sx
    sK[1] *= sy

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        sK, dist, (w, h),
        alpha=alpha,          # <-- KEY: controls how much to crop
        newImgSize=(w, h)
    )

    # Precompute remap tables with LANCZOS-quality interpolation hint
    # (actual interpolation is set in cv2.remap call)
    m1, m2 = cv2.initUndistortRectifyMap(
        sK, dist, None, new_K, (w, h), cv2.CV_16SC2
    )
    return m1, m2, new_K, roi

# ─── Optional: inpaint black border mask ─────────────────────────────────────
def inpaint_borders(img):
    """Fill leftover black edge pixels using neighbouring content."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Mask = pixels that are exactly black after remap
    mask = (gray == 0).astype(np.uint8) * 255
    # Only bother if there are actually black pixels
    if mask.any():
        img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return img

# ─── Stream ───────────────────────────────────────────────────────────────────
container = av.open(
    "rtsp://admin:123456@192.168.10.151:554/stream1",
    options={"rtsp_transport": "tcp"}
)

rectify    = True
use_inpaint = False   # toggle with 'i' — slower but fills any remaining gaps
alpha       = 0.1     # adjust with +/- keys
snapshot_dir = "snapshots"
os.makedirs(snapshot_dir, exist_ok=True)

print("Controls:  Q=quit  S=snapshot  R=toggle rectify  I=toggle inpaint  +/-=alpha")

for frame in container.decode(video=0):
    if frame.width == 0 or frame.height == 0:
        continue
    try:
        img = frame.to_ndarray(format='bgr24')
    except Exception as e:
        print(f"[WARN] Preskačem frame: {e}")
        continue

    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    h, w = img.shape[:2]

    # Rebuild maps if resolution or alpha changed
    if (w, h, alpha) != last_size:
        map1, map2, scaled_new_K, scaled_roi = build_maps(w, h, alpha)
        last_size = (w, h, alpha)
        print(f"[INFO] Maps rebuilt: {w}×{h}  alpha={alpha:.2f}")

    if rectify:
        # INTER_LANCZOS4 → sharper edges, less ringing than LINEAR
        img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Crop valid region (alpha controls how much is cropped)
        x, y, rw, rh = scaled_roi
        if rw > 10 and rh > 10:
            img = img[y:y+rh, x:x+rw]

        # Optional inpaint pass to fill any remaining black edge pixels
        if use_inpaint:
            img = inpaint_borders(img)

    # OSD
    label = f"{'RECT' if rectify else 'RAW'}"
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    img=cv2.resize(img,(0,0),fx=1.4,fy=1.4)
    cv2.imshow("RTSP Frame", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(snapshot_dir, f"snapshot_{ts}.jpg")
        cv2.imwrite(fname, img)
        print(f"[INFO] Saved: {fname}")
    elif key == ord('r'):
        rectify = not rectify
        print(f"[INFO] Rectify: {'ON' if rectify else 'OFF'}")
    elif key == ord('i'):
        use_inpaint = not use_inpaint
        print(f"[INFO] Inpaint: {'ON' if use_inpaint else 'OFF'}")
    elif key == ord('+') or key == ord('='):
        alpha = min(1.0, round(alpha + 0.05, 2))
    elif key == ord('-'):
        alpha = max(0.0, round(alpha - 0.05, 2))

cv2.destroyAllWindows()
container.close()