import av
import cv2
import numpy as np
import json
import os
import threading

# ─── Load intrinsics ──────────────────────────────────────────────────────────
with open("camera_intrinsics.json") as f:
    cal = json.load(f)

K     = np.array(cal["camera_matrix"])
dist  = np.array(cal["dist_coefficients"])
cal_w, cal_h = cal["calibration_info"]["image_size_wh"]

map1, map2   = None, None
last_size    = None
scaled_roi   = None

def build_maps(w, h, alpha=0.1):
    sx, sy = w / cal_w, h / cal_h

    sK = K.copy()
    sK[0] *= sx
    sK[1] *= sy

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        sK, dist, (w, h),
        alpha=alpha,
        newImgSize=(w, h)
    )

    m1, m2 = cv2.initUndistortRectifyMap(
        sK, dist, None, new_K, (w, h), cv2.CV_16SC2
    )
    return m1, m2, roi

def inpaint_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray == 0).astype(np.uint8) * 255
    if mask.any():
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return img

# ─── Shared frames (thread-safe enough for this use) ──────────────────────────
rtsp_frame = None
usb_frame  = None
running = True

# ─── RTSP Thread ─────────────────────────────────────────────────────────────
def rtsp_worker():
    global rtsp_frame, running
    try:
        container = av.open(
            "rtsp://admin:123456@192.168.10.151:554/stream1",
            options={"rtsp_transport": "tcp", "fflags": "nobuffer"}
        )

        for frame in container.decode(video=0):
            if not running:
                break

            try:
                img = frame.to_ndarray(format='bgr24')
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                rtsp_frame = img
            except:
                continue

    except Exception as e:
        print("[RTSP ERROR]", e)

# ─── USB Thread ──────────────────────────────────────────────────────────────
def usb_worker():
    global usb_frame, running

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("[WARN] USB camera failed")
        return

    while running:
        ret, frame = cap.read()
        if ret:
            usb_frame = frame

    cap.release()

# ─── Start threads ───────────────────────────────────────────────────────────
threading.Thread(target=rtsp_worker, daemon=True).start()
threading.Thread(target=usb_worker, daemon=True).start()

# ─── Controls ────────────────────────────────────────────────────────────────
rectify = True
use_inpaint = False
alpha = 0.1

print("Controls: Q=quit R=rectify I=inpaint +/-=alpha")

# ─── Main Display Loop ───────────────────────────────────────────────────────
while True:
    if rtsp_frame is None:
        continue

    img = rtsp_frame.copy()
    h, w = img.shape[:2]

    # Build maps if needed
    
    if (w, h, alpha) != last_size:
        map1, map2, scaled_roi = build_maps(w, h, alpha)
        last_size = (w, h, alpha)
        print(f"[INFO] Maps rebuilt: {w}x{h} alpha={alpha:.2f}")

    # Rectify
    if rectify:
        img = cv2.remap(img, map1, map2,
                        interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)

        x, y, rw, rh = scaled_roi
        if rw > 10 and rh > 10:
            img = img[y:y+rh, x:x+rw]

        if use_inpaint:
            img = inpaint_borders(img)

    # USB processing
    usb = None
    if usb_frame is not None:
        usb = usb_frame.copy()
        uh, uw = usb.shape[:2]
        scale = img.shape[0] / uh
        usb = cv2.resize(usb, (int(uw * scale), img.shape[0]))

    # Labels
    cv2.putText(img, "RTSP", (10, img.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    if usb is not None:
        cv2.putText(usb, "USB", (10, usb.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Combine
    if usb is not None:
        combined = np.hstack((img, usb))
    else:
        combined = img

    # OSD
    label = f"{'RECT' if rectify else 'RAW'} alpha={alpha:.2f}"
    cv2.putText(combined, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    combined=cv2.resize(combined,(0,0),fx=1.5,fy=1.5)

    cv2.imshow("RTSP + USB", combined)

    # Keys
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        rectify = not rectify
    elif key == ord('i'):
        use_inpaint = not use_inpaint
    elif key == ord('+') or key == ord('='):
        alpha = min(1.0, alpha + 0.05)
    elif key == ord('-'):
        alpha = max(0.0, alpha - 0.05)

# ─── Shutdown ────────────────────────────────────────────────────────────────
running = False
cv2.destroyAllWindows()