import cv2

cap_usb = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap_usb.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap_usb.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap_usb.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap_usb.set(cv2.CAP_PROP_FPS, 30)

if not cap_usb.isOpened():
    print("Error: Could not open USB camera.")
    exit()

import pyzed.sl as sl

# Create a ZED camera object
zed = sl.Camera()

# Set configuration
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  # or HD1080
init_params.camera_fps = 30
init_params.sdk_verbose = True

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Error: {err}")
    exit()

# Prepare image container
image_zed = sl.Mat()

while True:
    # USB camera
    ret_usb, frame_usb = cap_usb.read()
    frame_usb = cv2.resize(frame_usb,(0,0),fx=0.7,fy=0.7)
    if not ret_usb:
        print("USB camera failed")
        break

    # ZED camera
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        frame_zed = image_zed.get_data()  # returns numpy array
        frame_zed = cv2.resize(frame_zed,(0,0),fx=0.5,fy=0.5)
    else:
        print("ZED camera failed")
        break

    # Show both
    cv2.imshow("USB Camera", frame_usb)
    cv2.imshow("ZED Camera", frame_zed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_usb.release()
zed.close()
cv2.destroyAllWindows()