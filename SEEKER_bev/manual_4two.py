import pyrealsense2 as rs
import numpy as np
import cv2
import os

class ManualHomographyCalibrationDual:
    def __init__(self, width=1280, height=800, fps=8):
        self.cam1_serial = "151422250222"
        self.cam2_serial = "151422253555"

        self.pipelines = {}

        for cam_id, serial in zip(["cam1", "cam2"], [self.cam1_serial, self.cam2_serial]):
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            pipeline.start(config)
            self.pipelines[cam_id] = pipeline

        self.chessboard_size = (8, 6)
        self.square_size = 60  # mm

        self.bev_width = 600
        self.bev_height = 800
        self.mm_per_pixel = 3

        self.data = {
            "cam1": {"points": [], "matrix": None},
            "cam2": {"points": [], "matrix": None}
        }

        self.point_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        self.active_cam = None

        self.save_dir = "saved_homographies"
        os.makedirs(self.save_dir, exist_ok=True)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cam = self.active_cam
            if len(self.data[cam]["points"]) < 4:
                self.data[cam]["points"].append((x, y))
                print(f"✓ {cam} - Selected {self.point_labels[len(self.data[cam]['points'])-1]}")

    def compute_homography(self, cam):
        src_points = np.float32(self.data[cam]["points"])

        W_mm = 3* self.square_size
        H_mm = (self.chessboard_size[1] - 1) * self.square_size
        W_px = int(W_mm / self.mm_per_pixel)
        H_px = int(H_mm / self.mm_per_pixel)

        if cam == "cam1":
            margin_x = 0
        else:
            margin_x = self.bev_width - W_px

        margin_y = self.bev_height - H_px - 100

        dst_points = np.float32([
            [margin_x, margin_y],
            [margin_x + W_px, margin_y],
            [margin_x + W_px, margin_y + H_px],
            [margin_x, margin_y + H_px]
        ])

        H = cv2.getPerspectiveTransform(src_points, dst_points)
        self.data[cam]["matrix"] = H

        path = os.path.join(self.save_dir, f"homographytest2_{cam}.npy")
        np.save(path, H)

        print(f"✅ Homography computed and saved for {cam}")

    def capture_and_select(self, cam):
        print(f"\n--- Capture for {cam} ---")
        print("Press 's' to capture")

        pipeline = self.pipelines[cam]

        while True:
            frames = pipeline.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data())
            cv2.imshow(cam, img)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                captured = img.copy()
                cv2.destroyWindow(cam)
                break

        self.active_cam = cam
        self.data[cam]["points"] = []

        cv2.namedWindow(f"Select {cam}")
        cv2.setMouseCallback(f"Select {cam}", self.mouse_callback)

        while True:
            display = captured.copy()
            for pt in self.data[cam]["points"]:
                cv2.circle(display, pt, 2, (0, 255, 0), -1)

            cv2.imshow(f"Select {cam}", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(self.data[cam]["points"]) == 4:
                self.compute_homography(cam)
                break
            elif key == ord('r'):
                self.data[cam]["points"] = []

        cv2.destroyWindow(f"Select {cam}")

    def merge_bevs(self, bev1, bev2):
        h, w = bev1.shape[:2]
        canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
        canvas[:, :w] = bev2
        canvas[:, w:] = bev1
        return canvas

    def show_raw_and_merged_bev(self):
        print("\n--- RAW + MERGED BEV VIEW ---")

        while True:
            bevs = {}

            for cam in ["cam1", "cam2"]:
                frames = self.pipelines[cam].wait_for_frames()
                img = np.asanyarray(frames.get_color_frame().get_data())

                H = self.data[cam]["matrix"]
                bev = cv2.warpPerspective(img, H, (self.bev_width, self.bev_height))

                bevs[cam] = bev

                cv2.imshow(f"RAW {cam}", img)

            merged = self.merge_bevs(bevs["cam1"], bevs["cam2"])
            cv2.imshow("MERGED BEV (cam2 LEFT | cam1 RIGHT)", merged)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def run(self):
        try:
            for cam in ["cam1", "cam2"]:
                self.capture_and_select(cam)

            self.show_raw_and_merged_bev()

        finally:
            for cam in self.pipelines:
                self.pipelines[cam].stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    calib = ManualHomographyCalibrationDual()
    calib.run()
