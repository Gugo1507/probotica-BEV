import pyrealsense2 as rs
import numpy as np
import cv2


class BirdsEyeViewAruco:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(self.config)

        # ========== ARUCO SETTINGS ==========
        self.ARUCO_DICT = cv2.aruco.DICT_4X4_50
        self.MARKER_SIZE_MM = 200  # ✅ YOUR MARKER SIZE (200x200 mm)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # ========== BEV SETTINGS ==========
        self.matrix = None
        self.matrix_inv = None
        self.bev_width = 800
        self.bev_height = 800
        self.mm_per_pixel = 2  # 1 pixel = 2 mm

    # ============================
    # ARUCO DETECTION
    # ============================
    def detect_aruco(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is None:
            return False, None, None

        # ✅ FIXED: extract 4x2 corner matrix
        marker_corners = corners[0][0]   # shape = (4, 2)
        marker_id = ids[0][0]

        return True, marker_corners, marker_id

    # ============================
    # HOMOGRAPHY
    # ============================
    def compute_bev_transform(self, corners):
        src_points = np.float32([
            corners[0],  # TL
            corners[1],  # TR
            corners[2],  # BR
            corners[3]   # BL
        ])

        W_mm = self.MARKER_SIZE_MM
        H_mm = self.MARKER_SIZE_MM

        W_px = int(W_mm / self.mm_per_pixel)
        H_px = int(H_mm / self.mm_per_pixel)

        margin_x = (self.bev_width - W_px) / 2
        margin_y = (self.bev_height - H_px) / 2+100

        dst_points = np.float32([
            [margin_x, margin_y],                       # TL
            [margin_x + W_px, margin_y],               # TR
            [margin_x + W_px, margin_y + H_px],        # BR
            [margin_x, margin_y + H_px]                # BL
        ])

        H = cv2.getPerspectiveTransform(src_points, dst_points)
        self.matrix = H
        self.matrix_inv = np.linalg.inv(H)

        np.save("homography.npy", self.matrix)
        np.save("homography_inv.npy", self.matrix_inv)

        print("✓ Saved homography.npy")
        print("✓ Saved homography_inv.npy")

        return src_points

    # ============================
    # DRAWING
    # ============================
    def draw_marker(self, image, corners, marker_id, src_points=None):
        display = image.copy()

        # ✅ FIX: reshape for OpenCV drawing
        draw_corners = corners.reshape(1, 4, 2)
        cv2.aruco.drawDetectedMarkers(display, [draw_corners])

        if src_points is not None:
            for pt in src_points:
                x, y = pt.astype(int)
                cv2.circle(display, (x, y), 8, (0, 255, 0), -1)

        cv2.putText(display, f"ID: {marker_id}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        return display

    def draw_centering_guides(self, frame, corners):
        xs = corners[:, 0]
        ys = corners[:, 1]

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                      (0, 255, 0), 2)

        L = x_min
        R = frame.shape[1] - x_max
        T = y_min
        B = frame.shape[0] - y_max

        cv2.putText(frame, f"L: {L}px", (10, frame.shape[0] - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"R: {R}px", (10, frame.shape[0] - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"T: {T}px", (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"B: {B}px", (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return frame

    # ============================
    # MAIN LOOP
    # ============================
    def run(self):
        print("Press 'H' to compute homography")
        print("Press 'Q' to quit\n")

        src_points = None

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                frame = frames.get_color_frame()
                if not frame:
                    continue

                img = np.asanyarray(frame.get_data())
                found, corners, marker_id = self.detect_aruco(img)

                if found:
                    display = self.draw_marker(img, corners, marker_id, src_points)
                    #display = self.draw_centering_guides(display, corners)
                else:
                    display = img.copy()
                    cv2.putText(display, "Looking for ArUco...",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

                cv2.imshow("Input", display)

                if self.matrix is not None:
                    bev = cv2.warpPerspective(
                        img, self.matrix,
                        (self.bev_width, self.bev_height)
                    )

                    black = np.zeros((1200, 1200, 3), dtype=np.uint8)
                    h, w = bev.shape[:2]
                    black[200:200 + h, 200:200 + w] = bev
                    cv2.imshow("Birds Eye View", black)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                if key == ord('h') and found:
                    src_points = self.compute_bev_transform(corners)

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    bev = BirdsEyeViewAruco()
    bev.run()
