import pyrealsense2 as rs
import numpy as np
import cv2


class DualBEVChessboard:
    def __init__(self, width=640, height=480, fps=30):

        # ------------------------------
        # SERIAL NUMBERS (SET YOUR OWN!)
        # ------------------------------
        self.cam1_serial = "151422250222"
        self.cam2_serial = "151422253555"

        self.width = width
        self.height = height
        self.fps = fps

        # ----------- CAMERA 1 -----------
        self.pipeline1 = rs.pipeline()
        self.config1 = rs.config()
        self.config1.enable_device(self.cam1_serial)
        self.config1.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # ----------- CAMERA 2 -----------
        self.pipeline2 = rs.pipeline()
        self.config2 = rs.config()
        self.config2.enable_device(self.cam2_serial)
        self.config2.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        print("\nStarting both cameras...")
        self.pipeline1.start(self.config1)
        self.pipeline2.start(self.config2)
        print("Both cameras started ✓")

        # Chessboard
        self.chessboard_size = (8, 6)
        self.square_size = 60

        # BEV settings
        self.mm_per_pixel = 2
        self.bev_width = 600
        self.bev_height = 800

        # Hold homographies for both cameras
        self.H1 = None
        self.H2 = None

    # ------------------------------------------------------------
    # CHESSBOARD FINDER
    # ------------------------------------------------------------
    def find_chessboard(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return ret, corners

    # ------------------------------------------------------------
    # COMPUTE BEV HOMOGRAPHY MATRIX
    # ------------------------------------------------------------
    def compute_bev_transform(self, corners):
        src_points = np.float32([
            corners[0][0],
            corners[self.chessboard_size[0] - 1][0],
            corners[-1][0],
            corners[-self.chessboard_size[0]][0]
        ])

        W_mm = 3* self.square_size
        H_mm = (self.chessboard_size[1] - 1) * self.square_size

        W_px = int(W_mm / self.mm_per_pixel)
        H_px = int(H_mm / self.mm_per_pixel)

        # CENTER on mask
        margin_x = (self.bev_width - W_px) / 2
        margin_y = (self.bev_height - H_px) / 2

        dst_points = np.float32([
            [margin_x, margin_y],
            [margin_x + W_px, margin_y],
            [margin_x + W_px, margin_y + H_px],
            [margin_x, margin_y + H_px]
        ])

        H = cv2.getPerspectiveTransform(src_points, dst_points)
        return H

    # ------------------------------------------------------------
    # MAIN PROGRAM LOOP
    # ------------------------------------------------------------
    def run(self):
        print("\n=== Dual-camera BEV system ===")
        print("Press 1 → Capture BEV for CAMERA 1")
        print("Press 2 → Capture BEV for CAMERA 2")
        print("Press q → Quit\n")

        try:
            while True:
                # Read frames
                frame1 = self.pipeline1.wait_for_frames().get_color_frame()
                frame2 = self.pipeline2.wait_for_frames().get_color_frame()

                img1 = np.asanyarray(frame1.get_data())
                img2 = np.asanyarray(frame2.get_data())

                display1 = img1.copy()
                display2 = img2.copy()

                # Try to detect chessboard in both
                ret1, corners1 = self.find_chessboard(img1)
                ret2, corners2 = self.find_chessboard(img2)

                if ret1:
                    cv2.drawChessboardCorners(display1, self.chessboard_size, corners1, True)

                if ret2:
                    cv2.drawChessboardCorners(display2, self.chessboard_size, corners2, True)

                # Show inputs
                cv2.imshow("CAMERA 1", display1)
                cv2.imshow("CAMERA 2", display2)

                # If both BEVs exist → render them together
                if self.H1 is not None and self.H2 is not None:
                    mask = np.zeros((self.bev_height, self.bev_width * 2, 3), dtype=np.uint8)

                    bev1 = cv2.warpPerspective(img1, self.H1, (self.bev_width, self.bev_height))
                    bev2 = cv2.warpPerspective(img2, self.H2, (self.bev_width, self.bev_height))

                    mask[:, :self.bev_width] = bev1
                    mask[:, self.bev_width:] = bev2

                    cv2.imshow("COMBINED BEV (LEFT=CAM1, RIGHT=CAM2)", mask)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                # PRESS 1 to capture BEV for camera 1
                if key == ord('1') and ret1:
                    self.H1 = self.compute_bev_transform(corners1)
                    #np.save("H_cam1.npy", self.H1)
                    print("✔ Saved homography for CAMERA 1")

                # PRESS 2 to capture BEV for camera 2
                if key == ord('2') and ret2:
                    self.H2 = self.compute_bev_transform(corners2)
                    #np.save("H_cam2.npy", self.H2)
                    print("✔ Saved homography for CAMERA 2")

        finally:
            self.pipeline1.stop()
            self.pipeline2.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    d = DualBEVChessboard()
    d.run()
