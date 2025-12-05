import pyrealsense2 as rs
import numpy as np
import cv2

class DualRealTimeBEV:
    def __init__(self, width=640, height=480, fps=30):

        # =========SERIAL NUMBERS =========
        self.cam1_serial = "151422250222"
        self.cam2_serial = "151422253555"
        # ===============================================

        # Original BEV warp resolution
        self.src_width = 600
        self.src_height = 800

        # ★ RESIZE SCALE (0.5 = half size) ★
        self.scale = 0.9

        # Compute final BEV size
        self.bev_width = int(self.src_width * self.scale)
        self.bev_height = int(self.src_height * self.scale)

        # ★ SPACE BETWEEN BEVs (vertical space in pixels) ★
        self.gap_px = 10

        # Final canvas height = BEV1 + gap + BEV2
        self.canvas_height = self.bev_height * 2 + self.gap_px
        self.canvas_width = self.bev_width

        # Load homography matrices
        self.H1 = np.load("homography.npy")
        self.H2 = np.load("homography.npy")

        print("Loaded homography.npy")

        # --- Setup camera 1 ---
        self.pipeline1 = rs.pipeline()
        self.config1 = rs.config()
        self.config1.enable_device(self.cam1_serial)
        self.config1.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # --- Setup camera 2 ---
        self.pipeline2 = rs.pipeline()
        self.config2 = rs.config()
        self.config2.enable_device(self.cam2_serial)
        self.config2.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.pipeline1.start(self.config1)
        self.pipeline2.start(self.config2)

    def resize_bev(self, bev):
        return cv2.resize(bev, (self.bev_width, self.bev_height), interpolation=cv2.INTER_LINEAR)

    def run(self):
        print("Running BEV with spacing & resize. Press 'q' to quit.")

        try:
            while True:
                # ========== GET FRAMES ==========
                frames1 = self.pipeline1.wait_for_frames()
                frames2 = self.pipeline2.wait_for_frames()

                frame1 = frames1.get_color_frame()
                frame2 = frames2.get_color_frame()

                if not frame1 or not frame2:
                    continue

                img1 = np.asanyarray(frame1.get_data())
                img2 = np.asanyarray(frame2.get_data())

                # ========== APPLY HOMOGRAPHIES ==========
                bev1_large = cv2.warpPerspective(img1, self.H1, (self.src_width, self.src_height))
                bev2_large = cv2.warpPerspective(img2, self.H2, (self.src_width, self.src_height))

                # Flip BEV2 vertically
                bev2_large = cv2.flip(bev2_large, 0)

                # Resize both BEVs
                bev1 = self.resize_bev(bev1_large)
                bev2 = self.resize_bev(bev2_large)

                # ========== CREATE CANVAS ==========
                canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)

                # Top BEV placement
                canvas[0:self.bev_height, :] = bev1

                # Bottom BEV placement with GAP
                start_y = self.bev_height + self.gap_px
                canvas[start_y:start_y+self.bev_height, :] = bev2

                # ========== SHOW RESULTS ==========
                cv2.imshow("Camera 1 RAW", img1)
                cv2.imshow("Camera 2 RAW", img2)
                cv2.imshow("BEV Combined Output", canvas)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline1.stop()
            self.pipeline2.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    rt = DualRealTimeBEV()
    rt.run()
