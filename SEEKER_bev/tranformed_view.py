import pyrealsense2 as rs
import numpy as np
import cv2

class RealTimeBEV:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(self.config)

        # Load homography matrix
        self.H = np.load("homography.npy")
        print("Loaded homography matrix from homography.npy")

        # BEV size must match original code
        self.bev_width = 1200
        self.bev_height = 2000

    def run(self):
        print("Real-time BEV running. Press 'q' to quit.")

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                frame = frames.get_color_frame()
                if not frame:
                    continue

                img = np.asanyarray(frame.get_data())

                bev = cv2.warpPerspective(img, self.H, (self.bev_width, self.bev_height))

                cv2.imshow("Input", img)
                cv2.imshow("BEV", bev)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    rt = RealTimeBEV()
    rt.run()
