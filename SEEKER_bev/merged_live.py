import pyrealsense2 as rs
import numpy as np
import cv2

class LiveBEVFromSavedHomography:
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

        self.bev_width = 600
        self.bev_height = 800

        self.H_cam1 = np.load("saved_homographies/homographytest_cam1.npy")
        self.H_cam2 = np.load("saved_homographies/homographytest_cam2.npy")

    def merge_bevs(self, bev1, bev2):
        h, w = bev1.shape[:2]
        canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
        canvas[:, :w] = bev2
        canvas[:, w:] = bev1
        return canvas

    def run(self):
        print("\n--- LIVE MERGED BEV FROM SAVED MATRICES ---")

        try:
            while True:
                bevs = {}

                for cam in ["cam1", "cam2"]:
                    frames = self.pipelines[cam].wait_for_frames()
                    img = np.asanyarray(frames.get_color_frame().get_data())

                    H = self.H_cam1 if cam == "cam1" else self.H_cam2
                    bev = cv2.warpPerspective(img, H, (self.bev_width, self.bev_height))

                    bevs[cam] = bev
                    cv2.imshow(f"RAW {cam}", img)

                merged = self.merge_bevs(bevs["cam1"], bevs["cam2"])
                merged= cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
                cv2.imshow("MERGED BEV (LOADED)", merged)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            for cam in self.pipelines:
                self.pipelines[cam].stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = LiveBEVFromSavedHomography()
    app.run()
