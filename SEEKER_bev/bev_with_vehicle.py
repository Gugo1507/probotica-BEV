import pyrealsense2 as rs
import numpy as np
import cv2

class DualRealTimeBEV:
    def __init__(self, width=640, height=480, fps=30):

        # ========= FILL IN YOUR SERIAL NUMBERS =========
        self.cam1_serial = "151422250222"
        self.cam2_serial = "151422253555"
        # ===============================================

        # Original BEV warp resolution
        self.src_width = 600
        self.src_height = 800

        # ★ RESIZE SCALE (0.5 = half size)
        self.scale = 0.8

        # Final BEV sizes
        self.bev_width = int(self.src_width * self.scale)
        self.bev_height = int(self.src_height * self.scale)

        # ★ Vertical gap between BEVs
        self.gap_px = 10

        # Final canvas size
        self.canvas_height = self.bev_height * 2 + self.gap_px
        self.canvas_width = self.bev_width

        # Load homographies
        self.H1 = np.load("homography.npy")
        self.H2 = np.load("homography.npy")

        print("Loaded homography_cam1.npy and homography_cam2.npy")

        # Setup camera 1
        self.pipeline1 = rs.pipeline()
        self.config1 = rs.config()
        self.config1.enable_device(self.cam1_serial)
        self.config1.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Setup camera 2
        self.pipeline2 = rs.pipeline()
        self.config2 = rs.config()
        self.config2.enable_device(self.cam2_serial)
        self.config2.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.pipeline1.start(self.config1)
        self.pipeline2.start(self.config2)

    def resize_bev(self, bev):
        return cv2.resize(bev, (self.bev_width, self.bev_height), interpolation=cv2.INTER_LINEAR)

    def draw_vehicle(self, canvas):
        """Draw a more realistic top-down vehicle representation"""
        # Vehicle dimensions (as percentage of BEV width)
        vehicle_width = int(self.bev_width * 0.3)
        vehicle_length = int(self.gap_px * 3.5)
        
        # Center the vehicle horizontally
        center_x = self.canvas_width // 2
        center_y = self.bev_height + self.gap_px // 2
        
        # Main body dimensions
        x1 = center_x - vehicle_width // 2
        x2 = center_x + vehicle_width // 2
        y1 = center_y - vehicle_length
        y2 = center_y + vehicle_length
        
        # Draw main vehicle body (darker blue/gray)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 80, 40), -1)
        
        # Draw windshield (lighter blue - top section)
        windshield_height = vehicle_length // 4
        cv2.rectangle(canvas, (x1 + 5, y1), (x2 - 5, y1 + windshield_height), 
                     (180, 140, 80), -1)
        
        # Draw rear window (bottom section)
        cv2.rectangle(canvas, (x1 + 5, y2 - windshield_height), (x2 - 5, y2), 
                     (180, 140, 80), -1)
        
        # Draw side mirrors
        mirror_size = 8
        # Left mirror
        cv2.circle(canvas, (x1 - 3, center_y - vehicle_length // 4), 
                  mirror_size // 2, (80, 60, 30), -1)
        # Right mirror
        cv2.circle(canvas, (x2 + 3, center_y - vehicle_length // 4), 
                  mirror_size // 2, (80, 60, 30), -1)
        
        # Draw wheels (black circles)
        wheel_radius = 6
        wheel_offset_x = vehicle_width // 3
        wheel_offset_y = vehicle_length // 3
        
        # Front-left wheel
        cv2.circle(canvas, (center_x - wheel_offset_x, y1 + 10), 
                  wheel_radius, (30, 30, 30), -1)
        # Front-right wheel
        cv2.circle(canvas, (center_x + wheel_offset_x, y1 + 10), 
                  wheel_radius, (30, 30, 30), -1)
        # Rear-left wheel
        cv2.circle(canvas, (center_x - wheel_offset_x, y2 - 10), 
                  wheel_radius, (30, 30, 30), -1)
        # Rear-right wheel
        cv2.circle(canvas, (center_x + wheel_offset_x, y2 - 10), 
                  wheel_radius, (30, 30, 30), -1)
        
        # Draw outline
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Add direction indicator (front of vehicle - triangle)
        triangle_pts = np.array([
            [center_x, y1 - 8],
            [center_x - 10, y1],
            [center_x + 10, y1]
        ], np.int32)
        cv2.fillPoly(canvas, [triangle_pts], (0, 255, 0))

    def run(self):
        print("Running BEV with spacing, resize, and vehicle rectangle. Press 'q' to quit.")

        try:
            while True:
                # ====== READ FRAMES ======
                frames1 = self.pipeline1.wait_for_frames()
                frames2 = self.pipeline2.wait_for_frames()

                frame1 = frames1.get_color_frame()
                frame2 = frames2.get_color_frame()

                if not frame1 or not frame2:
                    continue

                img1 = np.asanyarray(frame1.get_data())
                img2 = np.asanyarray(frame2.get_data())

                # ====== APPLY WARP ======
                bev1_large = cv2.warpPerspective(img1, self.H1, (self.src_width, self.src_height))
                bev2_large = cv2.warpPerspective(img2, self.H2, (self.src_width, self.src_height))

                # Flip BEV2 vertically
                bev2_large = cv2.flip(bev2_large, 0)

                # Resize
                bev1 = self.resize_bev(bev1_large)
                bev2 = self.resize_bev(bev2_large)

                # ====== CREATE CANVAS ======
                canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)

                # Place BEVs
                canvas[0:self.bev_height, :] = bev1

                start_y = self.bev_height + self.gap_px
                canvas[start_y:start_y + self.bev_height, :] = bev2

                # ====== DRAW VEHICLE ======
                self.draw_vehicle(canvas)

                # ====== DISPLAY ======
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