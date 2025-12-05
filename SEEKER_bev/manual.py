import pyrealsense2 as rs
import numpy as np
import cv2

class ManualHomographyCalibration:
    def __init__(self, width=1280, height=800, fps=8):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.pipeline.start(self.config)

        # Chessboard settings
        self.chessboard_size = (8, 6)
        self.square_size = 60  # mm

        # BEV settings
        self.matrix = None
        self.matrix_inv = None
        self.bev_width = 600
        self.bev_height = 800
        self.mm_per_pixel = 5

        # Manual selection state
        self.selected_points = []
        self.current_image = None
        self.display_image = None
        self.point_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for corner selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.selected_points) < 4:
                self.selected_points.append((x, y))
                print(f"✓ Selected {self.point_labels[len(self.selected_points)-1]}: ({x}, {y})")
                
                # Redraw the image with all selected points
                self.display_image = self.current_image.copy()
                self.draw_selected_points()
                cv2.imshow("Manual Corner Selection", self.display_image)
                
                if len(self.selected_points) == 4:
                    print("\n✓ All 4 corners selected!")
                    print("Press 'c' to compute homography")
                    print("Press 'r' to reset and select again")
            else:
                print("All 4 corners already selected. Press 'r' to reset.")

    def draw_selected_points(self):
        """Draw selected points and connecting lines."""
        for i, pt in enumerate(self.selected_points):
            # Draw circle at point
            cv2.circle(self.display_image, pt, 1, (0, 255, 0), -1)
            cv2.circle(self.display_image, pt, 2, (255, 255, 255), 2)
            
            # Draw label
            label = self.point_labels[i]
            cv2.putText(self.display_image, label, (pt[0] + 15, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw lines connecting the points
        if len(self.selected_points) >= 2:
            for i in range(len(self.selected_points)):
                pt1 = self.selected_points[i]
                pt2 = self.selected_points[(i + 1) % len(self.selected_points)]
                if i < len(self.selected_points) - 1 or len(self.selected_points) == 4:
                    cv2.line(self.display_image, pt1, pt2, (255, 0, 0), 2)
        
        # Draw instructions
        instruction = f"Click {self.point_labels[len(self.selected_points)]} corner" if len(self.selected_points) < 4 else "Press 'c' to compute or 'r' to reset"
        cv2.putText(self.display_image, instruction, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw count
        cv2.putText(self.display_image, f"Points: {len(self.selected_points)}/4",
                   (10, self.display_image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def compute_homography(self):
        """Compute homography from manually selected corners."""
        if len(self.selected_points) != 4:
            print("❌ Need exactly 4 points to compute homography!")
            return False
        
        # Source points (from user selection)
        src_points = np.float32(self.selected_points)
        
        # Calculate destination points in BEV space
        W_mm = (self.chessboard_size[0] - 1) * self.square_size
        H_mm = (self.chessboard_size[1] - 1) * self.square_size
        
        W_px = int(W_mm / self.mm_per_pixel)
        H_px = int(H_mm / self.mm_per_pixel)
        
        margin_x = (self.bev_width - W_px) / 2
        margin_y = self.bev_height - H_px - 50
        
        dst_points = np.float32([
            [margin_x, margin_y],                       # TL
            [margin_x + W_px, margin_y],               # TR
            [margin_x + W_px, margin_y + H_px],        # BR
            [margin_x, margin_y + H_px]                # BL
        ])
        
        # Compute homography
        self.matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.matrix_inv = np.linalg.inv(self.matrix)
        
        # # Save matrices
        # np.save("homography.npy", self.matrix)
        # np.save("homography_inv.npy", self.matrix_inv)
        
        print("\n✓ Homography computed successfully!")
        print("✓ Saved homography matrix → homography.npy")
        print("✓ Saved inverse matrix → homography_inv.npy")
        
        return True

    def capture_frame(self):
        """Capture a single frame for manual selection."""
        print("\nCapturing frame...")
        print("Position the chessboard in view and press 's' to capture")
        print("Press 'q' to quit\n")
        
        while True:
            frames = self.pipeline.wait_for_frames()
            frame = frames.get_color_frame()
            if not frame:
                continue
            
            img = np.asanyarray(frame.get_data())
            
            display = img.copy()
            cv2.putText(display, "Press 's' to capture frame, 'q' to quit",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Live Feed", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self.current_image = img.copy()
                self.display_image = img.copy()
                cv2.destroyWindow("Live Feed")
                return True
            elif key == ord('q'):
                return False

    def manual_selection_mode(self):
        """Enter manual corner selection mode."""
        print("\n=== Manual Corner Selection Mode ===")
        print("Click on the 4 corners of your chessboard in this order:")
        print("1. Top-Left corner")
        print("2. Top-Right corner")
        print("3. Bottom-Right corner")
        print("4. Bottom-Left corner\n")
        
        cv2.namedWindow("Manual Corner Selection")
        cv2.setMouseCallback("Manual Corner Selection", self.mouse_callback)
        
        self.draw_selected_points()
        cv2.imshow("Manual Corner Selection", self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and len(self.selected_points) == 4:
                if self.compute_homography():
                    break
            elif key == ord('r'):
                print("\n↻ Resetting points...")
                self.selected_points = []
                self.display_image = self.current_image.copy()
                self.draw_selected_points()
                cv2.imshow("Manual Corner Selection", self.display_image)
            elif key == ord('q'):
                break
        
        cv2.destroyWindow("Manual Corner Selection")

    def show_bev_preview(self):
        """Show BEV preview after homography is computed."""
        if self.matrix is None:
            print("❌ No homography matrix computed yet!")
            return
        
        print("\n=== BEV Preview Mode ===")
        print("Showing bird's eye view transformation")
        print("Press 'q' to quit\n")
        
        while True:
            frames = self.pipeline.wait_for_frames()
            frame = frames.get_color_frame()
            if not frame:
                continue
            
            img = np.asanyarray(frame.get_data())
            
            # Apply homography
            bev = cv2.warpPerspective(img, self.matrix,
                                     (self.bev_width, self.bev_height))
            
            # Create black canvas
            black = np.zeros((2000, 1000, 3), dtype=np.uint8)
            h, w = bev.shape[:2]
            
            x_offset = 200
            y_offset = 200
            
            black[y_offset:y_offset+h, x_offset:x_offset+w] = bev
            
            # Show original and BEV
            cv2.imshow("Original", img)
            cv2.imshow("Bird's Eye View", black)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()

    def run(self):
        """Main execution flow."""
        try:
            # Step 1: Capture frame
            if not self.capture_frame():
                print("Exiting...")
                return
            
            # Step 2: Manual corner selection
            self.manual_selection_mode()
            
            # Step 3: Show BEV preview if homography was computed
            if self.matrix is not None:
                self.show_bev_preview()
            
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    calibration = ManualHomographyCalibration()
    calibration.run()