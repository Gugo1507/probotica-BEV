import pyrealsense2 as rs
import numpy as np
import cv2

class BirdsEyeViewChessboard:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(self.config)

        # Chessboard
        self.chessboard_size = (8, 6)
        self.square_size = 60  # mm

        # BEV settings
        self.matrix = None
        self.matrix_inv = None
        self.bev_width = 600
        self.bev_height = 800
        self.mm_per_pixel = 2

    def preprocess_image(self, image):
        """Enhanced preprocessing for better corner detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        return enhanced

    def find_chessboard(self, image):
        """Improved chessboard detection with multiple strategies."""
        # Try enhanced preprocessing
        gray = self.preprocess_image(image)
        
        # Strategy 1: Try with enhanced flags
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                 cv2.CALIB_CB_NORMALIZE_IMAGE + 
                 cv2.CALIB_CB_FAST_CHECK)
        
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags)
        
        # Strategy 2: If failed, try without fast check
        if not ret:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags)
        
        # Strategy 3: If still failed, try with basic grayscale
        if not ret:
            gray_basic = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_basic, self.chessboard_size, flags)
        
        # Refine corners if found
        if ret:
            # Use tighter criteria for better precision
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
            
            # Refine with cornerSubPix
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Additional validation: check if corners form reasonable pattern
            if not self.validate_corners(corners):
                return False, None

        return ret, corners

    def validate_corners(self, corners):
        """Validate that detected corners form a reasonable chessboard pattern."""
        if corners is None or len(corners) != self.chessboard_size[0] * self.chessboard_size[1]:
            return False
        
        # Check that corners are not all clustered in one area
        xs = corners[:, 0, 0]
        ys = corners[:, 0, 1]
        
        width = xs.max() - xs.min()
        height = ys.max() - ys.min()
        
        # Ensure minimum size (at least 100 pixels in each dimension)
        if width < 100 or height < 100:
            return False
        
        # Check aspect ratio is reasonable (between 0.5 and 2.0)
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False
        
        return True

    def compute_bev_transform(self, corners):
        """Compute homography matrix."""
        src_points = np.float32([
            corners[0][0],
            corners[self.chessboard_size[0] - 1][0],
            corners[-1][0],
            corners[-self.chessboard_size[0]][0]
        ])

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

        H = cv2.getPerspectiveTransform(src_points, dst_points)
        # H[1,2]=0
        print(H)
        self.matrix = H
        self.matrix_inv = np.linalg.inv(H)
        # np.save("homography.npy", self.matrix)
        # np.save("homography_inv.npy", self.matrix_inv)
        print("✓ Saved homography matrix → homography.npy")
        print("✓ Saved inverse matrix → homography_inv.npy")

        return src_points

    def draw_chessboard(self, image, corners, src_points=None):
        display = image.copy()
        cv2.drawChessboardCorners(display, self.chessboard_size, corners, True)

        if src_points is not None:
            for pt in src_points:
                x, y = pt.astype(int)
                cv2.circle(display, (x, y), 8, (0, 255, 0), -1)

        return display

    def draw_centering_guides(self, frame, corners):
        """Draw distance from chessboard bounding box to frame borders."""
        # Compute bounding box
        xs = corners[:, 0, 0]
        ys = corners[:, 0, 1]

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        # Draw the bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Margin distances
        L = x_min
        R = frame.shape[1] - x_max
        T = y_min
        B = frame.shape[0] - y_max

        # Overlay distances on frame
        cv2.putText(frame, f"L: {L}px", (10, frame.shape[0] - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"R: {R}px", (10, frame.shape[0] - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"T: {T}px", (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"B: {B}px", (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, "Center chessboard using margins, then press H",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def run(self):
        print("Press 'h' to compute homography matrix (and save it)")
        print("Press 'q' to quit\n")

        src_points = None

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                frame = frames.get_color_frame()
                if not frame:
                    continue

                img = np.asanyarray(frame.get_data())
                ret, corners = self.find_chessboard(img)

                if ret:
                    display = self.draw_chessboard(img, corners, src_points)
                    display = self.draw_centering_guides(display, corners)
                else:
                    display = img.copy()
                    cv2.putText(display, "Looking for chessboard...",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

                cv2.imshow("Input", display)

                if self.matrix is not None:
                    bev = cv2.warpPerspective(img, self.matrix,
                                              (self.bev_width, self.bev_height))
                    
                    black = np.zeros((2000, 1000, 3), dtype=np.uint8)
                    h, w = bev.shape[:2]

                    # Top-left corner where BEV will be placed
                    x_offset = 200   # shift right
                    y_offset = 200   # shift down

                    # Paste BEV inside black canvas
                    black[y_offset:y_offset+h, x_offset:x_offset+w] = bev
                    cv2.imshow("Black", black)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                if key == ord('h') and ret:
                    src_points = self.compute_bev_transform(corners)

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    bev = BirdsEyeViewChessboard()
    bev.run()