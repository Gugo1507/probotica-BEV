import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

class BirdsEyeView4Aruco:
    def __init__(self,
                 width=1280, height=800, fps=8,
                 mm_per_pixel=1):
        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.pipeline.start(self.config)

        # ArUco settings
        self.aruco_dict_id = cv2.aruco.DICT_4X4_50
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_id)
        # DetectorParameters_create is compatible across many versions
        self.aruco_params = cv2.aruco.DetectorParameters()

        # The 4 marker IDs you said
        self.ids_needed = [1, 2, 3, 4]

        # REAL WORLD layout in mm (change if different)
        # IMPORTANT: This maps marker ID -> (X_mm, Y_mm) in world coordinates
        # The chosen mapping here:
        # ID 1 = bottom-left (0,0)
        # ID 2 = bottom-right (width_mm, 0)
        # ID 3 = top-right (width_mm, height_mm)
        # ID 4 = top-left (0, height_mm)
        width_mm = 350
        height_mm = 250
        self.WORLD_POINTS_MM = {
            1: (0.0, 0.0),
            2: (float(width_mm), 0.0),
            3: (float(width_mm), float(height_mm)),
            4: (0.0, float(height_mm)),
        }

        # BEV parameters
        self.mm_per_pixel = mm_per_pixel  # 1 px = mm_per_pixel mm
        self.bev_w_px = int(width_mm / self.mm_per_pixel)
        self.bev_h_px = int(height_mm / self.mm_per_pixel)

        # larger black canvas to paste BEV into (for nicer visualization)
        self.canvas_w = max(self.bev_w_px + 400, 1200)
        self.canvas_h = max(self.bev_h_px + 400, 900)
        self.canvas_offset_x = 200
        self.canvas_offset_y = 200

        # homography matrices
        self.H = None
        self.H_inv = None

    def get_marker_centers(self, corners_list, ids_array):
        """
        corners_list: list/array of marker corner arrays returned by detectMarkers
        ids_array: corresponding ids (N,1)
        returns dict id->(cx,cy) in image pixels
        """
        centers = {}
        if ids_array is None:
            return centers

        # ensure numpy
        ids_flat = np.array(ids_array).reshape(-1)
        for i, marker_id in enumerate(ids_flat):
            # corners_list[i] may be shape (1,4,2) or (4,1,2) depending on version
            c = np.array(corners_list[i])
            # normalize shape to (4,2)
            pts = c.reshape(-1, 2)
            cx, cy = pts.mean(axis=0)
            centers[int(marker_id)] = (float(cx), float(cy))
        return centers

    def compute_homography_from_centers(self, centers):
        """
        centers: dict id->(cx,cy)
        Computes homography from image points -> real-world mm coordinates
        """
        # Ensure all ids present
        for id_needed in self.ids_needed:
            if id_needed not in centers:
                return False

        img_pts = []
        world_pts = []
        # Use the order of ids_needed so mapping is consistent
        for mid in self.ids_needed:
            img_pts.append(centers[mid])
            world_pts.append(self.WORLD_POINTS_MM[mid])

        img_pts = np.array(img_pts, dtype=np.float32)
        world_pts = np.array(world_pts, dtype=np.float32)

        # findHomography maps image -> world
        H, status = cv2.findHomography(img_pts, world_pts, method=0)
        if H is None:
            return False

        self.H = H
        try:
            self.H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.H_inv = None

        # save
        np.save("homography.npy", self.H)
        if self.H_inv is not None:
            np.save("homography_inv.npy", self.H_inv)
        print("✓ Homography computed and saved (homography.npy).")
        return True

    def draw_centers_and_ids(self, img, centers):
        disp = img.copy()
        for mid, (cx, cy) in centers.items():
            cv2.circle(disp, (int(cx), int(cy)), 6, (0, 0, 255), -1)
            cv2.putText(disp, f"ID {mid}", (int(cx) + 6, int(cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return disp

    def warp_to_bev(self, color_img):
        """
        Warps the input image to the BEV using self.H as image->world transform.
        Produces a BEV image where 1 pixel = mm_per_pixel mm.
        """
        if self.H is None:
            return None

        # Destination size in pixels (width, height)
        dst_size = (self.bev_w_px, self.bev_h_px)

        # Note: world_coords are in mm units; because our world points used mm values,
        # warping with H will place pixels into mm coordinate space. Since our dst image
        # has pixel units corresponding to mm_per_pixel mm, warpPerspective with this H
        # directly will produce the correct scaling if mm_per_pixel == 1. For other
        # mm_per_pixel values you would scale dst accordingly. We set mm_per_pixel = 1 by default.
        bev = cv2.warpPerspective(color_img, self.H, dst_size, flags=cv2.INTER_LINEAR)

        return bev

    def draw_metric_grid(self, bev_img, step_mm=50):
        """Draw grid lines every step_mm in the BEV (assumes 1px = 1mm)."""
        if bev_img is None:
            return None
        img = bev_img.copy()
        h, w = img.shape[:2]
        step = int(step_mm / self.mm_per_pixel)
        color = (200, 200, 200)
        for x in range(0, w, step):
            cv2.line(img, (x, 0), (x, h), color, 1)
        for y in range(0, h, step):
            cv2.line(img, (0, y), (w, y), color, 1)
        # draw axis labels (mm)
        for x in range(0, w, step):
            cv2.putText(img, f"{x * self.mm_per_pixel}mm", (x + 2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        for y in range(0, h, step):
            cv2.putText(img, f"{y * self.mm_per_pixel}mm", (2, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        return img

    def run(self):
        print("Starting. Press 'q' to quit, 's' to save homography (if present).")
        print("Homography will be computed automatically when all 4 markers are visible.\n")
        try:
            last_compute_time = 0
            compute_interval = 1.0  # seconds between automatic recompute attempts

            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                img = np.asanyarray(color_frame.get_data())

                # Detect markers
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict,
                                                                 parameters=self.aruco_params)

                centers = {}
                if ids is not None and len(ids) > 0:
                    centers = self.get_marker_centers(corners, ids)

                # draw detection on input image
                display = img.copy()
                if centers:
                    display = self.draw_centers_and_ids(display, centers)
                else:
                    cv2.putText(display, "Looking for markers (1,2,3,4)...",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # If all markers visible, compute homography automatically (rate-limited)
                now = time.time()
                if all(mid in centers for mid in self.ids_needed) and (now - last_compute_time) > compute_interval:
                    ok = self.compute_homography_from_centers(centers)
                    last_compute_time = now
                    if not ok:
                        cv2.putText(display, "Homography failed", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # If we have H, produce BEV and put into black canvas
                bev = None
                if self.H is not None:
                    bev = self.warp_to_bev(img)
                    # draw metric grid and borders
                    bev_grid = self.draw_metric_grid(bev, step_mm=50)
                    # place onto black canvas
                    black = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
                    h, w = bev_grid.shape[:2]
                    x0 = self.canvas_offset_x
                    y0 = self.canvas_offset_y
                    black[y0:y0 + h, x0:x0 + w] = bev_grid
                    cv2.rectangle(black, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
                    cv2.putText(black, "Bird's-Eye View (1px=1mm)", (x0 + 6, y0 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Birds Eye View", black)

                # Show input
                cv2.imshow("Input", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s'):
                    if self.H is not None:
                        np.save("homography.npy", self.H)
                        if self.H_inv is not None:
                            np.save("homography_inv.npy", self.H_inv)
                        print("Saved homography.npy")
                    else:
                        print("No homography to save yet.")

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    bev = BirdsEyeView4Aruco(width=1280, height=800, fps=8, mm_per_pixel=1)
    bev.run()
