"""
generate_masks.py
=================
Clean white calibration masks, 1 px = 1 mm.
Front/back: 6×4 board (270×180 mm), gap = 270 mm
Left/right: rotated 90° → 4×6 board (180×270 mm), gap = 400 mm
"""
import cv2
import numpy as np
import os

SQ        = 45
COLS      = 6
ROWS      = 4
BOARD_W   = SQ * COLS   # 270 mm  (front/back orientation)
BOARD_H   = SQ * ROWS   # 180 mm

GAP_FB    = 270          # front/back: vehicle-edge → board
GAP_LR    = 270          # left/right: vehicle-edge → board

SHIFT     = 2000
VEHICLE_W = 490
VEHICLE_L = 740

# Left/right boards are rotated 90°: width=180, height=270
BW_LR = BOARD_H          # 180
BH_LR = BOARD_W          # 270

CANVAS_W = SHIFT + BW_LR + GAP_LR + VEHICLE_W + GAP_LR + BW_LR + SHIFT
CANVAS_H = SHIFT + BOARD_H + GAP_FB + VEHICLE_L + GAP_FB + BOARD_H + SHIFT
VEH_X    = SHIFT + BW_LR + GAP_LR
VEH_Y    = SHIFT + BOARD_H + GAP_FB


def draw_chessboard(canvas, bx, by, cols, rows):
    for r in range(rows):
        for c in range(cols):
            x0, y0 = bx + c * SQ, by + r * SQ
            colour  = (0, 0, 0) if (r + c) % 2 == 0 else (255, 255, 255)
            cv2.rectangle(canvas, (x0, y0), (x0 + SQ, y0 + SQ), colour, -1)
    cv2.rectangle(canvas, (bx, by),
                  (bx + cols * SQ, by + rows * SQ), (0, 0, 0), 1)


def make_mask(side, out_dir):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)

    if side in ("front", "back"):
        cols, rows = COLS, ROWS      # 6×4 → 270×180
        bw, bh     = BOARD_W, BOARD_H
        gap        = GAP_FB
    else:
        cols, rows = ROWS, COLS      # 4×6 → 180×270 (rotated)
        bw, bh     = BW_LR, BH_LR
        gap        = GAP_LR

    if side == "front":
        bx = VEH_X + (VEHICLE_W - bw) // 2
        by = VEH_Y - gap - bh
    elif side == "back":
        bx = VEH_X + (VEHICLE_W - bw) // 2
        by = VEH_Y + VEHICLE_L + gap
    elif side == "left":
        bx = VEH_X - gap - bw
        by = VEH_Y + (VEHICLE_L - bh) // 2
    else:  # right
        bx = VEH_X + VEHICLE_W + gap
        by = VEH_Y + (VEHICLE_L - bh) // 2

    draw_chessboard(canvas, bx, by, cols, rows)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"mask_wider_{side}.png")
    cv2.imwrite(path, canvas)
    print(f"  {side:5s}  board {cols}×{rows} ({bw}×{bh}mm)  gap={gap}mm  at ({bx},{by})  → {path}")
    return path


if __name__ == "__main__":
    out = "calibration_masks"
    print(f"Canvas {CANVAS_W}×{CANVAS_H} mm | 1px=1mm")
    print(f"GAP front/back={GAP_FB}mm  |  GAP left/right={GAP_LR}mm\n")
    for s in ("front", "back", "left", "right"):
        make_mask(s, out)
    print("\nDone.")