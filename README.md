## 📷 Multi-Camera Calibration & Bird’s-Eye View System

This repository contains scripts for generating calibration patterns, performing extrinsic calibration across multiple cameras, and producing a real-time bird’s-eye view (BEV) with optional WebRTC streaming.

---

##  Scripts Overview

###  Calibration Utilities

**`generate_mask.py`**  
Generates chessboard-pattern masks for all camera views:
- Front  
- Back  
- Left  
- Right  

These masks are used for calibration and alignment.

---

###  Extrinsic Calibration

**`calibrate_bev_zed_xone.py`**  
Performs extrinsic calibration for **4 ZED cameras** using the four corners of a chessboard pattern.  
- Supports both **manual** and **automatic** corner selection.

**`extrinsic_calibration.py`**  
Handles extrinsic calibration for a mixed setup:
- **3 ZED cameras**
- **1 USB fisheye camera**

---

###  Bird’s-Eye View (BEV)

**`result_remap.py`**  
- Combines inputs from all four cameras  
- Generates and displays the final **bird’s-eye view**

---

###  Streaming

**`web_rtc_stream.py`**  
Streams the generated BEV output over **WebRTC**, enabling real-time remote viewing.

---

###  Experimental / Test Scripts

**`combined.py`**  
Test script for combining:
- USB fisheye camera  
- Ethernet camera  

**`combined_web_rtc.py`**  
Similar to `combined.py`, but includes **WebRTC streaming support**.

---
