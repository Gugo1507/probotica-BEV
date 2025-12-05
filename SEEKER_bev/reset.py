import pyrealsense2 as rs

ctx = rs.context()
for dev in ctx.query_devices():
    print("Resetting device:", dev.get_info(rs.camera_info.serial_number))
    dev.hardware_reset()
