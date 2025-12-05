import pyrealsense2 as rs

def list_supported_modes():
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("❌ No RealSense devices found!")
        return
    
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"\n=== Device: {name} (S/N: {serial}) ===")

        sensors = dev.query_sensors()
        for sensor in sensors:
            sensor_name = sensor.get_info(rs.camera_info.name)
            print(f"\n  --- Sensor: {sensor_name} ---")

            # Go through all stream profiles the sensor supports
            for profile in sensor.get_stream_profiles():
                vprofile = profile.as_video_stream_profile()
                stream_type = vprofile.stream_type()
                fmt = vprofile.format()
                width = vprofile.width()
                height = vprofile.height()
                fps = vprofile.fps()

                print(f"  Stream: {stream_type}, Format: {fmt}, "
                      f"{width}x{height} @ {fps} FPS")

if __name__ == "__main__":
    list_supported_modes()
