from pupil_labs.realtime_api.simple import Device

try:
    # Look for a device on the network.
    print("Looking for a device...")
    ip = "172.20.116.207"
    device = Device(address=ip, port="8080")
    if device is None:
        print("No device found.")
        raise SystemExit()

    print(f"Connected to {device.serial_number_glasses}. Press Ctrl-C to stop.")

    # Stream gaze data.
    while True:
        # receive_gaze_datum() will return the next available gaze datum
        # or block until one becomes available.
        gaze = device.receive_gaze_datum()
        a = device. 
        # The gaze datum is a named tuple containing x, y, worn, and timestamp.
        # We can access these values as attributes.
        print(
            f"Timestamp: {gaze.timestamp_unix_seconds:.3f} | "
            f"Gaze (x,y): ({gaze.x:.2f}, {gaze.y:.2f}) | "
            f"Worn: {gaze.worn}"
        )

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    # Cleanly close the connection
    if "device" in locals() and device:
        device.close()
    print("Connection closed.")