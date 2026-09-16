### Fixed: a failed real-robot connect names the device that failed and its remedy

An unplugged arm now reads "the motors bus did not open on port '…': check
the USB cable and power, then find the port"; a camera that does not open is
named by its `cameras=` key ("camera 'wrist' did not open … fix or remove that
entry") for every camera backend lerobot registers - OpenCV, RealSense, ZMQ,
Reachy 2 - because the key is matched against the camera object whose name the
message carries. Before, every cause carried the same suffix - "Ensure robot is
calibrated and accessible on the specified port" - and an agent advised
recalibrating an arm that was not plugged in.
