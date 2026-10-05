### Fixed: a camera rate the device accepts but never delivers keeps the camera, and is no longer offered

Some UVC cameras (a Sonix USB2.0 camera on macOS AVFoundation) report their idle
rate, 5 fps, accept `CAP_PROP_FPS=5`, and then return no frame. The dashboard's
mode probe offered that idle rate as a native mode, and `OpenCVCamera.open`
gave up after one `read()` with "another process may hold the device", so an
arm spawned with the advertised mode connected without its camera.
`OpenCVCamera.open` now retries without the rate, then without the size, keeps
the first attempt that delivers a frame and names the refused mode in
`describe()["refused_mode"]`; when nothing delivers, the error names the mode it
tried. `DeviceManager.probe_modes` offers a mode only after a frame arrived at
it, and the Devices scan no longer publishes an un-configured capture's idle
rate as the camera's fps.
