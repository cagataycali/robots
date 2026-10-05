### Fixed: the dashboard names each Mac camera at the index OpenCV really opens

On macOS the Devices sheet paired ffmpeg's AVFoundation listing with OpenCV indices by
position, but OpenCV numbers cameras by sorted device id, so a Mac with a built-in and a
USB camera showed the two names swapped. Names now come from `system_profiler
SPCameraDataType` and are numbered in OpenCV's own order; ffmpeg is no longer needed for
them. If any camera lacks an id, no names are shown rather than wrong ones.
