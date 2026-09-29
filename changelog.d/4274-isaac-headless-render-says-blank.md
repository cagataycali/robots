### Fixed: Isaac's headless render mode says its frames are blank

`render_mode="headless"`, the default, renders no pixels, but cameras returned all-zero frames with `status: success`. The render envelope now marks the frame `blank_frame` with the remedy, `add_camera` warns in that mode, and the docs sketch uses `render_mode="rtx_realtime"`.
