### Fixed: an Isaac recording refused in headless render mode names the real reason

`start_recording` under `render_mode="headless"` told the caller the cameras they had just added were unknown. It now says the cameras exist but record no frames in that mode and names `render_mode="rtx_realtime"`; the GPU recording suite now passes on Isaac Sim 6.1.
