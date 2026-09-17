### Fixed: `render_video.py --camera default` collects pixels, not the PNG envelope

The example prefers a direct `mujoco.Renderer` with a body-tracking camera and
falls back to the engine's free camera - the path `--camera default` selects
outright, and the one any machine without an offscreen GL context takes. That
fallback read frames from `sim.render()`, which answers the agent-tool PNG
envelope (a `{"status", "content": [...]}` dict), so `frames` filled with dicts
and the first/last spread read killed the run before it encoded anything:
`TypeError: int() argument must be a string, a bytes-like object or a real
number, not 'dict'`.

It now reads `sim.get_frame()`, the raw-pixel counterpart the engine documents
for in-process consumers, at the `--width`/`--height` asked for. Same command
after the change: 400 `(480, 640, 3)` uint8 frames and an encoded MP4.
