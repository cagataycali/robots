### Changed: `examples/microduck/render_video.py` writes through `encode_clip`

The showcase example encoded its MP4 and GIF with a private `imageio.mimwrite`
call while `run_policy(video=...)` and every recorder in the package write
through `strands_robots.rendering.encode_clip`. The example now uses the same
encoder (libx264, yuv420p, quality 8, exact frame size), so its clip and a
library clip are the same bytes for the same frames, and the encoder's domain
checks and missing-plugin refusal reach it. `--fps` and `--gif-fps` take whole
numbers, which is what an MP4/GIF header stores.
