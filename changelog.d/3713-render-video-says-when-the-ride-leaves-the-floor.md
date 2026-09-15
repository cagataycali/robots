### Fixed: `render_video.py` says when a ride runs off the rendered floor

The world's `ground` plane is drawn to 5 m from the origin while colliding
without limit, so a robot that rolls past the edge keeps rolling on a floor the
frame no longer shows. The example's roller recipe (`--vx 0.3 --duration 8`)
rode to the lip of it - the roller covers about 0.7 m/s of ground at that
command, so 8 s finishes about 4.9 m out, a tenth of a metre inside the edge on
one node and past it on a faster one - and nothing said so either way. The
recipe now runs 6 s (finishing 3.68 m out, on the checkerboard), and after any
rollout the example reports the second the duck left the drawn floor, the ground
it covered, and what to change (`--duration` or `--vx`).
