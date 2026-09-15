### Fixed: `render_video.py` says when a ride runs off the rendered floor

The world's `ground` plane is drawn to 5 m from the origin while colliding
without limit, so a robot that rolls past the edge keeps rolling on a floor the
frame no longer shows. The example's roller recipe (`--vx 0.3 --duration 8`)
did that - the roller covers about 0.7 m/s at that command, crossed the edge at
7.5 s and finished at 5.41 m - and nothing said so. The recipe now runs 6 s
(finishing near 3.7 m, on the checkerboard), and after any rollout the example
reports the second the duck left the drawn floor, how far it travelled, and
what to change (`--duration` or `--vx`).
