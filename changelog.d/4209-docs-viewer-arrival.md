### Docs: the robot viewer arrives instead of popping in

While a robot's meshes stream, its thumbnail is now revealed from the floor up in
step with the download under a one-line caption, in place of the status card.
On the first load the camera starts a fifth farther out and eases in over 900 ms,
the joints wake into the rest pose one after another over 1.2 s (a model with a
keyframe wakes from its default pose, so a humanoid settles into `stand`; one
without unfolds from a fifth of the way to each joint's lower limit), the
environment sheen fades up over 400 ms rather than landing a frame late, and the
stage orbits slowly until the first pointer, wheel or key. The code panel flashes
the `robot.act` line a slider just changed, and a mono clock reads the simulated
time while Physics is on. Compile and Reset both land on the model's first
keyframe when it has one. Every motion is skipped under `prefers-reduced-motion`,
which the stage now honours inside its own shadow root, pinned by
`tests/test_docs_viewer_arrival.py` (#4209).
