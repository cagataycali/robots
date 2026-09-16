### Docs: `docs/policies/wbc.md` says how to record a walking G1

The "In simulation" snippet handed `run_policy(video=...)` readers to the
scene's `default` camera, whose pose is a fixed function of the compiled model:
the pelvis of a G1 walking at 0.4 m/s crosses the right edge of the 640x480
default view after 1.5 m, under 4 s, and the rest of the clip is empty floor. A
new "Recording it" subsection shows `add_camera` mounted on
`unitree_g1/pelvis` (in the pelvis frame) and named in `video`, points at the
shipped examples that mount a camera on a walking robot and that place a fixed
one, and notes that the camera must be added before the rollout.
