### Docs: `docs/policies/wbc.md` says how to record a walking G1

The "In simulation" snippet handed `run_policy(video=...)` readers to the
scene's `default` camera, which frames the origin from a fixed vantage and
loses a G1 walking at 0.4 m/s within a couple of seconds - the rest of the clip
is empty floor. A new "Recording it" subsection shows `add_camera` mounted on
`unitree_g1/pelvis` (a follow camera, in the pelvis frame) and names it in
`video`, records the fixed side camera the page's own clip was shot from, and
notes that the camera must be added before the rollout.
