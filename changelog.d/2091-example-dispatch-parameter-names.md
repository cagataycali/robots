### Fixed: the sim VLA example passed parameter names the dispatch router refuses, and discarded the refusal

`examples/vla/molmoact2_sim_pickplace.py` passed `camera_name` to `add_camera`
(whose parameter is `name`) and to `get_observation` (which takes `robot_name` /
`skip_images`). `camera_name` is the render-side spelling -- correct for
`render`, `render_depth`, `get_frame`, `get_camera_params` and `get_world_point`
-- so both mistakes read plausible.

Neither call checked its result. The router reports an unknown parameter by
RETURNING `{"status": "error", ...}` rather than raising, so the example ran on
with only the `default` camera: the policy was handed an observation without the
`front` view it was trained on and failed much later complaining about missing
image keys, pointing at the policy rather than at the call that was refused.
Both parameters are corrected and every dispatch in the example now goes through
a `_must` helper that raises on an error envelope, matching
`examples/lerobot/collect_train_run_molmoact2.py`.

A new guard statically scans every example for calls written with a literal
action name and a literal parameter dict and checks each name against the
router's own acceptance rule, derived from the live engine class so it tracks
the router rather than drifting from it. It matches by call shape rather than by
callee name, so an example that wraps the router in a local `_must` helper is
still covered.
