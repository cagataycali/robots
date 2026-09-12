### Fixed: the sim free view no longer takes the policy image slot a real camera needs

Both camera routers on `LerobotLocalPolicy` end in a positional fallback, and its
candidate order was the observation's order. `create_world` registers the
built-in free view under `"default"` before any `add_camera` call, so it leads
`get_observation()` - and when no camera matched a declared key by name it took
slot 0. Measured on a real MuJoCo SO-101 scene with two cameras the caller added
and a policy declaring `observation.images.front` / `observation.images.wrist`:

    observation image keys: ['default', 'cam_a', 'cam_b']

    before: default -> observation.images.front    # 3/4 debug view of the scene
            cam_a   -> observation.images.wrist
            cam_b   -> (dropped - no slot left)

    now:    cam_a   -> observation.images.front
            cam_b   -> observation.images.wrist
            default -> (dropped - the extra camera)

So the policy read a fixed three-quarter view of the whole scene, at the world
default resolution, in the slot its checkpoint was trained to read a task view
in - and paid for it with one of the caller's real cameras. That is the outcome
the exact-name rung of the sibling router already names in its own comment as the
reason it binds by name ("drops a real one when an extra free camera such as the
sim `default` is present"); it stayed reachable through the fallback rung
whenever no camera matched by name. `add_camera` already refuses to let a caller
*create* a camera under one of these names
(`reserved_camera_name_error`), so a view registered under one is never a task
view a caller asked to route.

`free_camera_routing_rank` ranks the `FREE_CAMERA_TOKENS` names behind every
other camera, and both routers sort their fallback candidates by it - the
preprocessor/VLA path too, since the only difference between the two is whether
the checkpoint ships a preprocessor, which is not something a camera binding may
depend on. The free view is ranked last, not dropped: a scene whose only camera
is the free view still fills the slot it filled before. Real cameras keep their
relative order among themselves, an exact name match still wins, and an explicit
`camera_key_map` still outranks both, so the ordering decides only which camera a
guess picks. The fallback stays loud (`positional_fallback_used` plus the
per-camera WARN) and `strict_keys=True` still refuses it outright.
