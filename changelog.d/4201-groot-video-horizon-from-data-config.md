### Fixed: a GR00T data config's video horizon is sent, not just declared

`Gr00tDataConfig.observation_indices` is the client's copy of the server's
`video.delta_indices`, which an N1.7 server checks every video tensor's time
axis against. Nothing read it: a single frame always went out as `T=1`, so the
shipped `unitree_g1_real` config (`[-20, 0]`, the base model's
`real_g1_relative_eef_relative_joints` pretrain tag) failed its first request
with `Video key 'ego_view's horizon must be 2. Got 1`. The policy now keeps a
per-camera frame history, repeats the first frame until the horizon is full,
clears it on `reset`, and passes a caller-stacked `(T, H, W, C)` value through.
`oxe_droid_relative_eef_relative_joint` now declares the `[-15, 0]` horizon its
pretrain tag carries.
