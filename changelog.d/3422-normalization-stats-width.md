### Fixed: `lerobot_local` refuses override stats of the wrong width where both widths are known

`ProcessorBridge.inert_normalization_features` reports stats that are ABSENT,
which LeRobot answers by returning the tensor unchanged. Stats that are PRESENT
at the wrong width are the opposite failure: LeRobot reaches the arithmetic and
raises `RuntimeError: The size of tensor a (6) must match the size of tensor b
(7) at non-singleton dimension 0` from the broadcast, naming neither the
feature, the step, nor either width -- and it raises on the first inference,
after a rollout has started and the robot has been commanded. `create_policy`
returned a policy whose `inert_normalization_features()` was `[]`, so nothing
reported a problem until the arm was already moving.

Both widths are known when the policy loads, on the same step object: the
step's `features` declares `observation.state (6,)` while its `_tensor_stats`
holds `(7,)`. A new `mismatched_normalization_widths` reads that pair and the
load refuses, naming each affected feature, the declared width and the
supplied width. Supplying stats is exactly what the inert-pipeline diagnostic
tells a caller to do, and width is what differs between embodiments -- a 6-DOF
SO-101, a 7-DOF arm and a 14-DOF bimanual all declare `observation.state` and
`action` -- so reaching for the wrong dataset's stats is a routine mistake.

The scoping rule that decides which declared normalization a transition
actually exercises (the preprocessor transition carries only the observation,
the postprocessor only the action, so only the feature type matching a
pipeline's position is ever touched) was owned by the inert detector. It is now
spelled once in `_declared_normalization_targets` and read by both checks, so
the two cannot disagree about which normalization a rollout performs -- a
preprocessor's `action` entry is never exercised at inference and is therefore
flagged by neither.

VISUAL features are exempt. LeRobot reshapes a flat `(C,)` visual stat to
`(C, 1, 1)` in `_reshape_visual_stats` on purpose, so a channel-wide stat is
correct for a `(C, H, W)` feature and comparing it to the flattened feature
width would refuse a working checkpoint.
