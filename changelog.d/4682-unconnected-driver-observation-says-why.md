### Fixed: a native driver that is not connected says why its observation is empty

`get_observation()` on a native driver that had not connected returned `{}`
with no word, while `state`, `send_action` and `run_policy` on the same driver
refused with "not connected - call connect_eagerly() first". So
`Robot("rby1", mode="real", port=...).get_observation()` read like a robot with
no joints. It still returns `{}` (the read does not raise, as on the simulation
engines), and now logs a WARNING naming the cause: the last `connect_eagerly()`
refusal when there is one - a missing SDK names its install line - otherwise
the call that connects. Applies to the Booster, Franka, Kinova, KUKA,
Microduck, RB-Y1, Robotiq, Spot, Stretch, UR, xArm and Yahboom M3 Pro drivers.
A Microduck, UR or xArm whose link dropped no longer hands back the joints
cached from before.
