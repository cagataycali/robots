### Added: `wbc_latent` - a VLA's SONIC motion tokens decoded into Unitree G1 joint targets

The recipe in "Bringing Humanoids to LeRobot" fine-tunes pi0.5 to predict 64
SONIC latent motion tokens plus two gripper commands instead of joint angles;
on the robot NVIDIA's SONIC decoder turns each token and the last ten frames of
proprioception into 29 joint targets tracked by a per-joint PD law. Nothing in
strands consumed a latent action space, so that checkpoint wrote 31 of its 66
values onto 29 joints as radians.

`WBCLatentPolicy` (`create_policy("wbc_latent", inner=...)`) wraps the
token-emitting VLA and one `SonicDecoder` (`nvidia/GEAR-SONIC`
`model_decoder.onnx`, one file fetched into the HF cache, never bundled; the
NVIDIA Open Model License notice is logged once). It runs once per 50 Hz tick,
re-queries the VLA every `replan_every` ticks (20, the 2.5 Hz of NVIDIA's
client), and returns 29 joint targets plus `left_gripper`/`right_gripper`.
`variant` selects the decoder (`default`, `low_latency`, `sonic_v1_1`): tokens
decode correctly only through the decoder of the encoder that produced them.
The MuJoCo engine installs `WBCLatentTorqueController` for it, SONIC's
armature-derived PD gains on all 29 joints at the decoder's 0.005 s x 4 clock.
The lerobot_local embodiment `unitree_g1_sonic` names the 66 dataset actions so
`align_action_values` keeps every token instead of truncating to the 31 state
keys. Docs page `learn/policies/wbc-latent.md`; the site word ceiling rises
once from 46,418 to 47,206 for it.
