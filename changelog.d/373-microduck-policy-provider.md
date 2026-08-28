### Added: `MicroduckPolicy` - Pollen Microduck locomotion policies as a native provider

The Microduck (Pollen Robotics' open 14-DOF biped) ships a family of ONNX
skills (`alpha_walking`, `alpha_stand`, `roulade`, `ball_kick_*`, `roller*`,
`alpha_ground_pick`), each an actor with the input normaliser fused into the
exported graph. `MicroduckPolicy` adapts one such export to the `Policy`
interface so it runs through the standard `Robot(...).run_policy` seam in
MuJoCo or on hardware.

The provider is almost configuration-free: it reads `joint_names`,
`default_joint_pos`, `action_scale` and `command_names` from the ONNX
`custom_metadata_map` on first inference (explicit constructor arguments win),
feeds the observation RAW (never re-normalising - normalisation is baked into
the graph), and decodes `motor_target = DEFAULT_POSE + action * action_scale`,
tracking the RAW action as the next tick's `last_action` block exactly as
Pollen's reference deployment does. The command vector width is parameterised
by `command_names` (13-D unified `twist + head_pose + body_pose`, or legacy
3-D twist), and unused command slots stay present and zero (the dead-weight
rule). `MicroduckPolicyBundle` holds several skills warm and hot-swaps the
active policy mid-rollout, optionally gating walk<->stand by twist magnitude.

`reset` restores the episode's starting command rather than clearing it. The
command is built once, inside the metadata-driven `_ensure_config` pass that
early-returns for the rest of the policy's life, and `get_actions` has no lazy
rebuild for it the way it re-zeroes `last_action` - so clearing it left an
already-configured policy with no command for the next tick. `PolicyRunner`
forwards `reset(seed=...)` before a seeded rollout and again per episode of a
multi-episode eval, so reusing one policy across two seeded rollouts, or
reaching episode 2 of a seeded eval, produced no valid action; a bundle
amplified it to every skill that had already run. Both the first episode and
every later one now read one helper, which returns a copy, so a tick's
`target_velocity` cannot become the command later episodes restore to.

Verified against Pollen's shipped `alpha_walking.onnx`: byte-compatible with a
raw onnxruntime session to 0.0 max abs delta on an identical 61-D observation,
and a real MuJoCo rollout moves the joints. Registered under the `microduck`
and `microduck_bundle` provider shorthands.
