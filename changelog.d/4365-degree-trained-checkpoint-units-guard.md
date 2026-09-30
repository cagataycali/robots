### Fixed: a degree-trained SO-arm checkpoint no longer drives a simulated arm into its joint limits

`run_policy(policy_provider="lerobot_local", policy_config={"pretrained_name_or_path": ...})`
with no `embodiment` packed a simulator's radian state straight into a checkpoint
whose stats were recorded in degrees (LeRobot's SO-arm driver default), and applied
the model's degree actions as radians. On Isaac a pi0.5 SO-101 fine-tune held joint 4
at its 1.658 rad limit and joint 5 at -2.793 rad for every frame of an agent-recorded
episode, and the rollout reported success. `LerobotLocalPolicy.get_actions` now reads
the units off the checkpoint's own stats before the first inference: when two or more
(and at least half) of the state columns span more than 2*pi, the stats are degrees.
A state keyed exactly like a shipped simulation embodiment that converts units
(`so101`: joints `1`..`6`, `so100`) then gets that embodiment applied, as if the caller
had named it, with the camera routing the model's declared features ask for;
`embodiment_adopted` names it and a warning says so. Anything else that would reach
the model in radians - a declared native map on those joints, or unrecognised keys
whose values all sit within one turn of zero - is refused with a `ValueError` naming
the stats, the state and the embodiment that converts, before any action is emitted.
A real arm through a LeRobot driver (`'<motor>.pos'` keys, the dataset's own units)
and radian or absent stats are untouched. `ProcessorBridge.recorded_value_ranges`
and `embodiment.degree_like_columns` / `registered_sim_embodiment` expose the
evidence the guard reads.
