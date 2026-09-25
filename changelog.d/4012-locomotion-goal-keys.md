### Docs: the locomotion examples name only `policy_kwargs` goal keys a policy reads

The three `examples/locomotion/*_g1.py` scripts steered the Unitree G1 through a
`locomotion_style` goal key. Its only consumer, the MotionBricks policy, was
removed with the rest of the policy tree-shake; no policy in the tree reads the
key today, and `get_actions(obs, instruction, **policy_kwargs)` drops an unknown
one in silence - so nothing refused it, warned about it, or recorded it.

Measured with the published GR00T-WBC Balance/Walk ONNX on the G1 in MuJoCo, the
`scripted_g1.py` four-segment schedule run twice - once as shipped, once with a
style added to every segment - traced 408 control steps at 50 Hz whose
`(x, y, z, yaw)` differed by `0.000000`, while `target_velocity` walked the pelvis
+1.788 m and veered it +56.2 deg. The keys WBC does read move the action: `height`
by 0.0212 rad and `target_orientation` by 0.0152 rad on one inference step.

`keyboard_g1.py` bound eight keys to the dead field and advertised them in its own
bindings line; `agent_g1.py` named it plus a seven-value style vocabulary in the
system prompt an LLM is handed, so the agent reported style switches it never
made. All three now name only `target_velocity`, `height` and
`target_orientation`, and the run-loop docstrings in `simulation/base.py` and
`simulation/policy_runner.py` - which already listed the live keys correctly - are
unchanged.

`tests/test_examples_document_the_interface_they_have.py` grows the rule that
keeps them honest: every goal key an example spells, whether in a dict literal, in
a `policy_kwargs={...}` fragment inside a string, or in a docstring's
slash-separated channel list, must appear in the `kwargs.get()` / `kwargs.pop()`
calls harvested from `strands_robots/policies/`. Citations are recognised by a
`target_*` anchor, because `height` / `command` / `video` / `seed` are read by a
policy and also spelled by unrelated camera configs and JSON-RPC envelopes.
