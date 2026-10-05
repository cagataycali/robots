### Fixed: send_action's unresolved-key WARNING matches the whole-batch refusal

Since #4486 (b0fb474) `SimEngine.send_action` refuses a whole batch when any
key fails to resolve to an actuator or joint: nothing is written and the world
does not advance. The return envelope says so correctly ("Nothing was applied
and the world did not advance."), but the WARNING log emitted on the same
pre-write refusal path still described the pre-#4486 silent-partial behaviour:

```text
[sim] action key 'nonsense_joint' (prefix='so101/') could not be applied:
 no actuator or joint. The value was dropped. Valid keys for this robot: ...
```

"The value was dropped" is a singular-key wording that implies the batch
continued and only the typo'd key was skipped - the exact mental model
`#4486` was filed to retire. An operator who trusted the WARNING (over the
return envelope that said the opposite) believed the valid keys in the batch
had landed, skipped the retry, and watched the arm stay still while they
debugged a controller output that never reached `ctrl`.

The warning helper now carries a `batch_refused` flag that tracks the two
call sites' opposite outcomes:

* `send_action`'s pre-write resolver (`simulation.py`) refuses the batch
  whole, and the warning now reads *"the whole batch was refused and nothing
  was written"* - matching the envelope.
* The action-controller fallback in `_apply_action_dict`
  (`rendering.py`) writes keys one-by-one, so a single unresolved key really
  is dropped while sibling valid keys land; the warning keeps the
  *"the value was dropped"* wording there, matching the `applied` list the
  envelope carries.

Fix at `strands_robots/simulation/mujoco/rendering.py:1114-1172` and the two
call sites at `rendering.py:1028` and `simulation.py:1346`. Repro at
`bugbash_repros/send_action_warn_text_stale_repro.py`.
