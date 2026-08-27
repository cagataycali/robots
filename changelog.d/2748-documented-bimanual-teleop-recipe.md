### Fixed: the documented bimanual teleop recipe runs as written

`docs/hardware/teleoperation.md` carried the only bimanual real-hardware recipe in the
documentation, and none of its three lines could run. It opened with
`Robot("bi_so", mode="real", left_port=..., right_port=...)`, which fails twice over and
independently: `bi_so` is neither a canonical registry name nor an alias, so `Robot()` refuses it
with `Unknown robot 'bi_so'` before anything is built; and `left_port`/`right_port` are accepted by
no robot in the registry. A bimanual device is two arms, so `BiSOFollowerConfig` declares a
`left_arm_config`/`right_arm_config` pair of per-arm config objects and no single `port` at all.
Supplying the registered spelling therefore only moves the reader on to the second error,
`Unknown kwarg(s) for robot_type='bi_so_follower': ['left_port', 'right_port']`. The
`attach_teleop("bi_so_leader", left_port=..., right_port=...)` line beneath it fails the same way
for the same reason: `BiSOLeaderConfig` requires the same per-arm pair.

The recipe is corrected to the form that was checked by running it - both arms constructed from
`SOFollowerConfig`, both leaders from `SOLeaderConfig` - and gains a sentence saying why a bimanual
device has no single `port`, which is the fact the old text got wrong.

Nothing in the repository could have noticed. `tests/test_docs_python_examples_are_callable.py`
already grades documented keywords, but it grades them against *signatures*, and its
`_accepted_keywords` deliberately returns "any keyword binds" for a callee carrying `**kwargs`.
`Robot()` ends in `**kwargs: Any`, so every keyword to every documented `Robot(...)` call was
outside what that module can see - correctly, for the question it asks. Neither does anything grade
the robot *name*. The block itself arrived in one 240-line documentation commit (`1c434330`, #632)
and was never executed.

`tests/test_docs_real_mode_invocations.py` closes both halves by grading what the runtime actually
enforces rather than what the signature admits. `mode="real"` resolves the robot's
`hardware.lerobot_type` to a lerobot config dataclass, and a keyword is accepted only when that
dataclass declares it or it appears in the cross-robot forwarding allowlist - so the accepted set is
a property of the named robot, not of the factory, and it is derived per call. The two halves are
split by dependency: the name rule needs only the package registry and always runs, while the
keyword rule needs lerobot to resolve the config and skips without it, so the more important half is
never silently skipped. The owned-keyword set is derived from the two entry points' own signatures
plus the allowlist rather than listed, so a parameter added to either is covered without editing the
test.

Across `docs/**/*.md` and `README.md` the sweep grades 25 documented `mode="real"` calls over seven
robots, of which this was the one defect; the other 24 were already correct, including every
`robot_ip` and `port` spelling.

A block that documents a refusal is excluded rather than reported. The section directly above
teaches that `Robot()` rejects every `*_leader` name and prints the `ValueError` as its own output,
so grading it would report the documentation's teaching point as a defect; such a block is
recognised by a comment line naming an exception type, and `so101_leader` is pinned as absent from
the graded set.

Both rules are additionally graded on constructed exemplars, because after this fix the
documentation corpus is clean and can no longer exercise a rejection - a sweep that only ever sees
valid input cannot distinguish a working rule from one that accepts everything. The old text is the
flagged exemplar in each case, alongside its corrected form and a single-`port` robot, with a
non-vacuity check that both verdicts still occur. Each rule lives in one predicate shared by the
sweep and the exemplars, so the two cannot drift apart.

Nine plausible regressions were applied: the control is clean, all nine are detected, and they fire
eight distinct sets. Reverting the documentation fires the name half; half-fixing it - correct name,
old `*_port` keywords - fires the keyword half instead, which is what shows the two are independent.
