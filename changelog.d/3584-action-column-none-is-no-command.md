### Fixed: an action column present as `None` is refused, not recorded as `0.0`

`DatasetRecorder.add_frame` refuses a frame whose action dict does not carry a
declared column it requires, because no placeholder for an un-issued command is
truthful -- `0.0` is itself a command on an absolute-position action space, so
`replay_episode` drives that joint to zero at servo speed.

`unrecordable_action_columns_error` asked whether the *key* was there
(`key not in action`). A required column present as `None` therefore passed the
guard, fell through to the `v is None` branch below it, and was written as
`0.0` -- under `status="success"`, in the same column and dtype as a real
command. That is precisely the fabrication the check exists to prevent, and it
was reached by the ordinary spellings of "no value": a policy that produced
nothing for one joint, a wire payload whose reading was `null`, an action dict
built by zipping column names against a shorter sequence of values.

The guard now reads the value (`action.get(key) is None`), which is what its
state sibling `unrecordable_state_columns_error` has always done -- so a state
column and an action column are graded the same way, and the action rule (the
stricter of the two, since a state column at least misstates a measurable
truth while no substitute for an un-issued command is truthful at all) is no
longer the softer one. Over a 60-step episode dropping one joint's command on 8
steps, the old guard recorded 8 zeros against a trajectory as far as 1.38 rad
away; the frame is now refused at the first of them, naming `a_elbow`.

The scoping is unchanged, and it is what keeps multi-robot recordings working:
columns *outside* an explicitly scoped `required_action_keys` -- in a shared
scene, the robots this rollout does not drive -- are still not this frame's to
supply however the frame spells their absence, and still take the documented
`0.0` fill. The inline comment at that fill asserted this invariant already
("every column this frame must supply was checked above"); it is now true.
