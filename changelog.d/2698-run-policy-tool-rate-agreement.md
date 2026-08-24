### Fixed: the `run_policy` tool settles its own two rates before it opens a recording

The `run_policy` tool owns both halves of one rule. `dataset_fps` becomes the LeRobotDataset's
declared frame rate; `control_frequency` is the rate the recorder is really driven at, one frame
per control step with no decimation. LeRobot derives every timestamp positionally from the
declared rate, so the two must be EQUAL - a differing pair cannot be honored, only mislabelled.

Each rate was already checked on its own domain: `control_frequency` by the tool's pre-flight,
`dataset_fps` by `start_recording` before it touches the target directory. Their *equality* was
left to the rollout entry point, which this tool reaches only inside the episode loop - after
`start_recording(overwrite=True)` has replaced whatever was at `dataset_root`. Measured over a
real MuJoCo rollout that had recorded one episode of five frames, `dataset_fps=30` with
`control_frequency=50.0` took `meta/info.json` from `total_episodes=1, total_frames=5` to
`0, 0`, removed the per-camera MP4, and reported `run_policy: 0/2 episodes ok`. Neither argument
was wrong on its own, so nothing refused the pair; the caller lost the dataset and recorded
nothing in its place. The tool states this principle for every other knob it forwards - `seed`
("reached NumPy inside the loop after step 2 had already created a dataset"), `video`, the
provider keyword bags and `stop_when` - and the per-knob sweep that pins them could not see this
one, because it derives the knobs forwarded to `Simulation.run_policy(...)` and a rule between
two parameters is not a property of either.

This is the third ordering of a disagreement whose other two are already refused: a rollout
started against an open recording (`dataset_rate_mismatch_reason`) and a recording opened against
a rollout in flight (`rollout_rate_mismatch_reason`). Each of those reads one rate off live state,
so neither can be asked before either exists. The new `requested_rate_mismatch_reason` covers the
case where both rates arrive as arguments to one call, and it reuses `rate_mismatch_explanation`,
so a caller who reorders two calls or supplies both at once gets one account of the distortion
rather than three. All three orderings are now held to the same verdict by a parity test.

Because the new guard runs *before* either rate has been through its own domain, its accepted set
is derived from those two domains rather than guessed: both rates are classified on
`numbers.Real`, exactly as `positive_whole_number_error` and `positive_finite_number_error` do, so
the `numpy.int64` / `numpy.float32` spellings a value read out of a config carries are judged too -
an `isinstance(int | float)` narrowing would have passed a colliding pair straight through. The
boolean question goes to the shared `is_boolean` predicate, since `bool` is a `numbers.Real` and
would otherwise be diagnosed as a genuine 1 Hz rollout. A value either domain refuses returns no
reason here, so it is still reported as the parameter error it is rather than as a rate
disagreement.

The check is gated on a requested recording: `dataset_fps` is forwarded nowhere when
`dataset_root` is `None`, so the documented recording-less smoke-test mode still runs at any rate.
