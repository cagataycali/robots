### Fixed: Newton's recorded action schema is declared from `robot_action_keys`

`DatasetRecorder` names the `action` feature's columns from the `action_names` it
is handed, and falls back to `joint_names` when a backend passes none. Newton was
the one backend on that fallback: Isaac and MuJoCo both declare `action_names`
from `robot_action_keys`, Newton passed none, so its action columns were named in
the joint vocabulary while its recording hook emits the actuator vocabulary
`robot_action_keys` defines.

The two vocabularies agreed, so no recording was miscolumned. They agreed only
because `_collect_recording_schema` and `NewtonSimEngine.robot_action_keys` each
excluded the floating base's 6-DoF free joint from its own copy of the rule, and
the fallback made that agreement load-bearing. An existing pin said as much in its
own docstring - it asserted the equality "rather than assumed from the fact that
Newton's schema fallback happens to be the scalar joint list".

Either copy drifting would have failed silently. `add_frame` reads the action dict
by declared name, so a column declared under a name the hook never emits is not a
mismatch the recorder can report: it takes the `0.0` fill and the frame records a
command nobody issued under `status="success"`. Every action space reaching a
recorded rollout is absolute position, so `PolicyRunner.replay` re-sends those
zeros to the robot as travel-to-zero targets. That is the fabrication reported in
#1715, reached with one robot and no narrow policy.

`_collect_recording_schema` now returns `action_names` alongside `joint_names` -
mirroring Isaac's 6-tuple and ordering - built from `robot_action_keys` with the
same `robot__` prefixing the recording hook applies, and `start_recording` passes
it to `create`. No recorded column changes for any robot this backend can build
today; what changes is that the naming no longer depends on two copies of an
exclusion rule staying in step. An AST guard asserts every backend's
`_DatasetRecorder.create` passes `action_names`, so no backend can re-enter the
fallback silently.
