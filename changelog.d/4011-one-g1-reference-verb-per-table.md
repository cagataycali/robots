### Changed: the G1's nine reference lookups are four verbs, one per table

`g1_joint_reference` / `g1_joint_name` / `g1_joint_index`,
`g1_list_motion_gates` / `g1_fsm_admits`, `g1_list_arm_actions` /
`g1_arm_action_admits` and `g1_list_error_codes` / `g1_decode_error_code` each
published a table's catalogue and its membership question as separate verbs.
`g1_joints`, `g1_motion_gates`, `g1_arm_actions` and `g1_error_codes` in
`strands_robots.tools.g1.g1_reference` answer both under one rule - no query
lists the table, a query resolves one entry in it - with each mode's payload
unchanged. A caller who supplied an arm action by both name and id used to
receive a refusal; the query is now one union-typed argument, so that
ambiguity cannot be expressed. `tools/` publishes 56 verbs over 26 files, down
from 61 over 29.
