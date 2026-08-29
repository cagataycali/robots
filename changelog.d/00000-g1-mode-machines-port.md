### Feature

- Added `strands_robots.tools.g1.g1_list_arm_ready_mode_machines` and
  `strands_robots.tools.g1.g1_mode_machine_admits_arm`: pure-reference
  agent-facing lookups over the `mode_machine` ids the G1 driver's
  `_check_motion_gates` treats as arm-ready when the loco-SDK
  `GetFsmId` RPC is wedged (returns `rc=3104`) but the robot is
  physically arm-ready. Snapshotted from the neon bundle's
  `ARM_READY_MODE_MACHINES` observation (`cagataycali/neon-the-g1/tools/_g1_common.py`,
  set `{5, 6}` — the two hardware-layout ids the firmware publishes
  on `rt/lowstate` when the balance controller admits an arm write).
  Each descriptor carries a `mode_machine` id and an
  `admits_arm_writes` flag (always `True`; every listed id is
  arm-ready by construction) so the payload shape matches the
  `g1_fsm_targets` / `g1_arm_actions` / `g1_balance_modes` verbs
  verbatim. Unlike those verbs the refusal path does not carry an
  SDK `rc=` code: the driver's `mode_machine` refusal is a local
  liveness string (`"mode_machine unknown - lowstate has not
  delivered yet"`) it quotes before either write gate is consulted,
  and the lookup surfaces that same string on both a
  `mode_machine=None` liveness query and a non-arm-ready
  membership query so a caller sees a single, consistent refusal
  channel. No DDS is touched, no `unitree_sdk2py` submodule loads
  at import (the same SDK-load-hygiene rule every other file under
  `strands_robots.tools.g1` carries, refs strands-labs/robots#358);
  the verbs answer the arm-ready membership question that
  complements the SDK-side FSM lookup already shipped in
  `g1_fsm_targets`, so a caller reading the driver's
  `get_status` envelope can decide the arm-ready refusal decidably
  before dispatching a `send_action`. Contract-graded off the
  module's own snapshot (14 tests: import hygiene, snapshot value
  and typing, refusal string, list-verb envelope shape,
  fresh-container guarantee, admits-a-ready-value on both `5` and
  `6`, refuses-a-non-ready-value, `None` liveness query, default
  query, `bool` refusal on both truth values, non-`int` refusal).
  No wire touches (no live `unitree_sdk2py`, no DDS bus, no driver
  instance); pins the driver-observed contract as a
  module-level snapshot the driver's `_check_motion_gates` refusal
  string can quote verbatim, refs strands-labs/robots#358.
