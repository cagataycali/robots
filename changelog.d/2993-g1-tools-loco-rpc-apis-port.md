### Feature

- Added `strands_robots.tools.g1.g1_list_loco_rpc_apis` and
  `strands_robots.tools.g1.g1_loco_rpc_api_admits`, two read-only
  ``@tool`` verbs that snapshot the six raw ``_Call`` api ids the neon
  bundle's ``_g1_common.py``
  (``cagataycali/neon-the-g1/tools/_g1_common.py::_loco_call``) reaches
  through ``LocoClient._Call(api_id, payload)`` under a single-writer
  lock: ``7001`` ``SetFsmId``/``GetFsmId`` (the read half is
  ``read_fsm_id``), ``7002`` ``SetFsmMode``/``GetFsmMode``
  (``read_fsm_mode``), ``7003`` ``SetBalanceMode``/``GetBalanceMode``
  (``read_balance_mode``), ``7004`` ``SetSwingHeight``/``GetSwingHeight``
  (``read_swing_height``), ``7005`` ``SetStandHeight``/``GetStandHeight``
  (``read_stand_height``), and ``7103`` ``SetSwingHeight`` (write) --
  the one write api id in the set, called by neon's ``set_swing_height``
  helper. The Unitree G1 locomotion SDK does not ship a canonical api-id
  to operation-name mapping, so the snapshot is the neon bundle's
  observation against the real robot rather than an SDK re-import; the
  module pulls no ``unitree_sdk2py`` submodule at import time (the
  SDK-load-hygiene contract every other file under
  ``strands_robots.tools.g1`` carries, refs strands-labs/robots#358).
  Each admitted api id carries an ``operation`` label naming the SDK
  method pair the neon bundle observed, a ``kind`` label partitioning
  the set into ``"read"`` (the five state readers) and ``"write"``
  (``7103`` alone), and a ``touches_motion_gate`` flag naming the one
  api id (the write) that shapes an ``rt/lowcmd``-adjacent locomotion
  write the driver's ``_check_motion_gates`` refuses on ``_fsm_id``
  outside ``WALK_FSMS`` (the same gate merged in
  strands-labs/robots#2916). The envelope also names
  ``walk_ready_fsm_ids`` quoting ``WALK_FSMS`` for the gate-set half of
  the decision, ``transport_refusals`` carrying the ``3102`` (RPC send
  fail) and ``3104`` (RPC timeout) codes and their decoded text from
  ``ERR_CODES`` (the transport-level refusals a caller can see
  independent of api-id validity), and ``refusals`` carrying the
  ``7404`` invalid-api-id code (the same gate-refusal shape a
  locomotion-write refusal quotes, because the SDK does not ship a
  distinct code for an unknown api id). The admit verb refuses a
  ``bool`` argument before the ``int`` lookup runs so a caller passing
  ``True`` sees the type mismatch named rather than a confusing lookup
  miss on ``1``; a non-integer non-bool argument is refused with the
  same code for the same reason; and a missing argument is refused
  with the same code so the lookup is decidable in every case. Refs
  strands-labs/robots#358 for the SDK-facing gate work the future
  driver-side wrapper for ``_Call`` will land on.
