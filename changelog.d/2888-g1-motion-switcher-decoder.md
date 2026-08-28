### Added: `strands_robots.tools.g1._motion_switcher` decodes the FSM id off `MotionSwitcherClient.CheckMode()` (issue #2765)

The G1 driver's ``_check_motion_gates`` compares ``_fsm_id`` against
``HANDSHAKE_FSMS`` and ``WALK_FSMS`` to decide whether an arm-SDK or
locomotion write is admitted. On the shipped tree ``_fsm_id`` had exactly
one writer (the ``None`` initialiser); the producer is the motion-switcher
API, not ``rt/lowstate``. This adds the *decoder* half — the mapping from
``CheckMode()``'s ``(status, result)`` return to the integer the gate
reads:

```python
from strands_robots.tools.g1._motion_switcher import decode_fsm_id, read_fsm_id

reading = decode_fsm_id((0, {"name": "ai", "form": 500}))
# FSMReading(fsm_id=500, mode_name='ai', refusal=None)

reading = read_fsm_id(client)  # calls CheckMode(), decodes, catches SDK errors
```

Wire-side invariants: ``unitree_sdk2py`` is not imported at module load
(mirrors the invariant ``_dds_engine`` and ``_g1_common`` already carry);
the ``(status, result) → _fsm_id`` mapping is spelled once, so no second
copy of "which key carries the FSM" can disagree; shapes the SDK never
returns (non-tuple, non-dict result, missing keys, ``bool`` masquerading
as the ``int`` FSM id) refuse with a message naming the received type
rather than defaulting to a value the gate might silently open on.

Wiring the decoder onto ``G1Driver._fsm_id`` is a separate PR because the
existing pin at
``tests/drivers/test_g1_battery_floor_is_gated_behind_the_unwired_fsm.py``
asserts exactly one ``_fsm_id`` writer today and must be replaced in the
same PR that adds the second one.
