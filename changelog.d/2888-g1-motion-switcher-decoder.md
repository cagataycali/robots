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

The seam that loads the SDK names
``unitree_sdk2py.comm.motion_switcher.motion_switcher_client``.
``MotionSwitcherClient`` ships under ``comm/`` rather than ``g1/`` because the
motion switcher is shared across platforms — the SDK's ``example/g1``,
``example/h1``, ``example/h1_2``, ``example/go2``, ``example/b2`` and
``example/b2w`` low-level examples all import it from that one place, and
``unitree_sdk2py/g1/`` holds only ``arm``, ``audio`` and ``loco``. The path is a
named module constant so it is a reviewable fact rather than a string buried in
a call, and it is graded two ways: a stand-in SDK tree registered at the real
path (dependency-free, so it holds on every install including CI) and a call
against the real SDK where one happens to be importable.

Wire-side invariants: ``unitree_sdk2py`` is not imported at module load
(mirrors the invariant ``_dds_engine`` and ``_g1_common`` already carry);
the ``(status, result) → _fsm_id`` mapping is spelled once, so no second
copy of "which key carries the FSM" can disagree; shapes the SDK never
returns (non-tuple, non-dict result, missing keys, ``bool`` masquerading
as the ``int`` FSM id) refuse with a message naming the received type
rather than defaulting to a value the gate might silently open on. The length
named in that shape refusal is read through
:func:`strands_robots.utils.sequence_length`, the package's single owner of
"how many components is this?", because a 0-d numpy array or torch tensor
declares ``__len__`` and raises from it — so a ``hasattr``/``len`` probe would
escape the refusal path with a bare ``len() of unsized object`` out of the one
function whose purpose is to answer an unusable input with a message.

Which key carries the FSM id is *not* settled by this change. ``CheckMode``
returns ``json.loads(data)`` straight from the robot, so the Python package
cannot say which keys that payload holds, and the string ``"form"`` appears
nowhere in ``unitree_sdk2py`` — every SDK example reads ``result['name']`` and
nothing else. That question is one of the wire-format decisions #2765 tracks and
answering it needs a robot; until then an active mode whose payload carries no
integer ``form`` is refused rather than defaulted, so a wrong guess surfaces as
a named refusal instead of an FSM id the gate might open on.

Wiring the decoder onto ``G1Driver._fsm_id`` is a separate PR because the
existing pin at
``tests/drivers/test_g1_battery_floor_is_gated_behind_the_unwired_fsm.py``
asserts exactly one ``_fsm_id`` writer today and must be replaced in the
same PR that adds the second one.
