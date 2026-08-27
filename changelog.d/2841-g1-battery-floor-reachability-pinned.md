### Added: the g1's battery floor is pinned as unreachable behind the un-wired FSM

``G1Driver._check_motion_gates`` refuses with ``FSM id unknown - motion-switcher
source has not been wired`` on every connected driver because ``_fsm_id`` has no
producer -- the decoder that would write it is the exact deferral harness#361
PR-C and #2765 name. Every guard behind that refusal, including the
``battery_floor_pct`` guard the caller is asked to think about, was
unreachable through the verb surface, and no test asserted it. The shipped
suite injected ``_fsm_id`` on every driver it built and read the battery
guard past a gate the vendor cannot open, so a caller who read the tests
green over ``send_action`` had no way to tell that on hardware the battery
guard never fires.

Three contracts are pinned as literals so a change to any of them fires this
file, not a distant one:

- On a fully-healthy driver except for ``_fsm_id`` (``_connected=True``,
  ``_mode_machine`` from a real ``rt/lowstate`` decode, ``_battery`` from a
  real ``rt/lf/bmsstate`` decode), ``send_action`` at 1.0% against a 15.0%
  floor refuses with ``FSM id unknown``. The refusal does not mention
  ``battery`` and does not carry the pack percentage -- the caller learns
  about the FSM, not the pack, because the FSM gate fires first.
- ``_fsm_id`` has exactly one assignment in ``strands_robots.drivers.g1`` and
  it is the ``None`` initialiser. The check is an :mod:`ast` walk over the
  shipped source, so a comment that mentions ``self._fsm_id = 500`` in a
  docstring is not counted as a write. The assigned value is graded as an
  ``ast`` node rather than by substring, because the snippet carries the type
  annotation ``int | None`` and so still reads ``None`` under a mutation like
  ``self._fsm_id: int | None = 501`` -- which is exactly the silent gate-open
  the contract has to refuse.
- The FSM check precedes the battery-floor comparison inside
  ``_check_motion_gates``. The reachability contract above reads whichever
  gate fires first, so that order is what makes it a statement about the
  battery floor. Pinning it separately means a reordering fails a test whose
  name says "ordering" rather than only the reachability test, whose message
  would otherwise report ``FSM id unknown`` for a change that was about the
  order. Whether FSM-before-battery is the right order is a separate
  question: the gate justifies it with "the caller has already been told the
  FSM if the FSM is the reason", which holds for the FSM-value refusal and
  not for the FSM-unknown branch that fires today.

The day a real motion-switcher decoder writes ``_fsm_id``, both assertions
fire -- correctly, because on the same day the battery floor becomes
reachable and this test file must be replaced by one that grades the new
reachability. The failure message names the second write site so the
replacement is targeted, not exploratory.
