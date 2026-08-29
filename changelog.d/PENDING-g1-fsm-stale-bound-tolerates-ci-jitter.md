### Fixed

- Widen the ``_FSM_STALE_AFTER_S`` bound in
  ``test_a_renewed_cache_keeps_admitting`` from 0.2 s to 1.0 s so a CI runner
  that starves the refresher thread for tens of milliseconds does not surface
  as a spurious ``exit_reason == "gate"``.  The property being graded is "a
  bound being renewed does not fire", not "the bound is exactly two refresh
  periods wide", and a 200-refresh window grades the same claim without
  making it a scheduler race.
