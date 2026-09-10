### Added: every native driver must have a documented `mode="real"` construction

A robot driven by `strands_robots.drivers` has no lerobot robot type behind it,
so these pages are the only place its hardware bring-up is written down - and the
sections sit beside catalog material that is generated, so a page edit can take
an operating procedure with it while the diff reads as catalog churn.
`tests/test_docs_document_every_native_driver_real_path.py` derives the roster
from `list_native_drivers()` and resolves spellings through `resolve_name()`, so
a driver added later is graded without editing the test and a page may use an
alias. One documented example per driver, not one per robot.
