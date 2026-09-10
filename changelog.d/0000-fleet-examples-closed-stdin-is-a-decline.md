### Fixed
- The fleet examples' `[y/N]` HITL gates treat a closed stdin (CI, a pipe that ran dry, a detached run) as a decline that is printed and recorded, instead of dying in an `EOFError` traceback - before, `echo y |` approved and executed the first dispatch and then crashed before the summary.
