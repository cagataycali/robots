### Tests: the training session store poses a finished run whose pid was reissued, not a probe answer no kernel gives

`tests/tools/test_train_session_store_keeps_a_live_pid.py` pins that the store
drops no record on the strength of a probe the load could not take, and its
module docstring explains which call answers "is it running". That explanation
named `psutil.pid_exists` and `Process(pid).is_running()`, which stopped being
the verdict when `list`/`status` moved onto
`_process_stop.session_is_running`: the verdict compares the recorded process
identity, and those two probes are the load path's own, taken by
`_report_uninspectable` for what they raise rather than for what they return.
Production already says so; this file promised existence, and it is the first
sentence a maintainer touching the store reads.

One of the three "a finished run keeps its record" cases posed `is_running()`
answering `False`. That call cannot: a freshly constructed `psutil.Process`
captures the identity it is about to be asked about, so it answers `True` or
raises, and the load path constructs one per read. Measured on the locked psutil
7.2.2, all 652 live pids on the machine answered `True` and none answered
`False`; a fresh construction on a just-exited pid and on a never-used pid both
raised `NoSuchProcess`. Only an object built before the exit and read after it
returns `False`, and no caller here holds one.

The case was also vacuous in the direction it claimed. `_report_uninspectable`
calls the probe for what it raises, so a stub returning `False` produces no
exception and no warning: with the stub flipped to answer `True` the same three
cells still pass, and neither assertion could fail for the reason the case id
gave.

It is re-posed as the state that spelling was reaching for - a run that finished
and whose pid the kernel handed out again. That is the only "finished" reaching
the identity comparison, so the record has to carry `pid_started_since_boot` and
the double has to sit on `_started_since_boot`, the seam the verdict reads on
Linux. The running-verdict control gains the same row: a gone pid alone cannot
separate the claim production makes - the pid still holds the process the record
was written for - from the weaker one that it exists.

Both rows grade a real regression, measured by mutation against this file before
and after. Pruning a record whose pid exists but is not its process leaves the
old cells 3-passed and fails the new `pid-reused` retention row; deriving the
verdict from pid existence alone, as it was before the move, leaves the old
control passing and fails the new `pid-reused` control row.

The docstring also records the asymmetry that makes this file's `psutil.Process`
stand-ins legitimate where the teleoperation sibling's had to move onto
`_started_since_boot`: a record here carries no identity unless a case seeds
one, so `session_is_running` short-circuits to existence and never consults the
double at all. Production is unchanged.
