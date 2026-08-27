### Fixed: `strands_robots.drivers.g1` module docstring reflects the verbs that are wired

The module-level docstring said all four task/policy verbs
(`start_task`, `run_policy`, `stop_task`, `get_task_status`) "return a
named 'not wired yet' envelope". That was true when the driver first
landed under harness#361 PR-B, and false since the control loop landed
under harness#361 PR-C: `run_policy` starts a 500 Hz thread (see
`_ControlLoop` in the same module), `stop_task` joins it and reports the
outcome, and `get_task_status` reports the loop's snapshot or the last
exit reason. Only `start_task` still refuses, and it refuses precisely
because the provider registry (Groot/ACT/Diffusion) lives in issue #358
whose vendoring decision is still open - a caller with an already-built
policy uses `run_policy` today.

A shipped docstring that says a verb refuses when the verb actually
publishes on `rt/lowcmd` sends a caller reading the module documentation
to write their own transport layer around a driver that already has one,
or to skip the driver entirely as "not ready". This corrects the
docstring to name what each verb does today and reserves the "refuses"
wording for the one verb it still applies to.

A pinning test in
`tests/drivers/test_g1_module_docstring_reflects_wired_verbs.py` reads the
runtime behaviour of each verb on a driver whose gates would pass
(healthy pack, allowed FSM, mode_machine known) and refuses the docstring
if any of the three wired verbs (`run_policy`, `stop_task`,
`get_task_status`) is described as returning "not wired yet". The test
does not touch `start_task` because that verb is genuinely still a
refusal today; it only pins that the docstring must name that specific
verb as the one that refuses, so a future wiring commit that promotes
`start_task` cannot leave the docstring behind.
