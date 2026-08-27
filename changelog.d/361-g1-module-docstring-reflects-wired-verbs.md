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
`tests/drivers/test_g1_module_docstring_reflects_wired_verbs.py` drives all
four verbs on a driver whose gates would pass (healthy pack, allowed FSM,
`mode_machine` known), derives from those responses which verbs actually
carry the refusal idiom, and then refuses any sentence in the docstring
that attributes the refusal to a verb outside that set. Deriving the set
rather than listing it is what makes the check a measurement: the stale
wording named four verbs in a single sentence that used the idiom once, so
counting occurrences of the phrase, or asking whether that sentence
mentions `start_task`, is satisfied by it.

Attribution is read on the subject side of the idiom - the verbs named
before it - because the corrected wording names `run_policy` after it, as
the verb a caller should use instead. Reading the whole sentence would
score that as a second refusal claim and refuse the corrected text.

The same file pins that `start_task` still answers a refusal naming issue
#358. When the provider registry lands and that verb is promoted, three
cells fire: the one pinning the refusal, the attribution rule (the derived
set becomes empty while the docstring still names a refusing verb), and
the premise that the derived set is non-empty. Each names a different part
of what the wiring commit has to update, so the docstring cannot be left
behind again.
