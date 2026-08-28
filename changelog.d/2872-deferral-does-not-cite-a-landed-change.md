### Fixed: a refusal that defers pending work no longer cites a change this repository already landed

Three caller-reachable refusals pointed an operator at bare issue numbers that resolve, in this
repository, to merged pull requests about unrelated subsystems. `drivers/dynamixel/driver.py`
refused four verbs with `"not wired yet (issue #359 bus)"`, and `#359` is "fix(sim): drive
tendon-transmission actuators via joint name" - merged, and about the simulator rather than a servo
bus; the same number reached a planning agent through that driver's `tool_spec` description.
`drivers/g1.py` deferred `start_task` to `#358`, which is "test(mesh): fix flaky
test_session_config" - merged, and about zenoh mock isolation.

A tracker reference inside a refusal is a remedy, and two speech acts carry one. A backward citation
explains where text came from (`"#168 bug I: the cached scene diverges"`) and wants a merged change,
so it gets the right referent. A forward deferral says the capability is not here yet and wants
outstanding work, so pointing it at a landed change tells the reader the refusal is stale rather
than that the capability is missing. That is a worse failure than an unresolvable reference: an
unowned `harness#361` fails loudly because there is no link to follow, while a bare `#359` resolves,
looks authoritative and misinforms. Neither capability has an open issue here, so the misdirecting
number is removed and the missing thing named instead; the `"not wired yet"` idiom a sibling guard
derives from is preserved.

`tests/test_deferral_strings_do_not_cite_a_landed_change.py` owns the destination half of the
contract whose resolvability half `tests/test_source_strings_resolve_their_issue_references.py`
already owned, reading that module's caller-reachable scope so the two rules cannot disagree about
which strings an operator sees. Both of its conditions are load-bearing: without the deferral test it
flags four legitimate backward citations, and without the landed test it flags the deferral that
correctly points at open issue `#2765`. The landed-number oracle is derived from this repository's
own history, and a non-vacuity floor fails loudly on a shallow clone rather than passing an
ungradable rule.
