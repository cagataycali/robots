### Fixed: the G1 write refusal pointed the operator at a repository with no owner, and the soft-stop helper documented itself as dead code

`G1Driver.send_action` refuses every write until the motion-switcher source that
feeds `_fsm_id` is wired, and the message it returns carried two tracker
references: a bare `#2765`, and `harness#361 PR-C`. The second one resolves to
nothing. It names a repository without an owner, so there is no coordinate for
the reader to open - not a link, not a path, not a search that lands anywhere.
The refusal it sat in is not a rare one either: `_fsm_id` has exactly one
assignment in the driver and it is the `None` initialiser, so this is the message
every real driver produces for every write, which made the unresolvable half the
most-read tracker reference in the package. Driven against the real driver with
`_connected` set, `rt/lowstate` delivered and a 92 percent pack, `send_action`
answered:

```
FSM id unknown - motion-switcher source has not been wired (harness#361 PR-C); see #2765 for the wire-side decision
```

and now answers:

```
FSM id unknown - motion-switcher source has not been wired; see issue #2765 for the wire-side decision
```

The verdict is unchanged - the gate still refuses, for the same reason - and both
phrases the existing tests pin (`FSM id unknown` and `motion-switcher`) survive.
What changes is that the remedy the message advertises can be followed. This also
brings the refusal into line with every sibling: the Dynamixel driver's
`_NOT_WIRED` constant and the G1's own `start_task` both cite an in-repo issue
(`issue #359 bus`, `issue #358`), and this was the one that did not.

`_build_zero_torque_lowcmd` had the complementary problem: its docstring
described a state the code left behind. It read "This helper is defined but not
yet wired: `G1Driver.stop` and `stop_task` currently return refusal envelopes
rather than publishing a frame, and no other call site exists." Every clause of
that is now false. `_ControlLoop._emit_zero_torque` calls the helper from the
loop's `finally`, on every exit path except one where the wire itself just
refused a publish, and `G1Driver.stop_task`'s own docstring 370 lines above
already said as much ("the loop publishes `_build_zero_torque_lowcmd` on the way
out"). Two docstrings in one module contradicting each other is worse than either
being vague: a reader who meets the helper first is told the soft-stop path does
not exist, and the reasonable conclusion is that one still has to be built beside
it. The paragraph now describes the call site and why the helper stays separate
from the loop.

Two guards hold the classes out, with scopes chosen by measurement rather than by
taste. The first sweeps every string a *caller* can receive across the package
and requires any tracker reference in it to be resolvable: a bare `#NNNN` for an
issue here, or an owner-qualified `owner/repo#NNNN` for a sibling. Both spellings
are pinned as accepted so the rule is not blanket strictness, and the predicate is
graded on constructed exemplars because a clean tree can no longer exercise a
rejection. Docstrings are deliberately excluded and a boundary test records why:
over twenty `robots-sim#NN` references sit in developer-facing prose in
`simulation/isaac`, `strands-labs/robots-sim` is a real repository, and a
maintainer reading a docstring has the sibling checkout that an operator holding
a refusal envelope does not. That test fails if the docstring population ever
empties, at which point widening the scan is free. Across the whole package
exactly one caller-reachable string violated the rule, and it was this refusal.

The second guard is scoped to the G1 module and to its module-private functions,
and requires that a helper the module calls does not document itself as uncalled.
That scope is the one in which matching a call by name is sound, because a
leading-underscore module-level function is called by its bare name inside its own
module. The same rule package-wide is unsound and is not shipped: a public method
name such as `send_action` is declared on several unrelated classes, so counting
calls by name would attribute every `robot.send_action` in the tree to whichever
driver happened to declare it, and the Dynamixel driver - which legitimately
documents its own bus as not yet wired - would be reported as drifting when it is
not.
