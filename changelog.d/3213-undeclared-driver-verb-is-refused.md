### Fixed

- Native drivers no longer run their halt verb for an agent `action` their own
  `tool_spec` does not declare. Eleven of the twelve shipped drivers dispatched
  with a bare `else`, so a typo (`"sensor"`), a verb borrowed from a sibling
  (`"sensors"` on a `URDriver`, whose read verb is spelled `state`; `"stop"` on a
  `CrazyflieDriver`, whose halt is spelled `land`) or a non-string all halted the
  robot and answered as though the caller's own verb had been dispatched. Each
  dispatcher now names its halt verb and refuses an undeclared one, naming every
  declared verb so an agent can correct itself in the same turn.
- `strands_robots.drivers.base` gains `declared_verbs` and
  `undeclared_verb_error` as the single owner of that verb list and that
  refusal, read back off the schema an agent planned against rather than
  restated per driver. `FeetechDriver` - which already refused - drops its own
  copy of both.
