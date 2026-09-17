### Added: `eval_policy(success_when=...)`

`eval_policy` accepts `success_when`, a predicate clause in the same DSL as
`run_policy`'s `stop_when` and a benchmark's `success` clause
(`{'predicate': 'body_above_z', 'body': 'cube', 'z': 0.2}`), compiled through
the closed registry and probed against the live scene before the first
episode, so a clause the scene cannot satisfy is refused up front instead of
scoring every episode a miss. Before, the only criterion an agent-tool call
could express was `success_fn='contact'`; a predicate spelled as a string was
refused without saying what IS accepted. The unknown-string refusal now names
`'contact'` and points at `success_when`.
