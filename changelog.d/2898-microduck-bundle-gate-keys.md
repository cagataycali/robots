### Fixed: a `MicroduckPolicyBundle` velocity gate is refused when a key names no held skill

`MicroduckPolicyBundle` checks its structural arguments carefully. It refuses an
empty mapping, refuses a value that is not a `MicroduckPolicy` by name and by
type, refuses an `active` skill outside its own keys while naming the keys it
does hold, and `switch` refuses an unknown name the same way. Its one number,
`switch_on_velocity`, goes through the shared `positive_finite_number_error`
domain. `move_key` and `idle_key` are skill names of exactly the kind `active`
is, and nothing checked them.

They are read only by the velocity gate, which opens by returning early when
either names no held skill — so a wrong key did not fail, it made the gate
inert. The reachable case is the default. The bundle defaults to
`move_key="walk"` / `idle_key="stand"` while Pollen ships its skills as
`alpha_walking`, `alpha_stand`, `roulade`, `ball_kick_*`, so a bundle keyed by
the weight names it loads constructs, reports a validated threshold, and never
switches: a biped commanded to walk at 0.3 m/s stays on `alpha_stand`, and every
tick reports success with the correct 14 joint targets for standing still. One
wrong key is enough, and it kills the whole gate rather than half of it — the
direction whose key *is* a held skill stops working too.

The membership is now asked at construction, naming the offending parameter, its
value, the skills the bundle does hold, and the gate as the reason. It is scoped
to the branch that reads the keys: with `switch_on_velocity` unset the gate never
runs and neither key is consulted, so a caller is not refused for a value the
bundle does not look at. An explicit `switch(...)`, a bundle keyed by the
defaults, a bundle whose keys name the shipped weights, and one skill named by
both keys all keep working unchanged.
