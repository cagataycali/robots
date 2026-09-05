### Fixed: a robot alias is keyed by the fold every lookup applies to its query

`resolve_name` folds a caller's query before looking it up - lowercase, trim,
dash as underscore - and canonical robot names are stored folded, so they match.
Aliases were keyed as *declared*, and that asymmetry did not merely fail a
lookup. An alias whose declared spelling was not already folded was unreachable
in every spelling, including the one it was registered with, because the query is
folded before it reaches the map; and the fold could carry it onto another
robot's key, so `register_robot(name="probe_arm", aliases=["Franka-Panda"])` was
accepted and `resolve_name("Franka-Panda")` then answered `panda`, with
`get_robot` returning the Franka's entry - a name declared for one robot
resolving to another.

The uniqueness constraints in `_validate_robots` compared declared spellings
too, so two aliases that are one key to every reader passed validation and the
alias map kept whichever entry merged last (the user overlay). Both sides now
compare under the fold, which is what lets the fail-closed check in
`register_robot` refuse such an alias at registration instead of persisting a
lookup that lands on someone else. An alias that folds to its OWNER's canonical
name stays legal - the same carve-out `_validate_policies` already makes with
`alias != provider_name` - which is what the shipped `reachy_mini` /
`reachy-mini` pair needs.

The fold had four hand-written copies across `discovery`, `robots` and
`user_registry`. It now has one owner, `loader.normalize_robot_name`, exported
from `strands_robots.registry` so a caller can predict which robot a name
reaches, and documented in `docs/api-reference.md` and
`docs/getting-started/robot-factory.md`.
