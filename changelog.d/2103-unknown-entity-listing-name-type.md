### Fixed: an unknown-entity message lists what IS registered whatever the requested name's type

`registered()` is total so a name that cannot be a registry key - a list, a dict, an
int, `None` - is *reported* rather than raising `TypeError: unhashable type`. The
five unknown-entity messages could not honor that handoff: each gated its whole
tail on `if known and isinstance(requested, str)`, but only the `difflib`
close-match needs a string. For every non-`str` name the availability listing and
the discovery action were suppressed too, and three of the five then asserted a
falsehood - `Robot '['front']' not found. No robots in the scene; add one with
action='add_robot'.` with a robot registered. The other two returned the bare
dead-end `"<Kind> 'X' not found."` these helpers were introduced to eliminate.

The type test now guards only the suggestion, via a shared `close_match_hint()`;
what is registered is a fact about the world, so it is listed for every name type
and the empty-scene claim is made only when the scene really is empty. Two lookups
that raised *before* their own report - `get_sensor_data` and `load_state` - are
total as well, so the message is reached at all. A `str` name's message is
byte-identical.
