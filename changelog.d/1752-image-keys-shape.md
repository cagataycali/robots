### Fixed: `image_keys` refuses a shape it cannot honor instead of reading a bare string per character

`image_keys` names an ordered list of key names on two policy providers - the
LeRobot local provider declares the model's VISUAL feature keys with it, and the
VERA provider names the observation cameras to width-concat with it. Neither
validated the shape, and both reduced the value with `list(...)`.

`str` is iterable, so `image_keys="wrist"` was read as `['w', 'r', 'i', 's', 't']`
- five names the caller never wrote, one per character. Nothing downstream could
tell that apart from a deliberate five-entry list, so it was accepted and the
consequence surfaced far from the call: on the LeRobot path with no embodiment
configured, a model built declaring one bogus feature per character; on the VERA
path, a `KeyError: 'w'` raised mid-rollout, after the policy server had been
launched and the model loaded.

Two neighbouring shapes failed the same way. A non-`str` entry became a key. A
repeated entry could not be honored as written: the LeRobot side builds a feature
dict, where a duplicate collapses and declares fewer features than asked for, and
the VERA side concatenates one panel per entry, where a duplicate doubles the
width of the frame the model sees.

All four surfaces that receive the value - `LerobotLocalPolicy.__init__`,
`LerobotLocalPolicy.preflight`, `derive_image_keys` and `VeraPolicy.__init__` -
now share one shape domain (`strands_robots.utils.name_list_error`) and each
refuses before the work it guards: the weight download, the embodiment
early-return that previously skipped the check entirely, and the VERA server
handshake. The refusal names the per-character reading and the wrapped list to
pass instead. A falsy `image_keys` keeps its existing meaning of "not supplied",
so the list is still derived.
