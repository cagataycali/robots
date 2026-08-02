### Fixed: a container guard no longer raises while rendering the container it refuses

The vector and list guards in `strands_robots/utils.py` interpolate the whole
container into their refusal text - `finite_vector_error`, `pose_vector_error`,
`coerce_pose_vector`, `coerce_rgba`, `coerce_size_vector` and `name_list_error`.
`repr` of a list recurses into its elements, so a container was unrenderable
whenever any *one* of its elements was, and the guard raised instead of returning
the structured refusal its callers document as the only channel a bad value is
reported on. The same guards also converted an element with a bare `float()` to
test finiteness, so an element past `sys.float_info.max` raised before any
message was rendered at all.

```
finite_vector_error("raycast", "origin", [10**5000])           # ValueError
pose_vector_error("add_object", "position", [10**5000], 3)     # ValueError
name_list_error([10**5000], "cameras", "render_all")           # ValueError
finite_vector_error("raycast", "origin", [10**400])            # OverflowError
coerce_rgba("add_object", "color", [10**400] * 4)              # OverflowError
```

These arrive from an agent tool call or a `device_connect` `@rpc()` payload -
`raycast(origin=)`, `add_object(position=, size=, color=)` and the recorder /
render `cameras` lists - where a list of JSON numbers is the normal shape and
Python integers are arbitrary-precision, so such a value is one request away.

Both escapes are now answered. Each is reported as a structured refusal:

```
raycast: 'origin' must contain numbers within the range of a 64-bit float,
got [1.0, <int of 16610 bits>, 3.0]
```

#### Why a container needed a rendering rather than a fallback

The scalar guards were made total by routing every rejected value through
`_refusal_repr`, which answers `<unrepresentable Foo>` when `repr` fails.
Applied to a container that is close to useless: it erases every element that
rendered perfectly well, and the element **count** with them - and the count is
frequently the refusal's entire reason, as in `must be a 3-element vector, got
4`. So containers get `_refusal_container_repr`, which describes the container
component by component and substitutes only the components that cannot print.

Three decisions the change had to settle, recorded because a later reader will
ask:

- **`repr` is tried on the whole container first.** That is not an optimisation:
  a `tuple`, a `dict` and a NumPy array have reprs no elementwise form
  reproduces (`(1.0, 2.0)`, `{'a': 1}`, `array([1., 2.])`), so the fast path is
  what keeps those containers reported in their own notation - and what makes the
  change text-identical for every input that answered before.
- **The offending component is located by position, not by an inserted index.**
  `finite_vector_error` and `name_list_error` already report a per-element index
  in their own text (`cameras[1] must be a name`); an index inserted by the
  renderer as well would state it twice in two forms that could disagree.
- **Nothing is elided, and the rendering is one level deep.** Truncating would
  erase elements that rendered fine, which is the exact failure being avoided.
  Recursing would need cycle detection that the interpreter's own `repr` provides
  for the fast path, and an inner container is never what the message is about -
  a nested list is itself the refusal's reason.

A `Mapping` is rendered as a mapping rather than as the list of its keys, because
`name_list_error` refuses one *for* discarding its values: a message showing only
the keys would perform the discarding it is complaining about.

#### No verdict and no message text moved

The change is additive. Every input that produced a message before produces a
byte-identical one now, pinned as equality rather than as a substring across all
fourteen rendering branches, including the two quoted in the docs. The NumPy
component types the guards advertise as accepted are still accepted, and the
`coerce_*` guards still normalise to plain `float`.

#### Two module-wide invariants, both now at their strongest form

- The rendering scan in `tests/test_refusal_messages_never_raise.py` reports every
  function in `utils.py` that renders a caller value without a shared renderer.
  Its expected table is now **empty**: no function in the module does.
- The conversion scan in `tests/test_conversion_escape_is_closed.py` reports every
  unprotected `float()`. `finite_vector_error` has left it. The three names that
  remain convert only *after* `finite_vector_error` has accepted every element -
  so their conversion provably cannot raise - and that upstream guarantee is
  itself pinned, structurally and behaviourally, rather than hidden behind a `try`
  whose `except` could never run.

One escape in this family is left open and pinned rather than omitted: a value
whose `__iter__` raises something other than `TypeError` still escapes
`finite_vector_error`. It is a third mechanism needing its own message decision,
and it cannot be expressed in a JSON payload at all, so it is tracked separately
as #1878.

Closes #1875.
