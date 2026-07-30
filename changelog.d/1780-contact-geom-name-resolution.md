### Fixed: body-level contact predicates now resolve a body's geoms consistently

`get_contacts` reports a contact as a pair of geom **names**, and for a geom the
asset left unnamed it synthesizes `"<body>/geom_<id>"`. Two body-level predicates
consume those names -- `grasped` and `body_on(require_contact=True)`, the gate
every LIBERO `(on A B)` goal is compiled with -- and they resolved them
differently:

- `_body_contact` matched only the `<body>_g` prefix inline, so a geom named
  exactly after its body did not match, while `grasped` accepted that same name.
  For one contact list the two predicates returned **opposite** answers.
- Neither matcher recognised `"<body>/geom_<id>"`, the form `get_contacts` itself
  emits. A Panda scene has 81 unnamed geoms out of 82, so on real MJCF the payload
  producer and its consumers disagreed about the name format and a body physically
  resting on another read as "not touching": a settled cube carrying 2.119 N over
  four contacts failed `body_on(require_contact=True)` while the geometric-only
  check passed.

`_geom_belongs_to_body` is now the single owner of the body-to-geom name mapping
and covers the synthesized name, so the two predicates cannot disagree. The
synthesized form is matched exactly rather than as a `<body>/` prefix, because
bodies are themselves namespaced -- a broad prefix would let body `panda` claim
`panda/link0/geom_1`, which belongs to `panda/link0`.
