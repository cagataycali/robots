### Fixed: the documented mesh configuration names the transport selector, not only the extra that installs it

`STRANDS_MESH_BACKEND` is the variable that decides which transport a fleet runs on.
`strands_robots.mesh._backend_select` owns its vocabulary -- `zenoh` (the default), `iot` and
`bridge` -- and two readers consult it: `session._backend_choice` on every publish path, and
`transport.factory.get_transport` once that verdict is `iot` or `bridge`. It appeared in no
documentation page at all, while the `[mesh-iot]` extra that only installs its dependency appeared
in four.

`docs/security.md` then stated the mechanism as the extra: "Adding the `[mesh-iot]` extra routes
traffic through AWS IoT Core". Measured, it routes nothing. With the extra's dependency importable
and the variable unset, `select_backend()` answers `zenoh`, so a reader who followed that page
installed `awsiotsdk`, believed the fleet was on IoT, and got Zenoh. The variable that would have
moved it was missing from the configuration matrix `README.md`'s own AWS IoT Core section sends the
reader to "for the `STRANDS_MESH_*` knobs" -- the pointer led to a table without the one knob that
turns the feature on.

That is the shape `tests/test_docs_device_connect_env_reference.py` already exists for, where
`REACHY_DAEMON_TLS` -- the knob that encrypts a link -- was undocumented while the credential that
link carries was listed, and a reader configured half a posture. A selector documented only by its
dependency is half a configuration.

The matrix now carries a `STRANDS_MESH_BACKEND` row as the first entry of its mesh block, since it
selects which transport the rest of that block's rows apply to, and the security page states both
steps: the extra installs the dependency, the variable selects the transport, and the extra alone
leaves the fleet on Zenoh. A new guard grades all of it against `_backend_select` rather than
against a copied list, so a fourth transport is graded the hour it lands: every value in `BACKENDS`
must appear in the row, every spelling the row advertises must be one the resolver really selects,
and the printed default must be `DEFAULT_BACKEND`. A separate rule refuses the reading this change
closes -- a paragraph that names the extra and makes a routing claim has to name the selector too.
Because the corpus is clean once fixed, that rule is graded on constructed exemplars as well, and
the behaviour that made the omission matter is asserted directly, so it holds on both sides: an
unset variable selects the LAN transport, case and surrounding whitespace are normalised, and an
unrecognized value falls back to the default with one report naming it.

Deliberately unchanged: the other nineteen `STRANDS_MESH_*` variables the package reads and no page
documents. They are rate, size and path knobs *within* a transport, and which of them are public
API is a decision a documentation fix should not make. The selector is separable because it is the
only one whose accepted values the package enumerates in a module constant, which is what lets a
rule about its documented spellings be derived rather than restated.
