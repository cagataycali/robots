### Removed: the `vera` policy provider

`strands_robots/policies/vera/` is gone in full, along with the `[vera-sim]`
extra, its five `[tool.uv] conflicts` rows, the `vera/docker/` mypy exclude, the
`^vera://` URL-scheme branch in `strands_robots/registry/policies.py`, the
provider's `policies.json` entry, its `mesh/security.py` policy-type allowlist
row, 31 test modules, `tests_integ/policies/vera/`,
`examples/vera_mimicgen_panda/`, `docs/policies/vera.md` and its `mkdocs.yml`
nav entry.

VERA is distributed only as a git repository, so the provider could never
declare a dependency on the package it launched (`python -m vera.server` as a
subprocess) - PyPI rejects metadata carrying a VCS reference. The provider was
therefore the one entry in the catalogue whose install column said
`_(none - git-only)_`, and the only reason the manifest carried
`gymnasium==0.29.1`, `gym-pusht`, `robosuite` and a five-row resolution fork
keeping that pin away from every lerobot-0.6 extra.

Two modules that looked shared were vera-only copies and go with it:
`vera/_msgpack_numpy.py` (a near-copy of the cosmos3 packer) and
`vera/sim_ik.py` (a second thin subclass of the shared
`strands_robots.simulation.ik.MinkIKBridge`). The shared bridge and
`discover_ee_frame` are untouched, and cosmos3 keeps its own copies.

`discover_ee_frame`'s exhaustive ladder pin moved with the function rather than
being deleted: `tests/policies/vera/test_ee_frame_discovery.py` is now
`tests/simulation/test_ee_frame_discovery.py`, reading the shared
`strands_robots.simulation.ik` home instead of the provider's re-export. It was
the only coverage of all three discovery rungs, and `move_to` still depends on
every one of them.

Three guards were re-anchored rather than deleted, because each pins a fact that
outlives the provider. `tests/policies/test_policy_client_wire_reads_state_a_deadline.py`
drove its read-deadline and connection-discard cells through the VERA WebSocket
client; the rule belongs to the wire, and `Cosmos3WebsocketClient` states it in
the same two places (handshake published only after the metadata frame is read,
connection discarded on an incomplete exchange), so those cells now drive the
surviving client. `tests/registry/test_config_keys_agree_with_constructors.py`
used the provider as its no-`**kwargs` worked example; `cosmos3` is the
surviving one. `tests/policies/test_image_keys_shape.py` graded one option name
across two providers, and now grades the three LeRobot surfaces that remain -
with the emptiness verdict no longer a divergence, because only the
"declares the model's features" reading is left.

Two non-vacuity floors were re-measured against the smaller tree, not relaxed:
the child-stream scan finds 19 pipe reads (was 20) and the WebSocket scan finds
2 `recv()` calls in one client module (was 4 across two).

`uv lock` shrinks from 258 to 237 package entries (6,225 to 5,595 lines):
`gymnasium` drops from two forked entries to one (1.3.0), `transformers`
collapses from two entries to one, and `robosuite`, `gym-pusht`, `scikit-image`,
`shapely`, `tifffile` and `evdev` leave the lock entirely. Resolution forking
itself has not gone away - `autobahn` and `torchcodec` still carry two entries
each on platform markers - so the "every locked version below the floor, not
any" rule in `tests/test_lockfile_parity_gate.py` is unchanged; its docstring
now names the two distributions that are still multi-version. This supersedes
the line in `3056-remove-vendored-libero-benchmark.md` that recorded `robosuite`
staying at 1.4.1 through `[vera-sim]`.

Twenty-three unreleased `changelog.d` fragments describing this provider's own
fixes go with it, matching how `3056-remove-vendored-libero-benchmark.md` handled
the LIBERO ones: an entry documenting a knob on a provider removed in the same
release documents nothing a reader can reach. Fragments that merely *mention* the
provider alongside others are left verbatim - `changelog.d/` is a historical
record, and the extra-name guard deliberately exempts it for that reason.

`[all]` never included `[vera-sim]`, so the bundle is unchanged at 21 extras;
the total drops from 34 to 33. `README.md`, `docs/architecture.md`,
`docs/getting-started/installation.md`, `docs/policies/overview.md`,
`docs/policies/lerobot-local.md`, `docs/recording.md`, `examples/README.md` and
`AGENTS.md` follow.
