### Removed: the LIBERO benchmark adapter and the GR00T policy provider

`lerobot` ships both upstream, so the in-repo copies were duplicated surface:
two names for one capability, each with its own tests, docs page and install
extra to keep honest, and a reader deciding which to reach for had no way to
tell them apart. `strands_robots/benchmarks/` (the LIBERO adapter, its BDDL
parser and suite loader), `strands_robots/policies/groot/` (the ZMQ/HTTP
`Gr00tPolicy` and its 25 data configs), `strands_robots/training/groot.py` and
`strands_robots/tools/gr00t_inference.py` are removed whole, with their tests,
`tests_integ`, examples, `docs/policies/groot.md` and the `groot-service` /
`benchmark-libero` extras. Net -44,776 lines.

The seams the adapter drove are not part of the removal, because none of them
was LIBERO-specific: `world._backend_state["viz_option"]` and
`["action_controller"]` are still honoured by the MuJoCo and Isaac backends,
`strands_robots.simulation.isaac.delta_eef` still converts 7-dim task-space
delta-EEF actions into joint control, and
`strands_robots.simulation.predicates` still resolves the LIBERO / robosuite
`<name>_main` body and `<body>_g<idx>` geom naming conventions. An out-of-tree
adapter plugs into every one of them.

Four consequences a caller can observe, each deliberate. A `policy_provider=`
default that named `"groot"` now names `"cosmos3"`, the surviving service-mode
VLA with the same dial-a-port shape, across eleven signatures in
`hardware_robot.py`, `drivers/` and `tools/g1/g1_start_task.py`. `"groot"`
leaves `mesh.security._REGISTRY_POLICY_PROVIDERS`, which is documented as every
spelling of every provider in `registry/policies.json`, so a mesh peer naming it
is refused with a value rather than reaching a missing module. `resolve_policy`
no longer declares `^zmq://` or the blanket `nvidia` HuggingFace org -- both
were GR00T's -- so its ladder, its URL parser branch and its docstring drop them
together, which is what `test_resolution_order_names_the_shipped_url_forms.py`
already refuses to let drift apart. And `[all]` is now eighteen of
twenty-nine extras.

Two things the removal made vacuous rather than merely smaller, both replaced by
the property they were protecting rather than deleted. The
`coverage`+`numba`+`robosuite` environment audit in `test_dependency_audit.py`
existed because the LIBERO adapter imported robosuite's OSC controller; with no
such import the clash has no path into the package, so it pinned nothing. And
`test_zmq_timeout_ms_domain.py`'s pairwise "the two clients agree on every
verdict" cannot hold anything with one client left, so it now compares each
client's refusal against `coerce_zmq_timeout_ms` itself -- the stronger
statement, since a client that re-derived the same wording by hand would have
passed a peer comparison.

The shared provider tables were retargeted rather than thinned: the seven
trainer domain suites, the service-port domain, the state-key contract, the
registry public API and the mesh `policy_type` allowlist each keep the same
number of backends under test by moving GR00T's row to a surviving provider.
Where a substitution would have contradicted a table's own premise it was not
made -- `test_seed_domain.py`'s "a backend that ignores the field" class holds
only `MockTrainer` now, because `Cosmos3Trainer` reads `seed`.

The port-requirement pins in `test_hardware_policy_port_domain.py` and
`test_hardware_robot_lifecycle.py` now name `moveit2` explicitly instead of
leaning on the default provider. The demand for a port comes from the registry's
`requires` field, and `cosmos3` declares its own default port, so a test that
reaches that field through whichever provider happens to be the default answers
a different question than the one it asks.
