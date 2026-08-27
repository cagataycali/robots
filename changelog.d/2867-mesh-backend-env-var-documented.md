### Fixed: the mesh docs did not name the env var that actually selects the transport

`docs/mesh.md` covered the `[mesh-iot]` install extra but never mentioned
`STRANDS_MESH_BACKEND`, the variable `strands_robots/mesh/_backend_select.py`
is the sole owner of. A reader following the blog snippet
``os.environ["STRANDS_MESH_BACKEND"] = "iot"`` could not find the name in the
published docs and had cause to doubt whether the env var was live at all,
even though `_backend_select.BACKEND_ENV_VAR` is the string both the session
gate (`session._backend_choice`) and the transport factory
(`transport.factory.get_transport`) read from and typos are reported through
its `_UNKNOWN_WARNED` set.

The docs now name the variable, list its three accepted values with the
transport each one selects, spell out that the install extra and the env var
are the two halves of the same choice (extra installs the dependency, variable
selects the runtime), and record the case/whitespace normalisation plus the
once-per-distinct-value fallback for a typo. The `[mesh-iot]` extra is where
the `iot` and `bridge` values need their client library, so both rows point at
it; `zenoh` needs no extra and stays the default that every unset value
resolves to.

Closes cagataycali/robots-harness#375, which the blog PR
strands-agents/harness-sdk#4024 opened after the reviewer could not find
`STRANDS_MESH_BACKEND` anywhere under `docs/`.
