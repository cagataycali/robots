### Fixed: report an unknown ``STRANDS_MESH_BACKEND`` instead of falling back to Zenoh in silence

``strands_robots.mesh.session._backend_choice`` -- the reader
:class:`strands_robots.mesh.Mesh` consults -- mapped any unrecognised
``STRANDS_MESH_BACKEND`` value to ``zenoh`` and said nothing, so a typo such as
``STRANDS_MESH_BACKEND=ioT`` or ``iot-core`` left every publish on the LAN Zenoh
path while the operator believed the fleet was talking through AWS IoT Core.
Nothing distinguished it from a deliberate Zenoh deployment: no log record, and
no surface reports the resolved backend -- ``Mesh.__repr__`` carries the peer id
and liveness only -- so there was no channel on which to notice the substitution.

The unknown value is now reported once per distinct spelling, with the same
message :func:`strands_robots.mesh.transport.factory._select_backend` already
logs for that value, so the two readers of one environment variable answer
alike.

Both readers build that report with one function,
``strands_robots.mesh.transport.factory.unknown_backend_message``, which names
the variable, the accepted set and the consequence -- the peer joins the LAN
mesh only, so a cloud subscriber receives nothing from it -- and quotes the raw
value. The normalized form renders a whitespace-only variable as ``''``, which
reads as unset, and unset falls back by design and correctly says nothing.
