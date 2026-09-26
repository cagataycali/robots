### Tests: the mesh bridge cells share one transport stand-in, shaped like the real one

Every test of a mesh bridge's forwarding replaces the transport it resolves -
`ros_action`, `rosbridge_action` or `rtps_action` - with a recorder and reads what
the bridge asked for. Ten modules had grown one for that in eight spellings, and
every one took `**kwargs`, so a recorder accepted calls the transport would
refuse: dropping `gate=never_gated` from `_RtpsTransport.echo`, an argument all
three require and none defaults, left the 5,265 tests in `tests/mesh` green and
the first caller to reach the real transport a `TypeError`.

`tests/mesh/_transport_stand_in.py` is now the one stand-in. It binds each call
against the signature of the symbol it replaces, and `stands_in_for()` takes that
shape from the symbol being patched, so a stand-in cannot grade one transport
while wired to another. A stand-in installed over a stand-in resolves through to
the real callable rather than inheriting the previous one's own
`(*args, **kwargs)`, which would have made every later probe in a re-patching
test accept anything.

`tests/mesh/test_a_bridge_forwards_what_its_transport_accepts.py` reads the
forwards out of the tree - 13 call sites over the four bridge modules - and binds
each against the transport it goes to, so a fifth bridge is graded on arrival
instead of on the first real call.
