### Fixed: a mesh reply nobody can attribute no longer counts as an emergency stop acknowledgement

On the Zenoh backend, presence and command replies were published with a plain
`put` that carries no `SourceInfo`, so `Mesh._on_response` could never bind a
reply to the session that sent it and accepted any reply whose topic matched the
`responder_id` its body claimed. Any admitted peer could therefore acknowledge
`Mesh.emergency_stop` in another robot's name. Presence and replies now carry
their Zenoh session id, and on Zenoh a reply with no verifiable source, or on the
legacy response key that names no responder, is refused; such a peer is reported
in `peers_silent`, never as stopped. The `iot` and `bridge` backends, whose broker
binds the response topic to the sender, still accept replies without a session id.
