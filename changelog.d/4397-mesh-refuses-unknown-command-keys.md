### Fixed: a mesh command key the peer does not read is refused, not silently dropped

`mesh.security.validate_command` built its output from the keys it validated
and dropped the rest. Nothing unknown was forwarded, but nothing was reported
either, so a typo let the default it was meant to override take over:
`send(peer, {"action": "execute", ..., "durration": 0.5})` answered `success`
after the 30 s default rollout. Such a command is now refused on both the
sending and receiving side, naming each unread key, the closest key the action
does read, and the action's key list (`mesh.security.COMMAND_KEYS`). A key that
belongs to another action (`duration` on `status`) is refused the same way.
