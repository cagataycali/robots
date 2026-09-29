### Fixed: a GR00T `get_modality_config` reply from a current N1.7 server is decoded

`gr00t.policy.server_client.MsgSerializer` marks a `ModalityConfig` on the wire
with `__ModalityConfig__` (Isaac-GR00T 51d4c89 and later) and still reads the
older `__ModalityConfig_class__`, which was the only spelling this client read.
Against a current server every `get_modality_config` reply therefore came back
as its raw marker map, so nothing in service mode could learn which keys the
server declares. Both spellings are read now, as the server does, and a marker
without its `as_json` payload is refused instead of being returned half decoded.
