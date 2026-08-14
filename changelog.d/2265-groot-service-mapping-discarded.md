### Fixed: a GR00T service-mode key mapping is applied instead of discarded

`Gr00tPolicy` accepts `observation_mapping` and `action_mapping` on both
transports, and they are the only way to drive a model whose channel names
differ from the robot's. Resolving them ran only after a local model loaded, so
on the service transport both were stored and never parsed -- and neither
consumer of an unparsed mapping reported anything.

The observation half is the damaging one. The flat wire builder looked every
channel `data_config` declares up under the model's own name, so a robot naming
its camera `wrist_cam` for `video.wrist` matched nothing: measured on a service
policy carrying both mappings, the payload sent to the server was the
instruction alone -- no video and no state, under a successful call, with the
policy reporting an action chunk built from whatever the server does with an
empty observation. The action half fails one layer on: `_unpack_service_actions`
skips renaming on a falsy mapping, so the caller's requested actuator names were
absent from the returned steps and the bare model keys were returned in their
place.

Both mappings are now parsed on either transport, since each is a rename over a
flat dict and needs no model to read. What service mode still cannot do is
*infer* a mapping it was not given, or cross-check one against the model's
declared channels -- both read `modality_configs`, which only a loaded local
policy exposes -- so an omitted mapping stays unresolved rather than acquiring an
inferred mapping that nothing could validate, and a caller who passed none sends
and receives exactly what they did before.

The mapping is applied inside the flat wire builder rather than by switching
which builder runs. The nested `{"video": ..., "state": ...}` observation is what
the in-process Isaac-GR00T policy takes; every server version this client dials
reads the flat dotted keys, so selecting a robot key must not also decide the
payload's shape. A mapping that names only some channels adds those and leaves
the rest resolving by name, so no channel that reaches the server without a
mapping stops reaching it because one was supplied.
