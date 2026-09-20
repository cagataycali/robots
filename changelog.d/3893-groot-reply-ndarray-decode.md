### Fixed: a GR00T service-mode action chunk is decoded from the reply's own encoding

The two directions of the GR00T ZMQ wire are not symmetric.
`gr00t.policy.server_client.MsgSerializer` decodes both array envelopes -- this
client's `__ndarray_class__` / `as_npy` pair and its own -- so a request is read
by the reference server either way, but a reply is always packed by
`msgpack_numpy`: a map carrying `nd` / `type` / `kind` / `shape` / `data`. That
map was never decoded, so a well-formed `(1, 16, 5)` float32 chunk reached
`Gr00tPolicy._unpack_service_actions` as a dict, became a 0-D object array under
`np.asarray`, and was refused as `scalar (0-D) action value(s) ... The action
chunk is malformed` -- blaming the model for a chunk the client never decoded,
on every request to every reference server. The envelope is now decoded, and one
declaring an object dtype is refused instead: only `pickle` reads that, which no
path here does.
