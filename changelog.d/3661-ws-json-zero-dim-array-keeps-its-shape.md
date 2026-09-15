### Fixed: a 0-d array crosses the WS-JSON wire with its own shape

`encode_ndarray` declared the shape of its C-contiguous copy, and
`np.ascontiguousarray` returns an array of at least one dimension, so a 0-d
array was written into the envelope as `shape: [1]` and `decode_ndarray`
faithfully rebuilt `(1,)`. A scalar state value that is `ndim == 0`
in-process reached the served policy as a one-element vector; readers that
accept an array state value only when `ndim == 0` then reported the key
missing and substituted `0.0`, so a real joint reading became a zero-filled
slot behind `RemotePolicy`.

The declared shape is now the caller's. The buffer is one element either way,
so the wire format and `PROTOCOL_VERSION` are unchanged and a decoder already
handles `[]`.
