### Fixed: `encode_clip` refuses a `quality` the clip encoder cannot honor

`quality` was the one knob in `strands_robots.rendering.encode_clip`'s signature
with no domain, and the dependency's own bound is not a substitute for one.
`imageio-ffmpeg` enforces it with a bare `assert 1 <= quality <= 10`, so the
refusal was an `AssertionError` outside the documented `Raises:` set - and
`python -O` strips assertions, leaving no domain at all: `quality=-5` and
`quality=0` encoded a real but different clip, `nan` and `"8"` leaked raw
arithmetic errors out of the bitrate computation, and `500` surfaced only as
"the encoder wrote no clip". The docstring also advertised `0-10` while the
writer asserts `1 <=`, so `0` was a documented value the code refused, and
`True` was accepted as a silent quality of `1` - the lowest offered, and the
same substitution `mjpeg_frames`' quality guard already rejected in the same
module. A finite number in `[1, 10]` is now required, on both containers, and
a NumPy real is converted for the writer, whose `isinstance(quality, (float,
int))` gate refused `np.int64(8)` despite it naming a usable quality. The two
sibling guards in that module now share the same numeric domain, so an integer
too large to convert to a float reports instead of raising `OverflowError` out
of a function whose contract is to return the message.
