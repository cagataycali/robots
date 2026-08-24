### Fixed: `encode_clip`'s `macro_block_size` is held to the shared whole-number domain

`encode_clip` has three encoder knobs and only two were checked: `fps` resolved through
`positive_whole_number_error` and `quality` through `_clip_quality_error`, while
`macro_block_size` was handed to the writer unchecked. The accepted set was wrong in both
directions.

`imageio`'s ffmpeg plugin normalizes a falsy value (`macro_block_size = macro_block_size or 1`)
and only rounds when the value is `> 1`, so `0`, `-4`, `None`, `False` and `True` were all read
as "no rounding" while the caller was told the clip had been written. On 60x40 frames a request
for `macro_block_size=0` produced a clip byte-identical to `macro_block_size=1`, so nothing
could tell a dropped request from an honored one. And `imageio-ffmpeg` enforces the knob's type
with a bare `assert isinstance(macro_block_size, int)`, which `python -O` strips: on an
optimized interpreter `nan` produced that same byte-identical clip and reported success, `2.5`
and `inf` surfaced only as "the encoder wrote no clip", and `"8"` leaked a raw `TypeError` out
of the writer's comparison. The verdict for one call depended on an interpreter flag - the same
reason `quality` is not left to the writer's own assert either.

In the other direction, `8.0`, `np.int64(8)` and `np.float64(8.0)` each name the same block size
as the `8` that encodes, and the writer's `isinstance(..., int)` gate refused all three.

The knob now resolves through the shared `positive_whole_number_error` beside the existing `fps`
guard, so a value refused when it names a frame rate cannot be accepted when it names the
rounding of that same clip, and it is passed to the writer as `int(macro_block_size)` exactly as
`int(fps)` and `float(quality)` beside it already are. The guard runs before the optional-encoder
probe, so the same mistake reports identically whether or not `imageio` is installed, and it
applies to both containers: GIF has no macro blocks, but a call must not become valid by changing
an extension.

A block size that is a positive whole number and that the codec nonetheless refuses - `7` rounds
a 48x32 frame to 49x35, which libx264 will not encode - stays the post-encode `RuntimeError` it
already was. This function cannot know which rounded sizes a codec takes, so the domain applies
no ceiling of its own and that boundary is pinned either way.
