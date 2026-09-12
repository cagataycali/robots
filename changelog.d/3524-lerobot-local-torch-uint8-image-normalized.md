### Fixed: a `torch.uint8` camera frame is normalized like its numpy twin

`LerobotLocalPolicy`'s no-processor batch builder documents "Images (HWC uint8)
-> CHW float32 [0, 1]" and applied it only to `np.ndarray` frames. A caller who
handed the same frame as a `torch.uint8` tensor on an `observation.*` key - the
natural form for a torch-native integration, reachable with the documented
`use_processor=False` - got it laid out channel-first and batched but left as
Byte in [0, 255], 255x the scale the numpy path produces. The frame then reached
the model unscaled, where lerobot's image resize raised
`"upsample_bilinear2d_out_frame" not implemented for 'Byte'` - a message naming
neither the observation key nor the missing conversion. The builder now keys on
`torch.uint8` exactly as it keys on `np.uint8`, matching `_canonicalize_obs_images`
(the sibling converter used when a processor pipeline IS loaded), so the two
sides of that branch scale a frame identically. Already-scaled float frames and
non-image `uint8` entries are untouched.
