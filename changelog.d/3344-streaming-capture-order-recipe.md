### Fixed: the streaming recipe documented as chronological reads shuffled frames

`StreamingDatasetReader`'s "in-process eval / replay" Example and the streaming
section of `docs/recording.md` both passed `shuffle=False` alone and commented it
`# chronological for replay/eval`. It is not chronological. `shuffle` reaches
lerobot's `StreamingLeRobotDataset`, where it decides only WHICH generator drives
the reordering -- a `default_rng(seed)` reseeded identically on every exhaustion,
or the dataset's own advancing one -- so it decides reproducibility *across
epochs*, which is what lerobot documents it as ("whether to shuffle the dataset
across exhaustions"). The reader reorders either way, at two levels: it samples a
shard at random per frame, and it yields from a reservoir buffer of
`buffer_size`.

Reading a 60-frame recorded dataset back through the documented recipe yields
frame indices `28, 55, 18, 24, 7, ...`, and `shuffle=True` alongside
`buffer_size=1` reads in capture order anyway -- so the flag is not the order
knob in either direction. Nothing reports a shuffled read: every frame is
delivered, only out of order, so an eval or replay loop that followed the
documented recipe silently consumed its episode scrambled while its own comment
said otherwise.

Capture order is `buffer_size=1` (a reservoir of one has nothing to reorder)
together with `max_num_shards=1` (a single shard has nothing to interleave), and
`examples/06_agent_collect_and_stream.py` already used both. All four documented
recipes now show that pair, `open()` carries an `Ordering` note stating what
`shuffle` does decide, and two rules over the package, guide and examples keep a
surface from claiming capture order -- or offering `shuffle=False` in its place --
without naming the knob that delivers it.
