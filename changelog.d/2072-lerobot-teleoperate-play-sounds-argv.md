### Fixed: `play_sounds` reaches the lerobot argv instead of being advertised and dropped

`lerobot_teleoperate` declared `play_sounds: bool = True`, documented it as
"Enable audio feedback", forwarded it to `build_lerobot_command`, and
`build_lerobot_command` declared it too - and then read it nowhere. No mode put
it on an argv, so the documented option did nothing at all in either position,
and a model could set it through the tool spec
(`{"type": "boolean", "default": true}`) and be told nothing. Measured on
`f9c05ba4`:

```python
build_lerobot_command(**BASE, play_sounds=True) == build_lerobot_command(**BASE, play_sounds=False)
# -> True
```

Where the flag belongs is decided by lerobot rather than by symmetry. Against
`lerobot==0.6.1`, the floor of this package's `[lerobot]` extra, three of the four
entry points this module drives declare the field on their top-level
`@parser.wrap()` config - `RecordConfig` (`lerobot_record.py:188`),
`ReplayConfig` (`lerobot_replay.py:99`) and `RolloutConfig`
(`rollout/configs.py:256`) - so it is spelled `--play_sounds` and not nested
under `--dataset.*`. It is now emitted for `record`, `replay` and `dagger`, and
each of those modes checks it against the shared `boolean_flag_error` domain, so
a string such as `"false"` is refused rather than read by truthiness.

Plain teleoperation is deliberately excluded: `TeleoperateConfig` declares no
such field and the entry point makes no `log_say` call, so there the flag would
be an unrecognized argument rather than an accepted no-op - the same failure the
removed pre-0.5 flat flags cause. It therefore emits nothing for `play_sounds`
and refuses no value for it, which is the per-mode scoping the numeric knobs
already use.

The flag is emitted unconditionally as an explicit `true`/`false` literal, like
`dataset_video`, rather than only when set, like `display_data`. That split
follows the upstream default rather than a style preference: lerobot defaults
`play_sounds` to `True`, so absence says `True` too, and the explicit literal is
the only spelling of the opt-out that can reach the CLI at all.

This supersedes the note in the previous entry that `replay` emits no boolean
flag: it now emits exactly one.

Both halves were needed. The tool spec sits on `lerobot_teleoperate` rather than
on `build_lerobot_command`, and the tool's `replay` dispatch built its argv
without forwarding `play_sounds` - so making the builder honor the flag left the
argv reading `--play_sounds true` on the only model-reachable path to replay, and
a non-boolean was not refused there either. The dispatch now forwards it, pinned
at the tool level, since a builder-level test cannot observe a dropped forward.
