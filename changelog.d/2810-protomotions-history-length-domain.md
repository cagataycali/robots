### Fixed: the ProtoMotions history window is held to a whole-number domain before it is normalised

`ProtoMotionsPolicy.__init__` takes `history_length` as the number of past action
frames that feed the `historical_processed_actions` ONNX input, guarded it with a
bare `history_length < 1` test, and then normalised it with
`int(history_length)` to shape the rolling buffer. That test covers the floor and
not the domain, so the spellings a config carries were laundered into a different
window than the one they name. `history_length` is one of the provider's
advertised `config_keys`, so it arrives from a JSON or YAML policy config as
readily as from a keyword. Measured through `create_policy("protomotions", ...)`,
reading the shape off the session the policy drives:

```
policy config              construction   window the tracker read
{"history_length": 3}      built          (1, 3, 29)
{"history_length": 3.0}    built          (1, 3, 29)
{"history_length": 2.7}    built          (1, 2, 29)   <- a two-frame window
{"history_length": true}   built          (1, 1, 29)   <- a one-frame window
```

`ProtoMotionsConfig.__post_init__` in the same package already resolves this
ordering the other way for its own body indices, and states why: each index goes
through the shared whole-number domain *before* the `int()` normalisation,
because that conversion is what laundered a yaml `anchor_body_index: true` into
row 1 and a `2.7` into row 2. `load_config_from_yaml` hands both indices through
raw for the same reason. The constructor now follows that order too, through
`positive_whole_number_error` - the same whole-number family the sibling uses,
with the positive floor a buffer dimension needs.

Every spelling the previous code honored still is, because a window length read
from a config is legitimately an integral float: `3`, `3.0`, `np.int64(4)` and
`np.float32(4.0)` all build and produce a byte-identical buffer, and `0`, a
negative count and `False` are still refused. What changes is that a value that
cannot be honored as a count is now named rather than coerced, and that a count
past any allocatable buffer is refused by the parameter's own domain instead of
reaching NumPy, which refuses it with a message naming neither the parameter nor
the caller.
