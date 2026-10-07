### Removed: the `microduck_walk` and `microduck_stand` policy shorthands

Both names resolved to the same `MicroduckPolicy` as `microduck` and pinned no
weight, so `create_policy("microduck_stand", onnx_path="alpha_walking.onnx")`
built a walking policy under a standing name, without a warning. The weight
file is what picks the skill. Use `create_policy("microduck",
onnx_path="alpha_stand.onnx")` (or `alpha_walking.onnx`); the old names are
refused with exactly that call. A `MicroduckPolicy` built without `onnx_path`
now says the weight file is the skill and names two shipped exports, instead of
pointing at the test-only `session` argument.
