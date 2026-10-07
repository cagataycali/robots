### Fixed: the microduck docs name `velstand.onnx`, Pollen's default walk

Pollen's `pollen-robotics/microduck-policies` ships ten ONNX weights, and its
`manifest.json` makes `velstand.onnx` the default walk since v5: one network
that walks under a twist and stands still at zero command. The docs listed the
other nine and pointed every sketch at the older `alpha_walking` /
`alpha_stand` pair. The policy page now lists `velstand`, uses it in both
sketches and the scene table, and the Microduck robot page records a measured
sim rollout of it in place of the `alpha_stand` row.
