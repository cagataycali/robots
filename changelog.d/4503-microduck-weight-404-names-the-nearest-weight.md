### Fixed: a Microduck weight the Hub does not carry names the one it does

`MicroduckPolicy(onnx_path="alpha_walkinng.onnx")` failed with the Hub's bare
`404 Entry Not Found` and nothing else, so a one-letter typo sent the reader to
the docs to find the nine-plus names the repository ships. A failed fetch now
lists `pollen-robotics/microduck-policies` and appends
`Did you mean: 'alpha_walkinng.onnx' -> 'alpha_walking.onnx'?` for a near miss
(the same `strands_robots.utils.did_you_mean` clause the robot and teleoperator
refusals use), or the `.onnx` files the repository carries for a far one. The
Hub's own cause stays first; when the listing itself fails (offline, no access)
the message is unchanged.
