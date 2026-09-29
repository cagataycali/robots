### Fixed: the native drivers' `status` verb answers with one envelope

feetech (and dynamixel through it), g1, go2, ur, franka and reachy wrapped the
envelope `get_status()` already returns in a second one, so the fields sat two
levels down and `content[0]["json"]["port"]` raised `KeyError`, while the other
five native drivers answered one level deep. All eleven now return
`get_status()`'s envelope itself, and a registry-driven test pins the depth for
the whole fleet. Closes #4151.
