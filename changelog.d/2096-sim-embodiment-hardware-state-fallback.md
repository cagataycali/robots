### Quality: cover the sim-embodiment-on-hardware `observation.state` fallback

`PackStateProcessorStep.observation()` falls back to lerobot's `'<motor>.pos'`
keys when none of the embodiment's declared `state_keys` is present -- the
branch that makes `embodiment="so101"` bind a physically-attached arm and not
only the simulator. That branch had never executed under test: its predicate
`hardware_pos_keys` was exercised exhaustively while the consumer that packs
its result was not, so the motor-order binding, the raw (un-double-converted)
units, the key consumption and the declared-width reconciliation were all
unverified. Adds the tests that pin them.
