### Fixed: a GR00T grouped vector composes from a robot's per-joint keys

GR00T embodiments declare grouped modality keys - `state.single_arm` is five
joints wide, `action.single_arm` five columns - while a robot publishes one
reading per joint (`'1'` .. `'6'` from a MuJoCo SO-101). `observation_mapping`
and `action_mapping` are 1:1 name maps, so neither direction could be bridged
and both failed silently: several robot keys naming `state.single_arm` sent a
one-wide vector holding whichever reading dict order reached last, and one
actuator mapped to `action.single_arm` was commanded with the whole five-wide
row while the other four joints were never addressed. Against a live
Isaac-GR00T server the only expressible binding for an SO-101 was refused and
no action was applied. A model key may now name one slot of the model's own
vector - `state.single_arm[3]`, `action.single_arm[3]` - in either direction and
either inference mode; a mapping with a gap, a repeated slot, a reading the
observation lacks, or a slot the chunk does not carry is refused by name rather
than honoured as something narrower, since a zero component is
indistinguishable from a real reading. Towards #3818.
