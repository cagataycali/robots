### Fixed: an Isaac Lab export with a clipped action term keeps its deploy contract

Isaac Lab's IO descriptor writes an action term's `clip` as one `[low, high]`
pair per joint (`self._clip[0].tolist()`). `contract_from_io_descriptors` accepted
only a joint-name mapping or a single pair, so every real clipped run was refused
as `unrecognised action clip` and exported with `deploy_contract_missing`, the one
case where the clip matters. The per-joint shape is read now; each joint is bounded
by its own pair after the affine, as `JointAction.process_actions` does.
