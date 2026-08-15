### Fixed

- **policies/groot**: resolve model keys by name under either GR00T release key
  spelling. N1.6/N1.7 declare `ModalityConfig.modality_keys` bare (`"front"`)
  and N1.5 declares them prefixed (`"video.front"`); every consumer compared
  them as bare, so against an N1.5 checkpoint mapping resolution fell through to
  positional matching - pairing two identically named cameras by declaration
  order, so a wrist image could be sent under the model's front key with nothing
  reporting it. A correct explicit `observation_mapping` was also refused, and
  `strict_keys=True` raised for a key set that matches exactly by name. Both
  spellings now reduce to the bare name for comparison, and a resolved model key
  is stated in the spelling its model declares, leaving the emitted payload
  unchanged for every release.
