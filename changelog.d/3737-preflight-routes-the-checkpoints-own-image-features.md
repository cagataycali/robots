### Fixed

- `policies/lerobot_local`: the camera pre-flight check now compares an
  embodiment's image rename targets against the features the checkpoint itself
  declares, read from its `config.json` before the weight download.

  An embodiment's targets are only the model's declared features where the
  feature set is *built* from the embodiment (the MolmoAct2 path). A pretrained
  checkpoint records its own `input_features`: `embodiment="so101"` feeds
  `observation.images.image` + `.../wrist_image` while `lerobot/smolvla_base`
  declares `observation.images.camera1..3`. The refusal named the embodiment's
  targets as "the model's image feature(s)", and neither remedy it printed
  worked - a `camera_key_map` onto the named key is refused by
  `_resolve_camera_targets` ("the policy does not declare it"), and renaming
  cameras to the embodiment's source keys passed pre-flight only to have
  `EmbodimentMap.validate` reject the same rename after the download, discarding
  the whole processor pipeline including the embodiment's `state_units` /
  `action_units` conversion, so a MuJoCo state in radians reached a
  degrees-trained checkpoint unconverted under a successful-looking run.

  The mismatch is now reported up front, naming both sides and printing the
  `obs_rename_override` that routes the features the checkpoint does declare.
