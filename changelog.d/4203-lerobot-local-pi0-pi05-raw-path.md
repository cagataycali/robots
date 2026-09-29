### Fixed: `lerobot_local` runs pi0 and pi05 checkpoints on the raw obs/action path, and the agent tools name `embodiment`

Three faults met while running every public `lerobot_local` checkpoint end
to end on an L40S (`act`, `smolvla`, `molmoact2`, `pi0`, `pi05`):

- `lerobot/pi05_base` died at the first inference with `Preprocessor
  pipeline failed: 'numpy.ndarray' object has no attribute 'cpu'`. The
  heuristic (no-embodiment) remap handed the checkpoint's pipeline
  `observation.state` as an ndarray, while lerobot's own inference helper and
  the declarative `strands_pack_state` step hand it a tensor, and pi05's
  prepare-state step calls `.cpu()` on it. The remap now builds a float32
  tensor.
- `lerobot/pi0_base` with an embodiment the model refused (a 6-key map
  against its 32-wide action head, #4193) fell back to the raw flow and then
  died with `KeyError: 'observation.language.tokens'`: pi0/pi05 carry the
  PaliGemma tokenizer only as a pipeline step, their config names no
  `tokenizer_name`, so the fallback had nothing to tokenize with. The
  discarded pipeline now lends its tokenizer, max length and padding side to
  the raw flow before it is dropped, and the language-token check honours it.
- "State dim 6 < model expects 32 - zero-padding" was logged on every
  inference (40+ lines in a ten-second pi0 rollout). It is warn-once per
  policy now, like every sibling diagnostic.

The sim tool's and the hardware tool's `policy_config` descriptions now list
`embodiment` among the `lerobot_local` keys: it is the keyword every SO-arm
checkpoint needs and the one the preflight refusal names as the remedy, yet
neither schema advertised it, so an agent reading the schema alone could not
learn it before the first refusal. `tests/simulation/mujoco/test_policy_config_advertises_embodiment.py`
pins both schemas; `tests/policies/lerobot_local/test_discarded_pipeline_lends_its_tokenizer.py`
pins the three runtime fixes.
