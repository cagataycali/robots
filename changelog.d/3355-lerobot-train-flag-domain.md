### Bug Fixes

- **tools**: `lerobot_train` refuses a training posture that is not a boolean instead of reading
  it by truthiness. Five flags decide the `lerobot-train` argv - `resume`, `lora`,
  `train_expert_only`, `gradient_checkpointing` and `push_to_hub` - and each selects a posture
  rather than scaling a quantity, so every non-empty string a caller reaches for when opting out
  was truthy and selected the affirmative. Measured on `12dc48d` with `policy_type="act"` unless
  noted: `resume="false"` built `--config_path=<ckpt> --resume=true` and returned, so the fresh
  run that was asked for became a resume of a previous config and `--policy.device`, `--steps`,
  `--batch_size` and `--save_freq` were all dropped from the argv; `lora="false"` emitted
  `--peft.method_type=LORA`; `gradient_checkpointing="false"` on `pi0` emitted
  `--policy.gradient_checkpointing=true`; and `push_to_hub="no"` emitted
  `--policy.push_to_hub=no`, a token only lerobot's own parser inside the detached process can
  refuse. The remaining three answered with a refusal whose remedy the caller had already
  followed and so could not act on: `lora="false", train_expert_only="false"` raised "lora and
  train_expert_only are mutually exclusive ... Pick one fine-tuning strategy" at a caller who
  picked neither, `train_expert_only="false"` raised "only valid for ['pi0', 'pi05', 'smolvla']
  policies, not 'act'", and `gradient_checkpointing="false"` on `act` raised "drop
  gradient_checkpointing=". Nothing reported the silent postures: the argv goes to a detached
  process, the tool answers `status="success"` with a pid and a log path, and lerobot parses each
  of those argvs without complaint. The five are now held to the shared `boolean_flag_error`
  domain the sibling `build_lerobot_command` already applies to its own argv postures, checked
  before the pair below them is judged so the message names the flag rather than blaming the
  combination two unusable values happen to spell. The tool refuses them too, ahead of the
  operator approval gate it consults for `push_to_hub`: an opt-out spelled `"false"` reached that
  gate as `{"policy.push_to_hub": "false"}` and asked a human to approve a Hub publication nobody
  requested. Nothing is coerced, and all 33 valid boolean combinations build a byte-identical
  argv, numpy booleans included.
