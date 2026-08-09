### Fixed: agent tools no longer expose parameters the model cannot learn anything about

A `@tool` function's input schema is derived from its docstring, and the decorator
substitutes the placeholder `"Parameter <name>"` for any parameter it cannot find in
the `Args:` section. Thirteen parameters across `gr00t_inference`, `lerobot_teleoperate`
and `train_policy` reached the model that way, including `remove_volumes` (which
discards downloaded checkpoints), `lifecycle` (whose `"teardown"` phase removes the
container) and `hf_repo` (required for `download_checkpoint`). Ten of the thirteen were
described in the source already: seven sat under a `Container lifecycle args` header the
parser discards - along with the paragraph explaining why the build repo and image are
operator-configured rather than agent parameters - and three shared one
`lora_r / lora_alpha / lora_target_modules:` entry, which is read as a single parameter
named `"lora_r / lora_alpha / lora_target_modules"` and so described none of them.

Every parameter of all sixteen bound tools now describes itself, the operator-configured
note moved ahead of `Args:` so it reaches the tool description too, and a guard checks
both directions of the rule: no exposed parameter may carry the placeholder, and no
docstring entry may name a parameter the tool does not have.
