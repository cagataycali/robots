### Fixed: the hardware agent tool takes `policy_config`, so an agent can name a checkpoint on a real arm

`Robot(mode="real")`'s agent tool exposed `instruction`, `policy_provider`,
`policy_host`, `policy_port` and `duration`, and `stream` dispatched exactly
those five. The `**policy_kwargs` that `start_task` and `_execute_task_sync`
accept, and that the mesh `execute` RPC already fills, were unreachable from the
tool, while its own description told the agent that `lerobot_local` needs
`pretrained_name_or_path`. A Hugging Face checkpoint could drive a simulated arm
(the sim tool takes `policy_config`) and a real arm over the mesh, but not the
real arm in front of the agent.

The tool now takes `policy_config`, the same bag the sim tool takes, forwards it
to the dispatcher, and judges it first: a value that cannot be splatted is
refused with `policies.policy_mapping_error`'s sentence, `host`/`port` inside
the bag are refused (they are `policy_host`/`policy_port`, and the approval
prompt must describe the server the arm will dial), and a checkpoint provider
named without its checkpoint is refused by the registry's `requires` before the
operator is asked or the arm is energized. The approval prompt names the
checkpoint (`checkpoint pretrained_name_or_path lerobot/smolvla_base`), so two
models no longer read identically to the operator approving one of them.
