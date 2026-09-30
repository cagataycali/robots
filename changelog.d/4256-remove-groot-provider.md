### Removed: the hand-rolled `groot` policy provider, its ZMQ tool and the Isaac-GR00T trainer (BREAKING)

`strands_robots.policies.groot` (`Gr00tPolicy`, the ZMQ client, the N1.5 / N1.6 /
N1.7 loader shims and `data_configs.json`), `strands_robots.tools.gr00t_inference`
(the Docker lifecycle tool), `strands_robots.training.groot` (`Gr00tTrainer` over
Isaac-GR00T's `launch_finetune`), the `groot-service` extra and
`SUPPORTED_GROOT_VERSIONS` / `groot_version_error` are gone: about 6,600 lines of
source and 10,000 lines of tests that existed to keep three NVIDIA releases
loadable behind one class. GR00T N1.7 is lerobot 0.6's native `groot` policy type,
so `create_policy("lerobot_local", pretrained_name_or_path="nvidia/GR00T-N1.7-3B",
policy_type="groot")` loads it in process (new extra `strands-robots[groot]` =
`lerobot[groot]`), `create_trainer("lerobot_local")` with
`extra={"policy_type": "groot"}` fine-tunes it, and a GPU host elsewhere runs
`strands_robots.inference.server.PolicyServer` with the robot on the `remote`
provider. `nvidia/*` Hub ids now resolve to `lerobot_local`; `zmq://` is no longer a
declared URL scheme.

Breaking: `create_policy("groot")`, `create_trainer("groot")`, `resolve_policy("groot")`
and `policy_provider="groot"` on every robot, driver and mesh surface refuse with one
fixed sentence (`registry.policies.REMOVED_PROVIDERS`) and never reroute:
"policy_provider 'groot' was removed in 1.0: GR00T N1.7 runs through
lerobot_local(policy_type='groot'); for a remote GPU host run
strands_robots.inference.server.PolicyServer there and use policy_provider='remote'."
Every driver's `start_task` / `execute` default `policy_provider` moves from
`groot` to `lerobot_local`, so a call that names neither a provider nor a
checkpoint is refused on the missing `pretrained_name_or_path` instead of on a
missing port. The `groot` docs page is removed (its URLs redirect to
`learn/policies/lerobot-local`, which gains a "GR00T N1.7 through lerobot"
section); the site word ceiling drops 49,800 -> 49,108. The GR00T Whole-Body-Control
providers (`wbc`, `wbc_gait`) are unrelated ONNX locomotion and are unchanged.
