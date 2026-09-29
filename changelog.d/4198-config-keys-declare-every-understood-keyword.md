### Fixed: the registry declares every constructor keyword a provider understands

`config_keys` is what the dashboard's policy form and `build_policy_kwargs`
read, and it lagged the constructors: 13 `lerobot_local` keywords (`rtc_enabled`,
`camera_key_map`, `obs_rename_override`, `revision`, ...), 6 `groot` service
keywords (`observation_mapping`, `action_mapping`, `api_token`, ...),
`cosmos3.pretrained_name_or_path`, `moveit2.joint_name_map` and three `curobo`
keywords were bindable and documented but not declared, so the form could not
offer them and `build_policy_kwargs` dropped them. All 24 are declared, and a
reverse-direction guard with a reasoned exclusion list (injected objects, GR00T
local-mode keys removed in 0.7) keeps the two in step (#4198).
