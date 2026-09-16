### Fixed: a checkpoint provider with no checkpoint is refused before the arm is energized

`start_task` / `execute` with `policy_provider="lerobot_local"` and no
`pretrained_name_or_path` used to answer "Task started", connect and
energize the arm, and fail at its first action with "No model loaded". The
registry now lists the checkpoint as required for `lerobot_local`, and both
entry points refuse a missing required keyword - naming it, with a hint -
before the motors bus is claimed, as they already did for a missing port.
The quickstart's real-arm step passes the checkpoint it trained.
