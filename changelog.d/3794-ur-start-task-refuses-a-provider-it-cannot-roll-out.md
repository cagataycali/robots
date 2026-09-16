### Fixed: the UR arm refuses a policy it cannot roll out before the rollout starts

`URDriver.start_task` is the fleet's only driver that builds a policy from the
provider registry, and it validated none of the keywords that registry names as
required. `policy_provider="lerobot_local"` with no `pretrained_name_or_path`
answered "started" and then died at `steps: 0` with "No model loaded"; `groot`
with no `policy_port` answered "started" and then held the arm at `steps: 0` for
~15 s of a 2 s budget before a `ConnectionError` to a default port nobody
serves. Both are now refused before the policy is built, naming the missing
keyword and a hint for it - the check the real-arm surface gained in #3752, now
shared by both surfaces so the requirements cannot diverge between them.
