### Fixed: the real robot's tool description tells the agent what `execute` / `start` do

It now says the two actions pause for operator approval before the arm moves
(and how a headless script pre-approves), that they run for at most
`duration` seconds (default 30), and what each provider it names still needs:
`groot` (the default) and `moveit2` need `policy_port`, `lerobot_local` builds
in process but needs `pretrained_name_or_path`, and `mock` needs nothing and
ignores the instruction. Before, an agent asked to "wave the arm for 3
seconds" learned each of those from a refusal - or from the operator.
