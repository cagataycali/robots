### Fixed: the real robot's tool description tells the agent what `execute` / `start` do

It now says the two actions pause for operator approval before the arm moves
(and how a headless script pre-approves), that they run for at most
`duration` seconds (default 30), that the default `groot` provider needs
`policy_port` while `mock` and `lerobot_local` build in process, and that
`mock` ignores the instruction. Before, an agent asked to "wave the arm for
3 seconds" learned each of those from a refusal - or from the operator.
