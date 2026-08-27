### Changed: the dashboard operator agent runs on `robot_mesh` alone

The dashboard agent carried two overlapping mesh tools: a dashboard-local
`fleet` gateway (`peers`/`task`/`stop`/`stop_all`/`status`) and the SDK-native
`robot_mesh`. Every `fleet` verb was already reachable through `robot_mesh` -
which carries its own human-in-the-loop interrupt on physical actions - so the
second tool only widened the surface an operator (and the model) had to reason
about. `fleet` is removed and the agent is built on `robot_mesh` plus the
per-peer proxy tools: discovery stays `robot_mesh(peers)` / `robot_mesh(status)`,
a policy rollout is `robot_mesh(send, {"action": "execute", ...})` or
`robot_mesh(tell, ...)`, stopping one robot is `robot_mesh(send, {"action":
"stop"})` and stopping the fleet is `robot_mesh(emergency_stop)`. The text-agent
and voice-operator system prompts are rewritten to be `robot_mesh`-first, the
`BidiAgent` voice tool list is pointed at `robot_mesh` (it was the last caller
of the now-removed `_make_fleet_tool`), and the advertised-tools badge drops
from `["fleet", "robot_mesh"]` to `["robot_mesh"]`. No capability is lost:
policy rollouts stay reachable via `robot_mesh` tell/send and stop paths are
never gated.
