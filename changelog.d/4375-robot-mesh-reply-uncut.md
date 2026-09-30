### Fixed: robot_mesh hands the agent a peer's whole reply

`tell`, `send`, `stop` and `ping` return the peer's envelope as a `json` content block with no size cap, next to a text label naming the verb and the target. The reply used to be cut at 600 characters of text, so a rollout's metrics arrived as broken JSON. (#4172)
