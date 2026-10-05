### Fixed: an address the lerobot config would drop for its own default is refused

`Robot("unitree_g1", mode="real", driver="lerobot", port="10.0.0.5")` used to
build a `UnitreeG1Config` at its stock `robot_ip="192.168.123.164"`: `port` is
in the cross-robot allowlist, so a config without that field dropped it with a
DEBUG line and the call succeeded against the factory address. When the
caller's only address (`port`, `robot_ip`, `ip_address`, `remote_ip`, `host`)
is one the config does not declare, the call is now refused with a
`ValueError` naming the field the config is reached through (`robot_ip` here).
A call that names both, as a fleet-wide call does, still builds.
`Teleoperator(...)` applies the same rule to `port` and `ip_address`.
