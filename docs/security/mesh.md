---
description: Mesh security - Zenoh authentication modes, mTLS material, operator ACLs, fleet namespace isolation, the AWS IoT Core transport, and the audit log.
---

# Mesh authentication

Joining the Zenoh peer mesh is opt-in: `Robot(name, mesh=True)` (or `STRANDS_MESH=true`) joins one, a `Simulation` built directly joins by being assigned a started client, and a bare `Robot()` exposes no mesh surface at all. Once joined, the `robot_mesh` tool lets an agent enumerate, command and broadcast to every peer, and `STRANDS_MESH_AUTH_MODE` governs the wire.

## Development posture (insecure)

The example scripts run the mesh without authentication or access controls so they work out of the box. Any device on the same network can then command the fleet: acceptable on a trusted, isolated development network, never on an untrusted one. It is selected one of two ways:

- `STRANDS_MESH_LOCAL_DEV=1` - the developer preset. It defaults the auth mode to `none` and satisfies the insecure-acknowledgment second factor by itself.
- `STRANDS_MESH_AUTH_MODE=none` together with `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1` - the explicit form. `none` on its own is rejected, so wire security cannot be disabled by setting a single variable.

!!! warning "Wire security off is a loud signal"
    With wire security off the SDK logs a loud error on every session open (`WIRE SECURITY DISABLED - STRANDS_MESH_AUTH_MODE=none`). Treat that line as a signal that the process must never be on a shared or hostile network.

## Production posture (required off trusted networks)

Off a trusted LAN, `STRANDS_MESH_AUTH_MODE=mtls` is required, and it is the default when neither dev flag is set. mTLS authenticates peers at the transport layer (`transport/link/tls`) before any command is dispatched. It is not sufficient on its own - pair it with an access-control list:

- The built-in default ACL is permissive: any CA-signed peer may publish and subscribe on any key, and with no ACL supplied the SDK warns on every session open.
- Supply an operator ACL via `STRANDS_MESH_ACL_FILE` enumerating each peer's certificate CN and the key expressions it may use - see [`mesh_acl_example.json5`](https://github.com/strands-labs/robots/blob/main/examples/mesh/mesh_acl_example.json5) (role-scoped) and [`mesh_acl_strict_per_peer.json5`](https://github.com/strands-labs/robots/blob/main/examples/mesh/mesh_acl_strict_per_peer.json5) (per-peer).
- `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1` is the acknowledgement token that lets a **blacklist-shaped** operator ACL load. See [Blacklist ACL acknowledgement](#blacklist-acl-acknowledgement-strands_mesh_accept_permissive_acl) below.
- An ACL file the loader cannot read is refused, not ignored. A missing, oversize, non-UTF-8, malformed or too-deeply-nested file is reported as unloadable, which the start-time gate treats as the permissive default and so refuses to bring the wire up, naming the path and the reason. A typo in the ACL stops the mesh rather than quietly widening it.

!!! danger "A WAN or cloud Zenoh router MUST deploy a topic-level ACL"
    mTLS gives identity; an ACL gives least privilege. With mTLS and no topic-level ACL, any one authenticated certificate can subscribe to `**` - every device's state, camera and input streams - and publish to any device's `cmd` topic, so one stolen certificate is read/write control of the whole fleet. Deploy a per-peer ACL on every internet-facing router, not only on LAN peers.

### Transport credentials (mTLS material)

Three filesystem paths are the whole of the Zenoh transport's TLS configuration under the default `mtls` mode, and they are required *together*: with any one unset, `_resolve_tls_paths` raises `ValueError` naming all three and the session never opens rather than silently downgrading to plain TCP. A fleet with no dev flag and no TLS material does not come up at all.

- `STRANDS_MESH_TLS_CA` - the CA bundle used to validate peer certificates. It is the trust root that decides which peers are in the fleet, so the ACL's CN pinning is only as good as the CA that issued those CNs.
- `STRANDS_MESH_TLS_CERT` - this peer's certificate (PEM). Its CN is what an operator ACL pins, so it must match the CN that ACL names.
- `STRANDS_MESH_TLS_KEY` - this peer's private key (PEM). On POSIX the loader enforces mode `0600` and refuses a more permissive key, because a `0644` key on a shared host is an exfiltration surface. On Windows the check is skipped - POSIX modes do not map onto NTFS ACLs - with a one-shot WARNING saying so, so restrict the key by ACL to the account that runs the peer.

None may be a symlink: the loader rejects a symlinked CA, certificate or key by path *before* reading the file, so a redirected link cannot swap the material out from under the mode check.

### Fleet routing isolation (namespace)

`STRANDS_MESH_NAMESPACE` is the Zenoh `namespace` field on every peer. It prefixes every mesh key expression - presence, safety, sensors, commands - and Zenoh routes only between peers whose namespaces match, so two fleets with different namespaces cannot exchange application traffic even when their key expressions collide. That is the property fleet isolation rests on when a test rig shares a LAN with production hardware.

- Optional; defaults to `strands`. It must be the *same* value on every peer of one fleet: mismatched peers still complete the TLS handshake and then exchange no application traffic, so the failure mode is silent - a peer that appears absent rather than one that raises. Provision it alongside the TLS material.
- Empty and whitespace-only values fall back to the default, because the alternative would be topics like `//presence` where the leading `/` is the missing namespace and a wildcard a permissive ACL admits could match against them.
- The default tracks the `strands/...` prefix every mesh component emits (`mesh.core`, `mesh.sensors`, `mesh.input`, the IoT path). Change it only across every peer at once - a rolling change leaves one half of a fleet unable to see the other.

Reference: `strands_robots.mesh._zenoh_config.resolve_namespace`.

### Blacklist ACL acknowledgement (`STRANDS_MESH_ACCEPT_PERMISSIVE_ACL`)

An operator ACL supplied via `STRANDS_MESH_ACL_FILE` can be written in one of two shapes, and one of them is a load-bearing anti-pattern:

- `default_permission: "deny"` **+ explicit `allow` rules** - a *whitelist*. A gap in the rule set silently denies rather than exposes.
- `default_permission: "allow"` **+ explicit `rules`** - a *blacklist*. A gap - a key expression nobody named - is silently open on the wire.

`_acl_config._load_acl_file` refuses the second shape at ACL load with a `PermissiveACLError` unless `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL` is set to `1`, `true` or `yes` (case-insensitive, whitespace-stripped); any other spelling is not an acknowledgement and nothing raises. The token is read in `_acl_config._parse_acl_bytes`, the step `_load_acl_file` validates the file's bytes in, so the refusal applies to every read of the file. Its message names the path, the rule count and both remediations: rewrite as `deny` plus `allow` rules, or set the token.

The token has two further effects on the built-in permissive default (`default_permission: "allow"` with no rules), a different posture reaching a different gate: under `mtls` it is refused by `Mesh._refuse_under_permissive_default_acl` and the token is what lets the wire come up, and the per-session `WARNING` from `session._build_config` that `STRANDS_MESH_ACL_FILE` is unset is suppressed by the same token. So a token set to load a blacklist ACL in CI also waives the start gate: if the ACL file is later dropped from that environment, the fleet runs wire-open with no log signal.

Do not set this variable on a production fleet. It does not narrow the ACL; it records that an operator accepted a posture where an unenumerated key expression is open on the wire. Supplying an ACL with `default_permission: "deny"` is the narrower way to silence the WARNING.

Reference: `strands_robots.mesh._acl_config.permissive_acl_acknowledged` (the only reader of the variable; all three gates call it), `strands_robots.mesh._acl_config._load_acl_file`, `strands_robots.mesh._acl_config._parse_acl_bytes`, `strands_robots.mesh._acl_config.PermissiveACLError`, `strands_robots.mesh.core.Mesh._refuse_under_permissive_default_acl`, `strands_robots.mesh.session._build_config`, `strands_robots.doctor.check_mesh`.

## Cross-network fleets (AWS IoT Core)

Two steps route traffic through AWS IoT Core (MQTT5 with mTLS): the `[mesh-iot]` extra installs the dependency, and `STRANDS_MESH_BACKEND=iot` selects the transport. The extra alone changes nothing - the fleet stays on Zenoh - so set both. `STRANDS_MESH_BACKEND=bridge` selects a `BridgeTransport` instead, which keeps high-rate topics local while bridging presence, health and safety to the cloud. The device certificates and provisioning material are production secrets: provision per device, scope each IoT policy to the minimum topic set, and rotate or revoke them like any other fleet credential.

Both backends construct that transport with no arguments, so four environment variables are the whole of its configuration - `ProvisionedThing.env_vars()` hands the first three back after provisioning. With either required variable unset, `connect()` logs at ERROR and returns `False`: the mesh stays off rather than crash the host, so the symptom is a peer that never appears on the fleet.

- `STRANDS_IOT_THING_NAME` - required. The AWS IoT Thing name, sent as the MQTT `client_id`, so it must match both the certificate's CN and the Thing your IoT policy authorises through `${iot:Connection.Thing.ThingName}`. The trust model ties it to the `Mesh` peer id, because a peer publishes under `strands/{peer}/...`.
- `STRANDS_IOT_ENDPOINT` - required. The account's ATS endpoint, e.g. `a2acz9p1ge6619-ats.iot.us-west-2.amazonaws.com`.
- `STRANDS_IOT_CERT_DIR` - optional; defaults to `~/.strands_robots/iot`. The directory holding `{thing}.cert.pem`, `{thing}.private.key` and `AmazonRootCA1.pem` - the production secrets named above. A missing file is reported by path.
- `STRANDS_IOT_CA_FILE` - optional; defaults to `AmazonRootCA1.pem` inside `STRANDS_IOT_CERT_DIR`. Overrides the root CA path.

Reference: `strands_robots.mesh.session`, `strands_robots.mesh._acl_config`, `strands_robots.mesh.transport.iot_transport`.

See also: the [audit log](audit-log.md), and [command approval](commands.md).
