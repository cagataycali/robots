### Fixed: the security reference names the IoT credentials the transport reads

`docs/security.md`'s cross-network fleet section is where an operator configures the AWS IoT Core
path. It told the reader that the IoT device certificates and provisioning material are production
secrets and to provision, scope and rotate them, and it never named a variable. All four the
transport reads -- `STRANDS_IOT_THING_NAME`, `STRANDS_IOT_ENDPOINT`, `STRANDS_IOT_CERT_DIR` and
`STRANDS_IOT_CA_FILE` -- appeared nowhere in `docs/` or `README.md`, while
`mesh.iot.provision.ProvisionedThing.env_vars` hands the first three back to the operator after
provisioning and both the `iot` and `bridge` backends construct `IotMqttTransport` with no
arguments, so those variables are the whole of its configuration.

The failure mode is a silent one by design: with a credential unset, `connect()` logs at ERROR and
returns `False` so the mesh stays off rather than crash the host, which means the symptom is a peer
that never appears on the fleet. The names were reachable only one refusal at a time -- unset gives
`STRANDS_IOT_THING_NAME is required for IoT transport`, setting that gives
`STRANDS_IOT_ENDPOINT is required for IoT transport`, and setting both reports the certificate path
that `STRANDS_IOT_CERT_DIR` resolves, along with the three filenames it must contain.

The section now documents the four as bullets, the form this page already uses for every other
variable, and states which two are required and what the other two default to. The guard derives
the credential set from the transport's own literal environment reads and from the provisioner's
export list rather than from a list kept beside them, so a variable added to either surface is
graded on arrival; it also holds them to one section, because they configure a single channel and a
reader who finds only some of them configures half a connection.
