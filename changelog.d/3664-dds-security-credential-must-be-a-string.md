### Fixed: a `dds_security_config` credential must be a string the participant can carry

`RosTelemetryBase._validate_dds_security_config` documented every required
credential as "a non-empty string" but graded `str(value).strip()`, and
`str(None)` is `"None"` - so `None`, `0`, `False`, `[]` and `{}` all passed the
check that exists to refuse a half-filled config. `HardwareRtpsBridge` then read
the same credentials by truthiness when building the participant QoS and set
nothing for those keys, so a config the validator had accepted reached
`DomainParticipant` with the DDS-Security auth plugin wired and no private key or
governance - while the inbound arm-driving command surface opened, because the
gate on it is satisfied by any non-empty dict. A truthy non-string degraded the
other way and just as quietly: a `bytes` path was carried stringified, naming the
literal path `b'file:/etc/dds/participant_key.pem'`.

Each credential in hand is now required to be a non-empty string, and the
refusal names the key and the type that arrived, so a path read out of a binary
or JSON config is rejected at construction instead of being attributed later to
an opaque participant error. An optional key is graded by the same domain,
because a key present is a key the QoS carries. The QoS in turn carries every
key present rather than every truthy one: grading happens once, in the validator,
so the participant can no longer drop a credential the validator accepted.
