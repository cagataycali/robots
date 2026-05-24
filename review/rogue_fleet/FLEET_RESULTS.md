# Rogue Fleet Results

**Run at**: 2026-05-23 23:46:11 EDT
**Total**: 14/14 defences held

| Rogue | AV | Title | Posture | Held | Notes |
|---|---|---|---|---|---|
| `rogue_01_no_cert_outsider` | AV-01 | No-cert outsider cannot publish on a tls-only fleet bus | victim: mTLS + tls-only listen; rogue: no certs, plain-TCP c | ✅ | plain-TCP put issued; victim listen is tls-only so the link never established; sample dropped at routing layer |
| `rogue_02_rogue_ca_insider` | AV-02 | Rogue-CA insider rejected at TLS verify | victim trusts CA-A only; rogue presents leaf signed by rogue | ✅ | TLS handshake refused: ZError: Unsupported protocol: tcp. Supported protocols are: [Tls] at /root/.cargo/git/checkouts/z |
| `rogue_03_namespace_hopper` | AV-03 | Different-namespace publisher cannot hop into the victim's fleet | victim namespace=strands; rogue namespace=other; same CA | ✅ | namespace='other' isolated from 'strands'; saw 0 samples from victim ns |
| `rogue_04_jumbo_dos_publisher` | AV-04 | Jumbo cmd frame dropped at receiver-side low_pass_filter | victim cap=512B; rogue tries small (50B) then jumbo (32 KiB) | ✅ | published: 1 small, 1 jumbo (32 KiB), 1 status probe; replies seen=0 (>=0 means victim alive; cap drops jumbo silently a |
| `rogue_05_safety_replay_attacker` | AV-26+27 | Safety estop replay + peer_id permutation blocked by t-keyed cache | in-process Mesh; receiver-side _estop_replay_cache active | ✅ | replay_blocked=True permutation_blocked=True legit_engaged=True |
| `rogue_06_safety_envelope_forger` | AV-28+29+30+31 | Estop envelope freshness/shape gates reject 4 forgery variants | in-process Mesh; testing missing/stale/forward t + missing p | ✅ | missing_t_blocked=True; stale_t_blocked=True; forward_skew_blocked=True; missing_peer_id_blocked=True; empty_peer_id_blo |
| `rogue_07_acl_role_violator` | AV-06+07+08 | ACL loader rejects enabled:false / empty interfaces; accepts CN-only | in-process; ACL file loader under test | ✅ | enabled_false_raised=True; empty_interfaces_raised=True; cn_only_accepted=True; permissive_shape_detected=True |
| `rogue_08_audit_log_tamperer` | AV-21+22+23+24+25 | Audit log tamper detection holds across 4 attack variants | in-process; isolated audit dir; PSK-signed log | ✅ | hmac_tamper_detected=True; seq_gap_detected=True; unsigned_degrade_detected=True; psk_rotation_detected=True |
| `rogue_09_command_payload_fuzzer` | AV-09+10+11+12+13+14+15 | validate_command rejects 7 payload forgeries | in-process; security.validate_command direct call | ✅ | long_instruction=True; hostile_policy_host=True; hf_path_traversal=True; hf_unallowed_org=True; long_duration=True; non_ |
| `rogue_10_policy_host_pivot` | AV-10+33 | is_safe_policy_host and is_safe_server_address parse correctly | in-process; CIDR + hostname + IPv6 inputs | ✅ | loopback_allowed=True; localhost_allowed=True; public_blocked=True; explicit_host_allowed=True; cidr_member_allowed=True |
| `rogue_11_resume_token_replay` | AV-RESUME-HMAC | Resume token: missing/bad HMAC + missing nonce all rejected | in-process; STRANDS_MESH_OVERRIDE_CODE='real-secret' | ✅ | fail_closed_no_local_code=True; missing_proof_blocked=True; wrong_hmac_blocked=True; missing_nonce_blocked=True |
| `rogue_12_response_hijacker` | AV-32 | Forged response on a P2P turn dropped at responder_id check | in-process; pending point-to-point send to robot R | ✅ | forged_response_dropped=True; legit_response_accepted=True; broadcast_accepts_any=True; unknown_turn_ignored=True |
| `rogue_13_safety_rate_flooder` | AV-SAFETY-RATE | Per-issuer fairness limits the novel-t flood from one peer | in-process; STRANDS_MESH_RESUME_REPLAY_CACHE_MAX=8 | ✅ | cache_max=8 flood_attempts=32 flooder_engaged=2 good_op_engaged=True flooder_throttled=True |
| `rogue_14_iot_bootstrap_mitm` | AV-IOT-CA-PIN | IoT bootstrap rejects rogue CA bytes; env-var bypass scoped | in-process; provision._verify_ca_bytes & existing-file branc | ✅ | pin_set_nonempty=True; pin_format_valid=True; rogue_bytes_rejected=True; rogue_hash_no_match=True; env_bypass_does_not_a |
