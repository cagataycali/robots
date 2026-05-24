# `rogue_08_audit_log_tamperer`

**AV-21..AV-25 — audit log post-incident integrity.**

## Threat narrative

The audit log is the forensic trail for emergency stops, command
rejections, and resume attempts. After an incident, the operator's
first question is *what actually fired and when*. An attacker who
remained on-host post-compromise will try to delete their tracks.

Four primary scrubbing techniques:

1. **Tamper a record** (edit a `payload.i` field). HMAC over the
   record body breaks, `verify_audit_integrity` reports the line.
2. **Delete a record** (cause a gap in `seq`). Per-peer monotonic
   `seq` is published with each record; gaps are reported.
3. **Append an unsigned record** when the rest of the log is signed
   (mixed-mode degradation). Reported.
4. **Rotate the PSK** mid-stream so the verifier (with the old PSK)
   cannot validate new records. Reported via broken-sig OR
   PSK-fingerprint change tracking.

## Defences in scope

* `audit.log_safety_event` — each record carries an HMAC
  (`STRANDS_MESH_AUDIT_PSK`) and a per-peer monotonic `seq`.
* `audit.verify_audit_integrity` — walks the log, returns dict
  with `broken_signatures`, `sequence_gaps`, `unsigned_records`,
  `psk_fingerprint_changes`.
* O_NOFOLLOW + bounded-read at writer (cf. F2 of audit hardening)
  — already covered by `pentest_mesh.py::av_21_audit_symlink`;
  this rogue focuses on post-write detection.

## What this rogue does

1. Writes 4 legit events under `psk-A`.
2. Tampers record 1 → verify reports broken sig.
3. Resets, writes 4 events, deletes record 1 → verify reports gap.
4. Resets, writes 2 events, appends unsigned record → verify reports.
5. Resets, writes 2 events under `psk-A`, switches to `psk-B`,
   writes another event, verifies under `psk-A` again → reports.

## Pass criterion

All four sub-attacks are reported (non-empty diagnostic lists).

## Cross-references

* Unit-style: `pentest_mesh.py::av_22_audit_tamper_detected`,
  `av_23_audit_seq_rollback`, `av_24_psk_degrade_signed_to_unsigned`,
  `av_25_psk_rotation_detected`.
* Source: `audit.log_safety_event`, `audit.verify_audit_integrity`.
