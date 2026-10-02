---
description: Every environment variable the package reads, with its reader module and default, generated at build time.
---

# Configuration

Every environment variable the package reads, generated from the source at build time: the variable behind a behaviour you want to change, the module that reads it, and what the code assumes when it is unset.

There is no config file. Behaviour is set three ways, in precedence order:

1. Keyword arguments on the call: `Robot("so101", mode="real", driver="strands", port=...)`.
2. Environment variables, listed below. A variable named in the `grant` column of [refusal codes](refusal-codes.md) lifts one refusal; the `STRANDS_*_COMMAND_ALLOW` family pre-approves commands for the operator gate.
3. Defaults in the code, shown in the `default` column when the read passes a literal; `unset` means the code handles the missing variable itself, in the module named under `read in`.

Two rules apply everywhere: a boolean variable accepts `1`, `true`, `yes` (case-insensitive) and treats anything else as off; a variable that names a security posture (`STRANDS_MESH_AUTH_MODE=none`, `DEVICE_CONNECT_ALLOW_INSECURE`, `BYPASS_TOOL_CONSENT`) is read once and logged at WARNING when it weakens the default.

The `meaning` column is the first sentence of the reading module's docstring or comment naming the variable, else the module itself.

{{env_vars}}

### CA Pin Rotation Runbook

The AWS IoT transport pins the SHA-256 of the Amazon Root CA1 PEM; the accepted set is a collection, so old and new pins can be valid at once. When AWS rotates the root, every fleet member refuses the new certificate until a pin covering it is accepted; deleting the cached PEM only re-downloads the same unpinned bytes.

Recompute the pin the URL serves: `python -c "import hashlib, urllib.request as u; print(hashlib.sha256(u.urlopen('https://www.amazontrust.com/repository/AmazonRootCA1.pem').read()).hexdigest())"`.

Planned rotation: verify the new certificate out of band (a digest from the connection that served the bytes proves nothing); ship a release that adds the new pin and keeps the old one; wait for fleet uptake, bounded by the slowest member; drop the old pin in a follow-up release. Emergency: stage the verified pin in `STRANDS_MESH_CA_PINS` (comma-separated 64-char lowercase hex, additive, invalid entries skipped with a warning) and remove the override once the release is deployed. `STRANDS_MESH_DISABLE_CA_PIN` is not part of this procedure: it turns the check off and marks the result unverified-origin, a break-glass for a broken pin, never the answer to a rotation.
