---
description: The stable code a continuable refusal carries and the grant that lifts it: consumers match by identity, not message.
---

# Refusal codes

The package refuses before it acts when a request is well formed but not yet allowed; a continuable refusal carries a stable code so a consumer can offer the operator the grant that lifts it. After this page you match a refusal by identity, not by parsing its message, and know which variable each grant sets.

```python
from strands_robots.refusal_codes import REFUSAL_CODES, REFUSAL_GRANTS

for code in REFUSAL_CODES:
    print(f"{code:28} lifted by {REFUSAL_GRANTS[code]}")
```

Only continuable refusals get a code; one nothing can lift (an instruction over the length limit, an unknown joint name) stays a plain error naming the valid set. A coded refusal's message may change between releases; its code and `subject` attribute do not.

Three grants are allowlists the refusal's `subject` is appended to (`HF_REPO_NOT_ALLOWED`, `POLICY_TYPE_NOT_ALLOWED`, `POLICY_HOST_NOT_ALLOWED`). The other two are not: `STRANDS_TRUST_REMOTE_CODE` takes `1`, `STRANDS_MESH_INPUT_VALUE_ABS` a bound above the refused magnitude; a subject applied to those is a silent no-op, so read the `meaning` column before wiring a consent flow.

{{refusals}}

The dashboard's consent endpoint maps each code to a grant name (`trust_remote_code`, `hf_repo_allow`, `policy_type_allow`, `policy_host_allow`, `teleop_degree_units`) in `strands_robots/dashboard/consent.py`; see [the dashboard page](../learn/dashboard.md).
