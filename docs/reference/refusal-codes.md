# Refusal codes

The package refuses before it acts when a request is well formed but not yet allowed, and a continuable refusal carries a stable code so a consumer can offer the operator the grant that lifts it. After this page you can match a refusal by identity instead of parsing its message, and you know which variable each grant sets.

```python
from strands_robots.refusal_codes import REFUSAL_CODES, REFUSAL_GRANTS

for code in REFUSAL_CODES:
    print(f"{code:28} lifted by {REFUSAL_GRANTS[code]}")
```

Only continuable refusals get a code. A refusal nothing can lift (an instruction over the length limit, an unknown joint name) stays a plain error with a message that names the valid set. The message text of a coded refusal may change between releases; the code and the `subject` attribute do not.

Three grants are allowlists the refusal's `subject` is appended to (`HF_REPO_NOT_ALLOWED`, `POLICY_TYPE_NOT_ALLOWED`, `POLICY_HOST_NOT_ALLOWED`). The other two are not: `STRANDS_TRUST_REMOTE_CODE` takes `1`, and `STRANDS_MESH_INPUT_VALUE_ABS` takes a bound larger than the refused magnitude. Applying a subject to those two is a silent no-op, so read the `meaning` column before wiring a consent flow.

{{refusals}}

The dashboard's consent endpoint maps each code to a grant name (`trust_remote_code`, `hf_repo_allow`, `policy_type_allow`, `policy_host_allow`, `teleop_degree_units`) in `strands_robots/dashboard/consent.py`; see [the dashboard page](../learn/dashboard.md).
