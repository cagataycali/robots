---
description: Remote policy code - the trust_remote_code opt-in for HuggingFace checkpoints and the constrained GR00T inference container surface.
---

# Remote policy code

## HuggingFace policy code execution (`trust_remote_code`)

Some policy providers load models from the HuggingFace Hub with `trust_remote_code=True`, which instructs the HuggingFace libraries to download and execute Python code from the model repository with the privileges of the process running the agent. A malicious or compromised repository can therefore read your credentials, open a reverse shell, or command your robot simply by being loaded.

Because this is code execution, not data loading, an explicit opt-in is required before such a provider will load:

- The providers `lerobot_local` (`LerobotLocalPolicy`) and `kimodo` (`KimodoPolicy`) are on the remote-code list. Any provider that loads models with `trust_remote_code=True` must be listed in `_HF_REMOTE_CODE_PROVIDERS` so the opt-in is enforced.
- Loading is blocked by default: creating a gated provider without opting in raises `UntrustedRemoteCodeError` with an explanation rather than silently executing remote code.
- To opt in, set `STRANDS_TRUST_REMOTE_CODE=1` (`1` / `true` / `yes` are accepted). The example CLI enforces the same gate before it will run `--policy lerobot_local`.
- The dashboard writes that variable too: `runtime.trust_remote_code` in `settings.json` is published to `STRANDS_TRUST_REMOTE_CODE` by `dashboard.settings.apply_mesh_env`. A value that is not a boolean is read as `false` rather than as the value that failed to be one, so the gate cannot be opened by a setting it could not read.

Operator guidance:

- Set `STRANDS_TRUST_REMOTE_CODE=1` only for checkpoints from organizations you trust - ideally your own, or a small vetted allowlist (`lerobot/`, `nvidia/`). The opt-in is a per-process, whole-environment switch: once set it trusts every model the process loads for the life of that process. Scope it to the specific command rather than a shell profile, and pin checkpoints to a known revision where the loader supports it.
- Where a provider exposes a per-call `trust_remote_code` setting, use it. `KimodoPolicy` takes `trust_remote_code` (default `False`), so a process that has opted in to the provider can still refuse a given repository's code: the environment variable decides whether the provider may be built, the setting decides whether a checkpoint's code runs.
- Prefer providers that need no remote code. The default Mock policy, the GR00T container path and many LeRobot policy families do not need this flag.
- A mesh peer can request a model load too. A `pretrained_name_or_path` forwarded in an `execute`/`start` command is additionally constrained to an org allowlist (`STRANDS_MESH_HF_REPO_ALLOW`, default `nvidia,huggingface,lerobot`), so an authenticated peer cannot steer a robot into loading an arbitrary repo. Keep that allowlist as narrow as your fleet allows; it is independent of the per-process opt-in and both gates apply.

Reference: `strands_robots.policies.factory` (`_check_trust_remote_code`, `UntrustedRemoteCodeError`).

## GR00T inference containers

The `gr00t_inference` tool pulls a Docker image, downloads a checkpoint and starts a container. The agent-facing surface is intentionally constrained, and you should keep it that way:

- The agent cannot choose the image, bind-mount host paths or inject a container command - those are operator-config-driven only. The image is resolved from `STRANDS_GR00T_IMAGE` and checked against an allowlist (`STRANDS_GR00T_IMAGE_ALLOW`), and a guard blocks dangerous bind mounts (`/`, `/etc`, the Docker socket, `/proc`, `/sys`, credential directories) that would amount to host takeover.
- Keep `STRANDS_GR00T_IMAGE_ALLOW` and `STRANDS_GR00T_REPO_URL_ALLOW` narrow and exact; repo URLs are matched exactly, never by wildcard, so a look-alike repo (`...Isaac-GR00T-evil`) cannot slip past.
- Running the container still grants it a GPU and network. Run inference hosts with least privilege, on isolated networks where practical, and tear containers down when done (`gr00t_inference(action="stop", ...)` or `lifecycle="teardown"`).

Reference: `strands_robots.tools.gr00t_inference`.
