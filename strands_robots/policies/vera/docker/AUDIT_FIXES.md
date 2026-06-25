# Docker scaffold — audit & required fixes

Audited the docker scaffold against the **actual** VERA source
(`vera/server/start_server_*.py`) and the **actual** downloaded checkpoint
layout (`VERA/vera-ckpts/`). Verdict: architecture is correct (Docker isolation
is the right call), and `DockerServerRunner`/`make_server_runner`/`server_mode`
are already implemented. But there are **two real blockers for MimicGen** and a
couple of nits. PushT path is fine as-is.

---

## ✅ Verified correct
- **PushT wiring works.** Entrypoint sets `VERA_PUSHT_PLANNER_CKPT` +
  `VERA_PUSHT_DYNAMICS_CKPT` — these match `start_server_pusht.py` (lines 36–42)
  exactly. Files exist on disk: `pusht-dfot/model.ckpt`, `pusht-idm/model.ckpt`. ✅
  PushT is fully local (no wandb, no WAN base) → container runs offline. Ship it.
- NGC base, EGL headless, bash-array argv, port defaults, TCP healthcheck,
  `_set_if_exists` override precedence, list-arg docker runner. All good.

---

## 🔴 BLOCKER 1 — MimicGen needs the Wan2.1 base, which is NOT mounted

`start_server_mimicgen.py` loads the WAN planner via `algo_config.yaml`, which is
derived from `vera/configurations/algorithm/wan_t2v.yaml`. That yaml hardcodes:

    ckpt_path: /path/to/flow-planner/data/ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
    ckpt_path: /path/to/flow-planner/data/ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
    ckpt_path: /path/to/flow-planner/data/ckpts/Wan2.1-T2V-1.3B            # CLIP/dir

The frozen **Wan2.1-T2V-1.3B base (text-enc + VAE + CLIP)** is a SEPARATE upstream
download (`Wan-AI/Wan2.1-T2V-1.3B`) — it is **NOT** in `sizhe-li/VERA` `vera-ckpts/`.
The README's own MimicGen recipe requires BOTH:

    export VERA_WAN_CKPT_ROOT=/path/to/Wan2.1-T2V-1.3B
    export VERA_MIMICGEN_CKPT_DIR=./vera-ckpts/mimicgen-wan-1.3b

The container today mounts only `/ckpts` and never sets `VERA_WAN_CKPT_ROOT`, so
**MimicGen will fail to load the planner** (or read the bogus `/path/to/...` paths).

### Fix
1. **Mount the Wan base** as a second volume and expose it via env, e.g.:
   - Dockerfile: `ENV VERA_WAN_CKPT_ROOT=/wan`
   - compose: add `- ${VERA_WAN_CKPT_ROOT:?...}:/wan:ro` volume.
   - docker run docs: `-v $WAN_BASE:/wan:ro`.
2. **Entrypoint (mimicgen branch):** the algo_config still points at the hardcoded
   `/path/to/flow-planner/...` paths. You must make the WAN base discoverable. Two
   options — pick whichever VERA actually honors (VERIFY by reading
   `vera/video_model/algorithms/wan/wan_t2v.py` + how `algo_config.yaml` overrides
   `ckpt_path`):
   - (a) If VERA reads `VERA_WAN_CKPT_ROOT` to override the yaml ckpt_paths →
     just `export VERA_WAN_CKPT_ROOT=/wan` in the mimicgen branch. (Check: grep
     showed NO code reads `VERA_WAN_CKPT_ROOT` — it may only be referenced by the
     README/algo_config templating. CONFIRM before relying on it.)
   - (b) If not, render a patched `algo_config.yaml` at container start (envsubst
     or a tiny python shim) that rewrites the three `Wan2.1-T2V-1.3B` ckpt_path
     prefixes to `/wan`, then pass `--algo-config /tmp/algo_config.patched.yaml`.
3. Update the entrypoint to also set `VERA_MIMICGEN_CKPT_DIR=${CKPT_ROOT}/mimicgen-wan-1.3b`
   (the README's MimicGen recipe sets it; the specialist DiT + flow_decoder live there:
   `mimicgen-wan-1.3b/video_model.dit_bf16.ckpt` + `flow_decoder.ckpt` — both present). ✅ files exist.

---

## 🔴 BLOCKER 2 — MimicGen IDM run-id vs on-disk dir mismatch (offline resolve)

Entrypoint sets `VERA_DYNAMICS_RUN_ID=x21o0cwe`. But:
- VERA's own default is also `x21o0cwe` (`start_server_mimicgen.py:48`), AND
- the loader comment (lines 13–17) says the IDM is resolved by
  `motion_policy_loading.load_checkpoint`, which is **wandb-run-id based**.
- The DOWNLOADED dir is `vera-ckpts/idm-mimicgen-37oa162u/` (run id `37oa162u`),
  with a `config.yaml` sidecar — plus a second `vera-ckpts/idm-mimicgen/model.ckpt`.

So `x21o0cwe` (the code default) ≠ `37oa162u` (the hosted artifact in the README's
checkpoint table: `idm-mimicgen-37oa162u/`). Inside an OFFLINE container, resolving
a wandb run id will hit the network and fail.

### Fix
- Determine the correct **local** IDM path resolution. Likely set
  `VERA_DYNAMICS_RUN_ID=37oa162u` (matching the on-disk dir) AND ensure the loader
  is pointed at the LOCAL ckpt dir (an env like `VERA_DYNAMICS_CKPT_DIR` or a
  `--dynamics-ckpt-path` flag — VERIFY in `motion_policy_loading.load_checkpoint`
  how it finds a local run dir vs wandb). The README table lists
  `idm-mimicgen-37oa162u/` as THE MimicGen IDM artifact → trust that, not the
  `x21o0cwe` code default.
- Whatever the mechanism, the container must resolve the IDM **without wandb
  network access** (set `WANDB_MODE=offline` / `WANDB_DISABLED=true` as a belt-and-
  braces env in the Dockerfile, and point the loader at the local dir).

---

## 🟡 Nits
- **`flash-attn`**: Dockerfile comment says WAN falls back to SDPA — true, but the
  NGC 24.10 image may already ship flash-attn. Leave as-is; just don't `pip install`
  a conflicting flash-attn build.
- **`pip install -e ".[idm,video]"` then `".[eval]"`**: `eval` pulls
  robosuite/robomimic/mimicgen + mujoco. Only needed if you run the example sim
  rollouts *inside* the container. For pure policy-serving (the host runs the sim,
  the container only serves actions) `eval` is **not required** → drop it to slim
  the image, OR keep it only for a `--target with-sim` build stage. Decide based on
  whether the container ever runs VERA's own env_runner.
- **Healthcheck `retries: 40` @ 15s = 10 min** — fine for WAN cold load. PushT loads
  in seconds; consider per-embodiment override but not required.
- **README "host robots venv (numpy>=2)" diagram** is accurate and matches the
  isolation goal. 👍

---

## Suggested verification commands (run after fixes)

PushT (fully local, should Just Work):
    export VERA_CKPT_ROOT=/abs/vera-ckpts
    docker compose -f .../docker-compose.yml up      # VERA_EMBODIMENT=pusht default
    # host: create_policy("vera", embodiment="pusht", server_mode="docker", auto_launch_server=False)

MimicGen (after wiring WAN base + IDM):
    export VERA_CKPT_ROOT=/abs/vera-ckpts
    export VERA_WAN_CKPT_ROOT=/abs/Wan2.1-T2V-1.3B   # SEPARATE download
    VERA_EMBODIMENT=mimicgen docker compose ... up

Sanity (inside container) before serving:
    python -c "import vera, vera.server, vera.policy, vera.idm; print('vera ok')"
    ls -la /ckpts/mimicgen-wan-1.3b /wan                 # both must be populated

---

## Bottom line
- **PushT: ready.** Local ckpts, offline, correct env wiring.
- **MimicGen: blocked** on (1) mounting + wiring the Wan2.1-T2V-1.3B base, and
  (2) resolving the IDM ckpt locally (run-id `37oa162u` dir, no wandb network).
  Fix those two and verify with the `import vera` + `ls /ckpts /wan` checks above.
