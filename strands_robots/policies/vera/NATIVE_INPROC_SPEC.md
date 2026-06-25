# VERA in-process (native) policy — implementation spec

**Goal:** add an *optional* in-process execution path for the VERA policy that
calls `vera`'s two-stage `MotionPolicy` **directly via `import vera`**, removing
the websocket + msgpack hop used by the current service-mode `VeraPolicy`.

**Status:** spec for an agent to implement. The current `VeraPolicy`
(`provider.py` → `client.py` → `server_runner.py`) stays the DEFAULT. This adds
a sibling, it does not replace anything.

---

## ⚠️ READ THIS FIRST — the honest tradeoff

In-process is **only** a win if the perf bottleneck is the transport. For VERA it
mostly is NOT. One `infer` is a **WAN/DFoT diffusion forward pass** that takes
**seconds** and returns a whole `action_horizon` chunk. The websocket+msgpack hop
over localhost is **sub-millisecond → noise** against that.

So what you actually buy with in-process:
- **No serialization of `context_rgb`** each infer (a few hundred KB of uint8) — marginal.
- **One less process / no port** — operational simplicity.
- **Zero-copy of torch tensors / shared CUDA context** — the ONLY real perf lever,
  and it only matters if you later fuse a fast Jacobian-only servo loop with no WAN.

### The hard caveat (do not skip)
`vera` pins **`torch==2.6.0`** + needs deepspeed, vggt (git), WAN, CUDA 12.4.
`strands-robots` base env is deliberately light: **numpy 2.x, opencv-headless, NO torch**.
These trees **cannot be merged** without poisoning the light base install.

Therefore the in-process path is **ONLY valid when the process is already running
inside a venv that has `vera` + torch 2.6 installed** (i.e. you launched your agent
from the VERA env, or a unified env you accept the weight of). It MUST fail loudly
and clearly when `import vera` is unavailable, and the factory MUST keep defaulting
to service-mode.

**Decision rule:** service-mode (subprocess) = portable default. native in-proc =
opt-in for single-box GPU deploys where you've accepted the unified env.

---

## What to build

### File: `strands_robots/policies/vera/native.py`

A new `VeraNativePolicy(Policy)` that mirrors `VeraPolicy`'s OUTWARD contract
(same `get_actions` signature, same action-dict output, same rolling context
window + queue draining) but, instead of a `VeraWebsocketClient`, holds the real
adapter **in-process**.

Reuse VERA's OWN adapter so we inherit every deploy-time fix (session auto-reset,
cold-start, collapse canary, flow-comp, adaptive gains) for free — do NOT
reimplement the model glue:

```python
from vera.server.protocol.adapter_factory import make_adapter   # builds VeraPolicyAdapter
# make_adapter(embodiment, algo_config_path=..., dynamics_run_id=..., text=...,
#              sample_steps=..., action_horizon=..., run_dir=...) -> VeraPolicyAdapter
```

The `VeraPolicyAdapter` exposes the SAME logical surface as the wire server:
- `adapter.config`            → the `VeraServerConfig` dataclass (use as the handshake meta)
- `adapter.infer(obs) -> {"action": (H,D), "info": {...}}`
- `adapter.reset(reset_info)`
- (optional) `adapter.configure_runtime(**kwargs)` via `adapter._policy`

So `VeraNativePolicy` is essentially today's `provider.py` with the client
swapped for a direct adapter call. **The frame packing, view-key resolution,
width-concat, queue logic, and `_vector_to_action_dict` are IDENTICAL** — factor
them out so both providers share one implementation (see "Refactor" below).

#### Required methods (Policy ABC)
- `provider_name` → `"vera-native"`
- `requires_images` → `True`
- `set_robot_state_keys(keys)`
- `reset(seed)` → build `reset_info={"session_id": new_uuid, ...}` and call `adapter.reset(...)`
- `async get_actions(observation_dict, instruction, **kwargs)` → same body as
  `VeraPolicy.get_actions`, but `_infer` calls `adapter.infer(req)` directly
  (no websocket). NOTE: `adapter.infer` is SYNC + heavy (seconds, GPU). Run it in
  a thread so the event loop isn't blocked:
  ```python
  import asyncio
  out = await asyncio.to_thread(self._adapter.infer, req)
  ```
- `close()` → drop the adapter ref, `gc.collect()`, `torch.cuda.empty_cache()`
  (guard the torch import — only if torch present).

#### Wire-dict shape passed to `adapter.infer`
Identical to what `client.infer` sends today (the adapter reads the same keys):
```python
req = {
    "context_rgb": context_rgb,   # (T,H,W,3) uint8 — adapter normalizes to float[0,1]
    "view_keys":   list(view_keys),
    "view_widths": view_widths,
    "session_id":  self._session,
    # "prompt": instruction or self.prompt   # only if adapter.config.needs_prompt
}
```
Return: `np.asarray(out["action"], np.float32)`, promote 1-D → `(1,D)`. Same as now.

#### Metadata handshake (no websocket)
Replace `self._client.get_server_metadata()` with:
```python
import dataclasses
meta = dataclasses.asdict(self._adapter.config)   # VeraServerConfig -> dict
```
This yields the SAME dict the websocket server sends on connect
(`view_keys`, `context_frames`, `action_dim`, `gripper_dim_index`,
`gripper_is_raw`, `needs_prompt`, ...), so all downstream logic is unchanged.

---

## Refactor (so the two providers don't drift)

Pull the transport-agnostic helpers OUT of `provider.py` into a shared module,
e.g. `strands_robots/policies/vera/_obs.py`:

- `_is_image_value`, `_to_uint8_frame`
- `_resolve_view_keys(image_keys, observation_dict, meta)`
- `_extract_frame(observation_dict, meta, image_keys)`  → width-concat (T,H,W,3) uint8
- `_action_column_names(action_dim, meta, action_mapping)`
- `_vector_to_action_dict(vec, meta, action_mapping)`    → gripper binarize honored

Then BOTH `VeraPolicy` (service) and `VeraNativePolicy` (native) import these.
The only difference between the two providers becomes: **how `_infer` gets the
action chunk** (websocket `client.infer` vs in-proc `adapter.infer`) and **how
they get `meta`** (handshake recv vs `dataclasses.asdict(adapter.config)`).

Keep the diff minimal — do NOT rewrite the working service path's behavior.

---

## Factory / selection

Add a `mode` switch to how a VERA policy is constructed (config-driven, env-overridable):

- `VeraConfig` (in `config.py`): add field
  `execution: Literal["service", "native"] = "service"`,
  env override `VERA_EXECUTION` (`service` default). Keep default = service so
  nothing changes for existing users.
- A small factory `make_vera_policy(config, **kwargs) -> Policy`:
  - `service` → `VeraPolicy(...)` (unchanged)
  - `native`  → `VeraNativePolicy(...)`; on `ImportError` of `vera`, raise a clear
    error that points at: "install vera into THIS env (torch==2.6.0, CUDA 12.4) or
    use execution='service' (default) which runs vera in its own venv subprocess."
- Wire this into wherever robots resolves a policy by name (registry/policies or
  policies/factory) under the existing `"vera"` provider key, reading
  `config.execution`. Don't add a new public robot name unless the project wants one.

---

## Tests (add under tests/, mirror the existing vera client tests)

1. **Import-guard test**: monkeypatch so `import vera` fails →
   `make_vera_policy(execution="native")` raises a helpful `RuntimeError`/`ImportError`
   whose message mentions torch==2.6.0 + the service fallback. (Runs WITHOUT vera installed.)
2. **Shared-helpers parity test**: feed a synthetic 2-view observation through the
   refactored `_extract_frame` / `_vector_to_action_dict` and assert IDENTICAL output
   to the current `VeraPolicy` private methods (lock the refactor — no behavior drift).
3. **Native infer with a fake adapter** (DI): inject a stub object exposing
   `.config` (a real `VeraServerConfig`) and `.infer(req)->{"action": np.zeros((H,D))}`;
   assert `get_actions` returns H dicts with the right actuator names and gripper
   binarization, and that `adapter.infer` ran via `asyncio.to_thread` (event loop
   not blocked). NO torch / NO real WAN needed — pure contract test.
4. Mark any test that imports real `vera` with `@pytest.mark.slow` / skip-if-not-installed.

---

## Acceptance criteria

- [ ] `VeraNativePolicy` passes the **same** observation→action contract tests as `VeraPolicy`.
- [ ] Default behavior UNCHANGED: `Robot`/factory still builds service-mode VERA unless
      `execution="native"` (or `VERA_EXECUTION=native`) is set.
- [ ] Importing `strands_robots.policies.vera` still works in the LIGHT base env
      (no torch) — `native.py` must NOT import `vera`/torch at module top level; do it
      lazily inside `__init__`/factory so the package import stays light.
- [ ] Clear, actionable error when native is requested but `vera` isn't importable.
- [ ] No duplicated obs/action glue: service + native share `_obs.py`.
- [ ] `adapter.infer` runs off the event loop (`asyncio.to_thread`).

## Non-goals
- Do NOT delete or weaken the service-mode path.
- Do NOT add `vera`/torch to `strands-robots` base or any extra that the light
  install pulls in. If you add an extra, gate it (e.g. `[vera-native]`) and document
  that it drags in torch==2.6.0 + CUDA.
- Do NOT reimplement the WAN/IDM glue — reuse `vera.server.protocol.adapter_factory.make_adapter`.

---

## Why we reuse the adapter (not raw MotionPolicy)
`VeraPolicyAdapter.infer` already does, IN-PROCESS, all the deploy hygiene the
websocket server relied on: session auto-reset (Bug C), cold-start option-b,
collapse canary, optional flow-comp + adaptive gains, and msgpack-safe `info`.
Calling `MotionPolicy.predict_action_chunk` directly would re-expose every one of
those bugs. The adapter is transport-agnostic by design — that's the seam we ride.
