# Reconciling the paper's "generalist / all-embodiment" claim with our integration

You were right to push back. My earlier "only 4 embodiments, ~38 robots out of
scope" framing conflated two different things and **undersold** VERA's design.
Here is the accurate picture, straight from the paper + code.

---

## What the paper actually claims (title: *Turning Video Models into Generalist Robot Policies*)

> "decoupled video planning + faithful video-to-action translation is a viable
>  route to **zero-shot, cross-embodiment** robot control. **One video planner,
>  many IDMs.**"

Two stages, and the asymmetry is the whole point:

| Stage | Embodiment coupling | Cost to add a robot |
|---|---|---|
| **Video planner** (WAN/DFoT) — dreams future frames | **Embodiment-AGNOSTIC**, trained once | **Zero.** The OMNI planner is trained on a 4-env mixture and reused. It never sees actions. |
| **Jacobian IDM** — dream → actions | **Embodiment-SPECIFIC**, but *data-efficient* | A **small** model: frozen VGGT/DINO backbone + a flow→action head, trained per robot **from self-play** (TRAINING.md). High-DoF-scalable because it regresses a *local action↔flow Jacobian* and inverts it. |

So "generalist / all-embodiment" is a claim about the **architecture's
extensibility**, not about a single shipped checkpoint driving every robot:
- **One** planner serves all embodiments (genuinely embodiment-agnostic).
- **Each** new robot needs **one** cheap IDM (frozen backbone, head trained from
  self-play data) — *not* a full from-scratch VLA. That is the "data-efficient"
  generalist claim, and it is real.

**Where I was wrong:** I implied each new robot needs a heavy bespoke model like
a full GR00T/Pi0 retrain. It doesn't — the IDM is the small, swappable piece, and
the expensive planner is shared. The paper's bar to "support a new embodiment" is
*much* lower than I stated.

**Where I was right:** the IDM is still a **learned** model (verified:
`vera/idm/inverse_dynamics/models/idm_transformer_model.py` — VGGT/DPT backbone +
`nn.Linear` action head; `freeze_backbone=True`). It is *not* a zero-training
analytic Jacobian from URDF. So adding a brand-new embodiment is **cheap, but not
free** — you still train (a small) IDM on that robot's self-play `du`↔flow data.
And today only **2 IDMs ship** (PushT, MimicGen); Allegro/DROID/IIWA configs +
checkpoints are Wave-2.

---

## So does our `strands-robots` integration "work for all embodiments"?

Re-answered in three honest layers:

### Layer 1 — Transport / protocol: **embodiment-agnostic ✅ (already done)**
The websocket protocol, the docker runtime, and the provider's frame-packing +
metadata handshake make **zero** assumptions about the robot. Any embodiment the
server advertises (`view_keys`, `action_space`, `action_dim`, `gripper_*`) is
consumed generically. Add a new VERA embodiment server-side → our client speaks to
it with no code change. **This is the part that is genuinely "all-embodiment".**

### Layer 2 — Action binding: **now generic across action_spaces ✅ (just fixed)**
After today's fix, `provider.py` routes by the server's `action_space`:
- `joint_position` → columns bind to the robot's real joints (allegro, and any
  future joint-space embodiment).
- `eef_delta` / `cartesian_delta` → IK to joint targets (mimicgen, droid, and any
  future Cartesian embodiment) — **kinematics-general**, works for ANY arm whose
  MuJoCo model + ee-frame you pass to `set_ik_target`.
- `pos` / unknown → mapping/passthrough.
This means the *binding layer* now supports the **action-space families** VERA
uses, not just two hard-coded embodiments. A new VERA embodiment that uses one of
these action spaces drives our robots with no further provider changes.

### Layer 3 — Model availability: **gated by which IDM exists (not our code)**
A specific robot only *moves well* if a VERA IDM was trained for (or transfers to)
its kinematics + camera setup:
- **Today:** PushT (smoke) + MimicGen→Panda ship; mimicgen `eef_delta` + our IK
  transfers to kinematically-similar arms (FR3/Panda/UR/Kinova/Sawyer/IIWA) with
  per-arm validation (Cartesian deltas transfer better than joint-space).
- **Wave-2:** Allegro (hand, joint_position) + DROID (FR3, cartesian_delta) IDMs.
- **Any other of our 68 robots:** supported *as soon as someone trains its IDM*
  (cheap, per TRAINING.md) — our transport + binding are already ready for it.

---

## Corrected coverage statement

| Question | Answer |
|---|---|
| Is the *architecture* all-embodiment? | **Yes** — one shared planner, cheap per-robot IDMs. Paper's claim stands. |
| Is our *integration* embodiment-general? | **Yes, at the protocol + action-space layers** (transport agnostic; joint/eef/cartesian all bound + IK'd). |
| Does a *given robot* work today? | Only if a VERA IDM exists for it (today: PushT, MimicGen-Panda; soon: Allegro, DROID). Others = train a small IDM. |
| Is adding a robot a full VLA retrain? | **No** — frozen backbone + small head, self-play data. Cheap, but **not zero** (it's learned, not analytic). |

**Net:** the paper's "generalist" claim is about the *recipe*, and our integration
now matches that recipe's generality — the transport and the action-binding/IK are
embodiment-general. The only thing that isn't "free" is that each physical robot
still needs its (small, data-efficient) IDM checkpoint to exist. Our code no longer
blocks any embodiment; model availability does.

---

## What would make our side truly "any embodiment, plug-and-play"

1. **(done)** action_space-generic binding + Cartesian IK (this commit).
2. **Auto-configure `set_ik_target`** from the registry: when a robot is
   resolved, look up its ee-frame/body name + MjModel and call `set_ik_target`
   automatically for eef/cartesian embodiments (so users don't wire IK by hand).
3. **An IDM-training quickstart** in our docs that points at VERA's
   `config_jacobian_*` + self-play recorder, so adding a robot is a documented
   one-command path (train small IDM → drop ckpt → serve).
4. **A transfer-eval harness**: run a mimicgen/droid IDM on a *different* arm via
   our IK and report success + tracking error, to know empirically which of our
   arms a given IDM transfers to without retraining.
