---
description: What randomize() actually samples — colors, lighting, physics, cameras, asset textures.
---

# Domain randomization

`randomize` perturbs the simulation between episodes so policies generalise. This page
documents the sampling distributions per category.

## TL;DR

```python
sim.randomize(
    colors=True,
    lighting=True,
    physics=True,
    cameras=True,
)
```

Each kwarg toggles a category. Each category samples independently from its
distribution and applies in-place. Subsequent rollouts see the new world; reset to
fixed values via `randomize(reset=True)` (or recreate the sim).

## Categories

### `colors=True`

Resamples object and floor RGB colours from a uniform distribution over the full RGB
cube. Object alpha is fixed at 1.0. Affects:

- Every `add_object`-created object.
- The ground-plane texture (when present).
- Optionally robot link colours when the model exposes them.

```python
sim.randomize(colors=True)
```

### `lighting=True`

Perturbs the ambient and directional light parameters:

- Directional light direction (small random rotation).
- Light intensity (uniform around the default).
- Ambient component (uniform around the default).

The aim is to robustify policies against shadows and contrast — not to model arbitrary
camera-room lighting.

### `physics=True`

Perturbs material physical properties:

- Per-object mass (multiplicative, modest range around the registered mass).
- Per-geom friction (scale around 1.0).
- Joint damping (scale around 1.0).

Distributions are deliberately tight — the goal is policy robustness, not chaos.

### `cameras=True`

Adds small random perturbations to every camera's position and orientation:

- Position: uniform offset in a small cube.
- Orientation: small random rotation.

Field-of-view (`fovy`) and resolution are not randomised — those are part of the
camera's *type*.

### `textures=True`

(Where supported by the loaded model.) Re-samples geom textures from a built-in
texture atlas. Useful for visual policies trained on diverse backgrounds.

## When to use it

- **Recording a dataset** (chapter 6): randomise between episodes so the dataset
  covers a distribution, not a single look.
- **Eval** (`eval_policy(randomize=True, ...)`): each of the N episodes gets a fresh
  world. Reported success rate is over the distribution.
- **Sim-to-real** (chapter 8): heavy randomisation while training; deterministic
  evaluation matching the real-world conditions.

## Reset

```python
sim.randomize(reset=True)    # restore baseline values
```

If you only randomised some categories, only those get reset.

## When it's not enough

`randomize` covers the easy categories. If you need:

- **Procedural object placement** — call `add_object` in a loop instead.
- **Different scenes** — `load_scene` with multiple MJCF files.
- **Adversarial physics** — modify the MJCF directly before `create_world`.

These are not in scope for `randomize` because they break the "tweak the existing
world" model and require recompiling MuJoCo's mass matrix.

## See also

- [Simulation overview](overview.md) — `randomize` parameters.
- [World building](world-building.md) — for cases where randomisation isn't enough.
- [Tutorial 6 — Recording](../tutorial/06-recording.md) — typical randomisation
  pattern in a record loop.
- [Tutorial 8 — Real hardware](../tutorial/08-real-hardware.md) — sim-to-real
  considerations.
