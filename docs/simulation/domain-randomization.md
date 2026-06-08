---
description: What randomize() actually samples — colors, lighting, physics, positions.
---

# Domain randomization

`randomize` perturbs the simulation between episodes so policies generalise. This page
documents the sampling distributions per category.

## TL;DR

```python
sim.randomize(
    randomize_colors=True,
    randomize_lighting=True,
    randomize_physics=True,
    randomize_positions=True,
    position_noise=0.02,
    color_range=(0.1, 1.0),
    friction_range=(0.5, 1.5),
    mass_range=(0.5, 2.0),
    seed=42,
)
```

Each kwarg toggles a category. Categories sample independently and apply in-place.
`randomize` is **destructive** — it writes directly into MuJoCo's model arrays.
To restore baseline values, recompile the scene (e.g. call `load_scene` again or
recreate the sim).

## Categories

### `randomize_colors=True`

Resamples object and floor RGB colours from a uniform distribution over
`color_range=(min, max)` (default `(0.1, 1.0)`). Object alpha is fixed at 1.0. Affects:

- Every `add_object`-created object.
- The ground-plane texture (when present).
- Optionally robot link colours when the model exposes them.

```python
sim.randomize(randomize_colors=True, color_range=(0.2, 0.9))
```

### `randomize_lighting=True`

Perturbs the ambient and directional light parameters:

- Directional light direction (small random rotation).
- Light intensity (uniform around the default).
- Ambient component (uniform around the default).

The aim is to robustify policies against shadows and contrast — not to model arbitrary
camera-room lighting.

```python
sim.randomize(randomize_lighting=True)
```

### `randomize_physics=False`

Perturbs material physical properties using `friction_range` and `mass_range`:

- Per-object mass (multiplicative within `mass_range=(0.5, 2.0)`).
- Per-geom friction (scale within `friction_range=(0.5, 1.5)`).
- Joint damping (scale around 1.0).

Distributions are deliberately configurable — the goal is policy robustness.

```python
sim.randomize(randomize_physics=True,
              friction_range=(0.8, 1.2),
              mass_range=(0.9, 1.1))
```

### `randomize_positions=False`

Adds small random offsets to every object's position using `position_noise` (metres,
default `0.02`):

```python
sim.randomize(randomize_positions=True, position_noise=0.05)
```

## Reproducibility

Pass `seed=` to get a deterministic sequence:

```python
sim.randomize(randomize_colors=True, randomize_physics=True, seed=42)
```

## When to use it

- **Recording a dataset** (chapter 6): randomise between episodes so the dataset
  covers a distribution, not a single look.
- **Eval** (`eval_policy(...)`): call `randomize` in `success_fn` or between episodes
  to evaluate over a distribution. `eval_policy` has no `randomize=` kwarg — call
  `sim.randomize(...)` yourself.
- **Sim-to-real** (chapter 8): heavy randomisation while training; deterministic
  evaluation matching the real-world conditions.

## Undoing randomization

`randomize` writes directly into MuJoCo model arrays. There is no `reset=` kwarg.
To restore baseline values, recompile the scene:

```python
# Option 1: reload from file
sim.load_scene(scene_path="my_scene.xml")

# Option 2: recreate the sim
sim.destroy()
sim = Robot("so100")
```

## When it's not enough

`randomize` covers the common categories. If you need:

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
