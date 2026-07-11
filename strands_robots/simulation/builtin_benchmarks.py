"""Built-in benchmark specs shipped with ``strands_robots``.

The predicate/reward DSL (:mod:`strands_robots.simulation.predicates`) grew a
complete floating-base locomotion vocabulary - the ``base_velocity_tracking``
exponential-kernel tracking reward, the ``base_height`` / ``base_orientation``
posture regularizers, and the ``base_beyond_x`` (forward-progress success),
``base_tipped`` (topple) and ``base_below_z`` (height-collapse) predicates - all
reading the embodiment-agnostic floating-base surface from ``get_observation``.
Those primitives, however, wired into no runnable benchmark:
:func:`~strands_robots.simulation.benchmark.list_benchmarks` was empty until a
caller hand-authored a spec (``register_benchmark_from_file``) or loaded the
LIBERO suite. This module ships a canonical velocity-tracking locomotion
benchmark composed from those primitives so a floating-base robot has a
runnable, discoverable eval out of the box.

Registration is opt-in - a caller invokes :func:`register_builtin_benchmarks`
(mirroring how the LIBERO suite registers on demand via
:func:`~strands_robots.simulation.benchmark.register_benchmark`) rather than a
global import side effect - so importing ``strands_robots`` performs no registry
mutation and there is no import-order coupling.
"""

from __future__ import annotations

import copy
from typing import Any

from strands_robots.simulation.benchmark import register_benchmark
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark

# ``go2_walk_forward``: the canonical legged-locomotion velocity-tracking task -
# walk the Unitree Go2 quadruped forward at 1 m/s and stay standing.
#
#   - success: the base walks past x = 2 m (``base_beyond_x``) - a real
#     forward-progress terminal, not "did not fall" (a standing-still policy
#     never satisfies it).
#   - failure: the base topples more than ~53 deg off level (``base_tipped``,
#     tol=0.7) OR its height collapses below 0.18 m (``base_below_z``, the Go2
#     stands at ~0.32 m). Either fall mode terminates the episode early.
#   - dense_reward: the legged_gym-style shaping stack - an exponential-kernel
#     reward for tracking the commanded body-frame twist (vx=1.0, no lateral /
#     yaw command), a squared base-height regularizer toward the 0.32 m nominal
#     stance, and a flat-orientation regularizer. Composable, bounded, and
#     dense so an RL/BC policy gets a gradient every step.
#
# All predicates read ``base_pos`` / ``base_quat`` / ``base_lin_vel`` /
# ``base_ang_vel`` from ``get_observation`` - no per-embodiment base body name -
# so the same spec form transfers to any floating-base robot by swapping
# ``default_robot`` / ``supported_robots`` and the height thresholds.
_GO2_WALK_FORWARD: dict[str, Any] = {
    "name": "go2_walk_forward",
    "default_robot": "unitree_go2",
    "supported_robots": ["unitree_go2"],
    "max_steps": 1000,
    "success": {"all": [{"predicate": "base_beyond_x", "x": 2.0}]},
    "failure": {
        "any": [
            {"predicate": "base_tipped", "tol": 0.7},
            {"predicate": "base_below_z", "z": 0.18},
        ]
    },
    "dense_reward": [
        {
            "predicate": "base_velocity_tracking",
            "vx": 1.0,
            "vy": 0.0,
            "wz": 0.0,
            "lin_weight": 1.0,
            "ang_weight": 0.5,
            "tracking_sigma": 0.25,
        },
        {"predicate": "base_height", "target": 0.32, "weight": 0.5},
        {"predicate": "base_orientation", "weight": 0.5},
    ],
}

# Registry of shipped built-in specs, keyed by their canonical registry name.
# Extend this dict to ship more (e.g. a humanoid ``g1_walk_forward``); the same
# floating-base DSL applies with different height/orientation thresholds.
_BUILTIN_SPECS: dict[str, dict[str, Any]] = {
    "go2_walk_forward": _GO2_WALK_FORWARD,
}


def builtin_benchmark_specs() -> dict[str, dict[str, Any]]:
    """Return deep copies of the shipped benchmark spec dicts, keyed by name.

    Deep-copied so a caller inspecting or forking a spec cannot mutate the
    module-level canonical definitions.
    """
    return {name: copy.deepcopy(spec) for name, spec in _BUILTIN_SPECS.items()}


def register_builtin_benchmarks() -> list[str]:
    """Compile and register every shipped built-in benchmark into the registry.

    After this call the specs are discoverable via
    :func:`~strands_robots.simulation.benchmark.list_benchmarks` and runnable via
    :meth:`~strands_robots.simulation.base.SimEngine.evaluate_benchmark`.
    Idempotent by overwrite: re-registering a name replaces the prior instance
    (matching :func:`~strands_robots.simulation.benchmark.register_benchmark`).

    Returns:
        The sorted list of registered benchmark names, so a caller can
        immediately look them up / evaluate them.
    """
    names: list[str] = []
    for name, spec in _BUILTIN_SPECS.items():
        # from_dict overwrites spec["name"] with the registry key contract, so a
        # deep copy keeps the canonical dict pristine for builtin_benchmark_specs.
        bench = DeclarativeBenchmark.from_dict(copy.deepcopy(spec))
        register_benchmark(name, bench)
        names.append(name)
    return sorted(names)


__all__ = ["builtin_benchmark_specs", "register_builtin_benchmarks"]
