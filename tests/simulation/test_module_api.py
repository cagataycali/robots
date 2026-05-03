"""Module-level API tests: __getattr__ lazy-load + error paths.

Covers the lazy-loading tails in:
* ``strands_robots/simulation/__init__.py``
* ``strands_robots/simulation/mujoco/__init__.py``
"""

from __future__ import annotations

import pytest


def test_simulation_getattr_raises_on_unknown():
    import strands_robots.simulation as mod

    with pytest.raises(AttributeError, match="has no attribute 'DoesNotExist'"):
        _ = mod.DoesNotExist


def test_mujoco_module_alias_is_simulation_class():
    from strands_robots.simulation.mujoco import MuJoCoSimulation
    from strands_robots.simulation.mujoco.simulation import Simulation

    assert MuJoCoSimulation is Simulation


def test_mujoco_getattr_raises_on_unknown():
    import strands_robots.simulation.mujoco as mod

    with pytest.raises(AttributeError, match="has no attribute 'NotARealClass'"):
        _ = mod.NotARealClass
