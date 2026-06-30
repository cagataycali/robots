"""Regression: the locomotion-intent ``planning`` package is gone.

Locomotion intent is expressed through the well-known ``policy_kwargs`` goal
channel (``target_velocity`` / ``target_height`` / ``locomotion_style``) that
``run_policy`` forwards to the policy - see
``test_policy_kwargs_forwarding.py``. The separate ``strands_robots.planning``
package (a ``KinematicPlanner`` / ``InputSource`` abstraction that produced the
same dict) was redundant and removed, along with the ``planner=`` parameter on
the simulation run loop.

These pin the removal: the package must not be importable and the run-loop
entry points must not carry a ``planner`` parameter. Both fail on the pre-removal
code (where the package and parameter existed) and pass once deleted.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner


def test_planning_package_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("strands_robots.planning")


def test_run_policy_has_no_planner_parameter() -> None:
    assert "planner" not in inspect.signature(SimEngine.run_policy).parameters


def test_policy_runner_run_has_no_planner_parameter() -> None:
    assert "planner" not in inspect.signature(PolicyRunner.run).parameters
