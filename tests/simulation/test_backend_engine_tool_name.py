# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Every sim backend engine accepts the ``tool_name`` the factory injects.

``Robot(name, mode="sim", backend=...)`` builds its engine through one call for
every backend::

    sim = create_simulation(backend, tool_name=f"{name}_sim", **kwargs)

The MuJoCo engine declares ``tool_name``; the Newton engine did not, and once its
constructor started REJECTING residual keywords instead of dropping them (the
right call - a discarding ``**kwargs`` sink turned ``num_envs=4096`` into a
successful no-op) that omission became a hard failure of the documented entry
point::

    Robot("so100", mode="sim", backend="newton")
    TypeError: NewtonSimEngine got unexpected keyword argument(s): 'tool_name'.
               Accepted: solver, default_timestep, substeps, device, ...

The rejection runs before ``ensure_newton()``, so the ``TypeError`` replaced even
the "install strands-robots[sim-newton]" hint the call is supposed to give where
the backend is absent.

Asserted on the constructor signature rather than by instantiating, so the
contract holds wherever the tests run - constructing the Newton engine needs warp.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

#: (module, class) for each engine ``create_simulation`` can build. Isaac is
#: omitted: its engine module imports Omniverse at import time, so it cannot be
#: inspected off an Isaac Sim install.
_ENGINES = [
    ("strands_robots.simulation.mujoco.simulation", "MuJoCoSimEngine"),
    ("strands_robots.simulation.newton.simulation", "NewtonSimEngine"),
]

#: What ``strands_robots.robot.Robot`` passes positionally-by-keyword to every
#: backend it builds.
_FACTORY_KWARGS = {"tool_name": "so100_sim"}


@pytest.mark.parametrize(("module_path", "class_name"), _ENGINES)
def test_the_engine_accepts_the_factory_injected_kwargs(module_path, class_name) -> None:
    """Binding the factory's own call must not raise ``TypeError``."""
    engine = getattr(importlib.import_module(module_path), class_name)
    # bind() raises TypeError for a keyword the signature cannot absorb, which is
    # the same failure the caller would hit at construction.
    inspect.signature(engine.__init__).bind(engine, **_FACTORY_KWARGS)


@pytest.mark.parametrize(("module_path", "class_name"), _ENGINES)
def test_the_engine_declares_tool_name_explicitly(module_path, class_name) -> None:
    """Declared, not absorbed by ``**kwargs``.

    A ``**kwargs`` sink would satisfy ``bind()`` while leaving the keyword to be
    dropped or rejected downstream; the identifier is part of the contract, so it
    has to be a named parameter with a default.
    """
    engine = getattr(importlib.import_module(module_path), class_name)
    parameter = inspect.signature(engine.__init__).parameters.get("tool_name")
    assert parameter is not None, f"{class_name} does not declare tool_name"
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert isinstance(parameter.default, str) and parameter.default, parameter.default


def test_the_newton_rejection_message_lists_tool_name_as_accepted() -> None:
    """The error text is the only discovery surface for the accepted keywords."""
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    with pytest.raises(TypeError) as excinfo:
        engine._reject_unsupported_setup_kwargs({"sbsteps": 3})

    message = str(excinfo.value)
    assert "'sbsteps'" in message, message
    assert "tool_name" in message, message
    assert message.isascii()
