"""``import strands_robots`` must not load the simulation IK chain via cosmos3.

``strands_robots/__init__.py`` imports ``strands_robots.policies`` eagerly, which
imports ``strands_robots.policies.cosmos3``. Until this contract existed the
cosmos3 package re-exported ``sim_ik`` at import time, and ``sim_ik`` imports
``strands_robots.simulation.ik`` (the mink IK bridge) - so every process that
imported the package root paid for the IK stack without asking for a simulator.
The two bridge names are now resolved on first attribute access.

What this does NOT claim: with mujoco installed, ``strands_robots/__init__.py``
still imports ``strands_robots.simulation.mujoco.backend`` on purpose to pin
``MUJOCO_GL`` before any ``import mujoco``, and importing that submodule runs
``strands_robots/simulation/__init__.py`` (14 modules). That eager hook is a
separate finding; this test measures only the cosmos3 -> IK chain.
"""

import json
import subprocess
import sys

import pytest

from strands_robots.policies import cosmos3

_PROBE = """
import json, sys
import strands_robots
loaded = sorted(m for m in sys.modules if m.startswith("strands_robots."))
print(json.dumps(loaded))
"""

# The chain this branch cut. ``simulation.ik`` is the mink IK bridge - it defers
# the ``mink``/``qpsolvers`` imports themselves, so what the eager re-export cost
# was the simulation package around it; ``sim_ik`` is the cosmos3 side.
_IK_CHAIN = ("strands_robots.policies.cosmos3.sim_ik", "strands_robots.simulation.ik")


def test_import_strands_robots_does_not_load_the_cosmos3_ik_chain() -> None:
    out = subprocess.run([sys.executable, "-c", _PROBE], check=True, capture_output=True, text=True, timeout=120).stdout
    loaded = json.loads(out.strip().splitlines()[-1])
    # Non-vacuity: the probe did import the package and its policies.
    assert "strands_robots.policies.cosmos3" in loaded, loaded
    leaked = [m for m in loaded if m in _IK_CHAIN]
    assert leaked == [], f"import strands_robots loaded the cosmos3 IK chain: {leaked}"


@pytest.mark.parametrize("name", ["MinkIKBridge", "decode_cosmos_chunk_to_targets"])
def test_sim_ik_names_still_resolve_from_the_package(name: str) -> None:
    from strands_robots.policies.cosmos3 import sim_ik

    assert getattr(cosmos3, name) is getattr(sim_ik, name)
    assert name in cosmos3.__all__


def test_unknown_attribute_raises_the_standard_message() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        getattr(cosmos3, "nope")  # noqa: B009


def test_first_access_caches_the_resolved_name_in_the_module_dict() -> None:
    """Only the first access pays the lookup, as in ``simulation.newton``.

    Popping the cached entry puts the package back in its post-import state, so
    the next access goes through ``__getattr__`` again and must leave the object
    it resolved in ``vars()``.
    """
    vars(cosmos3).pop("MinkIKBridge", None)

    resolved = cosmos3.MinkIKBridge

    assert vars(cosmos3)["MinkIKBridge"] is resolved
