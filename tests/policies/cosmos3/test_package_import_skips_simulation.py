"""Importing the cosmos3 package must not load the simulation IK chain.

Until this contract existed the cosmos3 package re-exported ``sim_ik`` at import
time, and ``sim_ik`` imports ``strands_robots.simulation.ik`` (the mink IK
bridge) - so every process that imported the package paid for the IK stack
without asking for a simulator. The two bridge names are now resolved on first
attribute access.

The probe imports ``strands_robots.policies.cosmos3`` itself. It used to rely on
``import strands_robots`` doing that for it, because the root imports
``strands_robots.policies`` and that package exported ``Cosmos3Policy`` eagerly;
since #3587 the root import leaves the cosmos3 package unloaded (its modules
import numpy at module scope, and the bare import is documented as leaving numpy
out of ``sys.modules``), so a probe that stopped at the root would measure a
package that was never imported and the leak assertion would pass vacuously.
The root's own import footprint is graded in ``tests/test_package_lazy_imports.py``.
"""

import json
import subprocess
import sys

import pytest

from strands_robots.policies import cosmos3

# Two snapshots: after importing the package, and after reading one of the two
# lazily resolved bridge names. The second is what makes the first non-vacuous -
# it proves the modules named in ``_IK_CHAIN`` are the ones a bridge access
# loads, so a renamed module cannot turn the leak check into a check of nothing.
_PROBE = """
import json, sys
import strands_robots
import strands_robots.policies.cosmos3 as cosmos3
before = sorted(m for m in sys.modules if m.startswith("strands_robots."))
cosmos3.MinkIKBridge
after = sorted(m for m in sys.modules if m.startswith("strands_robots."))
print(json.dumps({"before": before, "after": after}))
"""

# The chain this branch cut. ``simulation.ik`` is the mink IK bridge - it defers
# the ``mink``/``qpsolvers`` imports themselves, so what the eager re-export cost
# was the simulation package around it; ``sim_ik`` is the cosmos3 side.
_IK_CHAIN = ("strands_robots.policies.cosmos3.sim_ik", "strands_robots.simulation.ik")


def test_importing_the_cosmos3_package_does_not_load_the_ik_chain() -> None:
    out = subprocess.run([sys.executable, "-c", _PROBE], check=True, capture_output=True, text=True, timeout=120).stdout
    snapshots = json.loads(out.strip().splitlines()[-1])
    before, after = snapshots["before"], snapshots["after"]
    # The probe did import the package under measurement.
    assert "strands_robots.policies.cosmos3" in before, before
    leaked = [m for m in before if m in _IK_CHAIN]
    assert leaked == [], f"importing strands_robots.policies.cosmos3 loaded the IK chain: {leaked}"
    # Non-vacuity: reading a bridge name is what loads exactly these modules.
    missing = [m for m in _IK_CHAIN if m not in after]
    assert missing == [], (
        f"reading cosmos3.MinkIKBridge did not load {missing}; _IK_CHAIN names a chain that no longer exists"
    )


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
