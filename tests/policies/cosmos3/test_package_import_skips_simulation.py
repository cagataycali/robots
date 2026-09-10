"""``import strands_robots`` must not load the simulation package via cosmos3.

``strands_robots/__init__.py`` imports ``strands_robots.policies`` eagerly, which
imports ``strands_robots.policies.cosmos3``. Until this contract existed the
cosmos3 package re-exported ``sim_ik`` at import time, and ``sim_ik`` imports
``strands_robots.simulation.ik`` - so every process that imported the package
root paid for the whole simulation package (25 extra modules, ~20% of import
time) without asking for a simulator. The two bridge names are now resolved on
first attribute access.
"""

import json
import subprocess
import sys

import pytest

from strands_robots.policies import cosmos3

_PROBE = """
import json, sys
import strands_robots
loaded = sorted(m for m in sys.modules if m.startswith("strands_robots.simulation"))
print(json.dumps(loaded))
"""


def test_import_strands_robots_does_not_load_simulation() -> None:
    out = subprocess.run([sys.executable, "-c", _PROBE], check=True, capture_output=True, text=True, timeout=120).stdout
    loaded = json.loads(out.strip().splitlines()[-1])
    assert loaded == [], f"import strands_robots loaded simulation modules: {loaded}"


@pytest.mark.parametrize("name", ["MinkIKBridge", "decode_cosmos_chunk_to_targets"])
def test_sim_ik_names_still_resolve_from_the_package(name: str) -> None:
    from strands_robots.policies.cosmos3 import sim_ik

    assert getattr(cosmos3, name) is getattr(sim_ik, name)
    assert name in cosmos3.__all__


def test_unknown_attribute_raises_the_standard_message() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        getattr(cosmos3, "nope")  # noqa: B009
