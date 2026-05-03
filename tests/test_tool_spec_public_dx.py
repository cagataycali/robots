import json

# Extract live alias table
import re
from pathlib import Path

from strands_robots.simulation.mujoco.simulation import Simulation

_src = (Path(__file__).resolve().parent.parent / "strands_robots/simulation/mujoco/simulation.py").read_text()
_m = re.search(r"_ALIASES\s*=\s*\{([^}]+)\}", _src)
_LIVE_ALIASES = {}
if _m:
    for _line in _m.group(1).splitlines():
        _mm = re.match(r'\s*"([^"]+)":\s*"([^"]+)"', _line.strip().rstrip(","))
        if _mm:
            _LIVE_ALIASES[_mm.group(1)] = _mm.group(2)


def test_every_tool_spec_action_has_a_public_method_or_documented_alias():
    """DevX contract: every action in tool_spec.json resolves to either
    a PUBLIC method ``sim.<action>()`` or to a PUBLIC method via the
    dispatcher's documented ``_ALIASES`` table. No private leading-underscore
    fallbacks are allowed.
    """
    spec_path = Path(__file__).resolve().parent.parent / "strands_robots/simulation/mujoco/tool_spec.json"
    spec = json.loads(spec_path.read_text())
    actions = spec["properties"]["action"]["enum"]

    offenders = []
    for action in actions:
        resolved = _LIVE_ALIASES.get(action, action)
        method = getattr(Simulation, resolved, None)
        if method is None:
            offenders.append(f"{action!r} → method {resolved!r} does not exist")
        elif resolved.startswith("_"):
            offenders.append(f"{action!r} → PRIVATE method {resolved!r} (leaky DX)")

    assert not offenders, "tool_spec actions must resolve to PUBLIC methods:\n  - " + "\n  - ".join(offenders)
