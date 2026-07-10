"""Regression tests for ``get_contacts`` geom-name resolution.

``MuJoCoSimulation.get_contacts`` reports each active contact as a pair of
human-readable geom identifiers. Geoms are frequently unnamed - a MuJoCo geom
is not required to carry a ``name`` attribute, and procedurally injected or
imported scene assets often omit them. For those cases the resolver walks a
fallback ladder so the agent-facing output never contains an empty string:

1. the geom's own name, when present;
2. ``"<body>/geom_<id>"`` when the geom is unnamed but its parent body is named;
3. ``"geom_<id>"`` when neither the geom nor its parent body carries a name.

It also degrades gracefully: if MuJoCo name resolution raises unexpectedly
while walking to the parent body, the resolver falls back to ``"geom_<id>"``
rather than propagating the error out of the tool.

These tests pin all three ladder rungs plus the degradation path against a
single scene of overlapping geoms so the contact-naming contract cannot
silently regress.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A named floor plus two free-jointed boxes resting on it. The boxes start
# penetrating the plane so ``mj_forward`` (run inside ``get_contacts``) reports
# contacts immediately, with no stepping required. One box lives in a NAMED
# body; the other in an UNNAMED body. Every box geom itself is unnamed, so the
# three resolution rungs are all exercised:
#   - the named floor plane -> its own name ("floor")
#   - the unnamed geom in the named body -> "named_box/geom_<id>"
#   - the unnamed geom in the unnamed body -> "geom_<id>"
_SCENE_XML = """
<mujoco model="unnamed_geom_contacts">
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.01" pos="0 0 0"/>
    <body name="named_box" pos="0 0 0.05">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
    <body pos="0.5 0 0.05">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _contact_geom_names(result: dict) -> set[str]:
    """Flatten every resolved geom identifier from a get_contacts result."""
    assert result["status"] == "success"
    json_block = next(b["json"] for b in result["content"] if "json" in b)
    names: set[str] = set()
    for contact in json_block["contacts"]:
        names.add(contact["geom1"])
        names.add(contact["geom2"])
    return names


@pytest.fixture
def contact_sim(tmp_path: Path):
    scene = tmp_path / "unnamed_geom_contacts.xml"
    scene.write_text(_SCENE_XML)
    sim = Simulation()
    assert sim.load_scene(str(scene))["status"] == "success"
    try:
        yield sim
    finally:
        sim.cleanup()


def test_get_contacts_resolves_all_name_ladder_rungs(contact_sim: Simulation) -> None:
    """Named geom, named-body fallback, and bare-id fallback all resolve."""
    result = contact_sim.get_contacts()
    names = _contact_geom_names(result)

    # No identifier is ever empty, whatever rung produced it.
    assert names, "expected the penetrating boxes to register contacts"
    assert all(name for name in names)

    # Rung 1: a geom with its own name resolves to that name.
    assert "floor" in names
    # Rung 2: an unnamed geom in a named body -> "<body>/geom_<id>".
    assert any(re.fullmatch(r"named_box/geom_\d+", name) for name in names)
    # Rung 3: an unnamed geom in an unnamed body -> "geom_<id>".
    assert any(re.fullmatch(r"geom_\d+", name) for name in names)


def test_get_contacts_degrades_to_bare_id_when_body_lookup_raises(
    contact_sim: Simulation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If parent-body name resolution raises, fall back to "geom_<id>" and keep succeeding."""
    import mujoco

    real_id2name = mujoco.mj_id2name

    def raising_id2name(model, objtype, obj_id):  # type: ignore[no-untyped-def]
        # Simulate MuJoCo raising while resolving a parent BODY name; geom and
        # other lookups behave normally so only the fallback branch is forced.
        if objtype == mujoco.mjtObj.mjOBJ_BODY:
            raise AttributeError("simulated mujoco body-name lookup failure")
        return real_id2name(model, objtype, obj_id)

    monkeypatch.setattr(mujoco, "mj_id2name", raising_id2name)

    result = contact_sim.get_contacts()
    names = _contact_geom_names(result)

    # The tool still succeeds (no error propagated out of the dispatch).
    assert result["status"] == "success"
    # The named geom is unaffected (resolved before any body walk).
    assert "floor" in names
    # Unnamed geoms whose body walk raised fall back to the bare id form.
    assert any(re.fullmatch(r"geom_\d+", name) for name in names)
    assert all(name for name in names)
