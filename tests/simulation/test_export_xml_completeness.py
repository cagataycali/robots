"""Regression tests: ``export_xml`` returns the WHOLE document.

``export_xml`` is the documented "serialise model to MJCF string" action
(``docs/simulation/overview.md``) and is agent-callable via ``tool_spec.json``.
It returned the XML only in a ``text`` block, hard-capped at 2000 chars, while
that block's own header announced the full length:

    "Model XML (18868 chars):" followed by 2003 chars of XML

The document was therefore cut mid-attribute, leaving unparseable MJCF - so a
caller could neither inspect the tail of a scene nor feed the result back through
``replace_scene_mjcf``, for any scene larger than a bare world (a single robot
already exceeds the cap by 9x).

The full document now ships in a ``json`` block, which is also the correct home
for machine-readable payloads under the tool-result contract
(``docs/contracts.md``); the ``text`` block keeps a bounded preview so a large
scene cannot flood an LLM's context.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.physics import _XML_PREVIEW_CHARS  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


def _json_block(result: dict) -> dict:
    for block in result["content"]:
        if "json" in block:
            return block["json"]
    raise AssertionError("export_xml returned no json block")


def _text_block(result: dict) -> str:
    for block in result["content"]:
        if "text" in block:
            return block["text"]
    raise AssertionError("export_xml returned no text block")


@pytest.fixture
def big_sim():
    """A scene whose MJCF comfortably exceeds the preview cap (a robot ~18 kB)."""
    s = Simulation(tool_name="export_xml_completeness", mesh=False)
    s.create_world()
    s.add_robot(name="panda")
    yield s
    s.destroy()


@pytest.fixture
def small_sim():
    """A mesh-free scene whose MJCF fits inside the preview cap."""
    s = Simulation(tool_name="export_xml_small", mesh=False)
    s.create_world()
    s.add_object(name="cube", shape="box", size=[0.04] * 3, position=[0.3, 0, 0.02])
    yield s
    s.destroy()


def test_json_block_carries_the_complete_document(big_sim) -> None:
    result = big_sim.export_xml()
    assert result["status"] == "success"
    payload = _json_block(result)

    xml = payload["xml"]
    # Pre-fix the only carrier was the text block, truncated to 2003 chars.
    assert len(xml) > _XML_PREVIEW_CHARS, "fixture scene must exceed the preview cap"
    assert payload["length"] == len(xml)
    assert xml.lstrip().startswith("<mujoco")
    assert xml.rstrip().endswith("</mujoco>"), "document was truncated"


def test_exported_xml_is_parseable(big_sim) -> None:
    """The whole point: the emitted string must be valid MJCF, not a fragment."""
    xml = _json_block(big_sim.export_xml())["xml"]
    # Parses as XML (compiling needs the mesh assets; see the docstring caveat).
    spec = mujoco.MjSpec.from_string(xml)
    assert spec is not None


def test_text_preview_is_bounded_and_says_so(big_sim) -> None:
    text = _text_block(big_sim.export_xml())
    assert len(text) < _XML_PREVIEW_CHARS + 300
    assert "json block" in text, "a truncated preview must point at the full payload"


def test_small_scene_preview_is_not_marked_truncated(small_sim) -> None:
    result = small_sim.export_xml()
    xml = _json_block(result)["xml"]
    assert len(xml) <= _XML_PREVIEW_CHARS
    assert "json block" not in _text_block(result)


def test_mesh_free_scene_round_trips_through_replace_scene_mjcf(small_sim) -> None:
    """A self-contained scene must survive export -> replace."""
    xml = _json_block(small_sim.export_xml())["xml"]
    assert small_sim.replace_scene_mjcf(xml=xml)["status"] == "success"
    model = small_sim.mj_model
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube") >= 0
