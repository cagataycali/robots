"""Regression test for export_xml after replace_scene_mjcf.

Prior to the MjSpec refactor, ``export_xml`` called ``mj.mj_saveLastXML``
which relies on MuJoCo's internal "last loaded XML" cache. That cache is
only populated when the model was loaded from an XML file via
``mj_loadLastXML`` / ``mj.MjModel.from_xml_*``. After the MjSpec-based
``replace_scene_mjcf`` (which compiles from an MjSpec instead), the
cache is empty and ``mj_saveLastXML`` raises a C-level ``FatalError``:
``No XML model loaded``.

The fix is to prefer ``spec.to_xml()`` when the world has a tracked
MjSpec in ``_backend_state['spec']``, falling back to ``mj_saveLastXML``
only when no spec is tracked (e.g. legacy ``load_scene`` paths).

Surfaced by the agent-in-the-loop probe at
``/tmp/e2e_agentic_test_85/notebooks/e2e_agentic_test_85.ipynb``
scenario ``S2_equality``, where the LLM called ``export_xml`` after
``replace_scene_mjcf`` and got an unhandled exception instead of a
clean tool-result dict.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="export_after_replace", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


class TestExportAfterReplace:
    def test_export_xml_after_replace_scene_mjcf(self, sim: Simulation) -> None:
        """export_xml must return a clean success dict with the actual XML,
        not a C-level FatalError, after replace_scene_mjcf."""
        sim.create_world()
        sim.replace_scene_mjcf(
            '<mujoco><worldbody><body name="alpha"><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        )
        result = sim.export_xml()
        assert result["status"] == "success", result
        text = result["content"][0]["text"]
        assert "Model XML" in text
        # The new body name must appear in the exported XML.
        assert 'name="alpha"' in text

    def test_export_xml_after_patch_scene_mjcf(self, sim: Simulation) -> None:
        """Same for patch_scene_mjcf - the live spec is what we should dump."""
        sim.create_world()
        sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "beta", "pos": [0, 0, 0.5]},
                {"op": "add_geom", "body": "beta", "type": "box", "size": [0.05, 0.05, 0.05]},
            ]
        )
        result = sim.export_xml()
        assert result["status"] == "success", result
        text = result["content"][0]["text"]
        assert "Model XML" in text
        assert 'name="beta"' in text

    def test_export_xml_to_file_after_replace(self, sim: Simulation, tmp_path) -> None:
        """The output_path path must also use spec.to_xml() when available."""
        sim.create_world()
        sim.replace_scene_mjcf(
            '<mujoco><worldbody><body name="gamma"><geom type="capsule" size="0.05 0.1"/></body></worldbody></mujoco>'
        )
        out = tmp_path / "out.xml"
        result = sim.export_xml(str(out))
        assert result["status"] == "success", result
        assert out.exists()
        content = out.read_text()
        assert 'name="gamma"' in content

    def test_export_xml_no_world_errors(self) -> None:
        """Unchanged baseline: no world -> clean error, not exception."""
        sim = Simulation(tool_name="export_nw", mesh=False)
        try:
            result = sim.export_xml()
            assert result["status"] == "error"
            assert "no world" in result["content"][0]["text"].lower()
        finally:
            sim.cleanup(policy_stop_timeout=0.5)
