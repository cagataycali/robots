"""Regression test for export_xml after replace_scene_mjcf / patch_scene_mjcf.

In the MjSpec backend every code path (create_world, load_scene,
replace_scene_mjcf, patch_scene_mjcf, inject_*) stashes the live
``MjSpec`` in ``world._backend_state["spec"]``. ``export_xml`` therefore
dumps via ``spec.to_xml()`` which always reflects the live scene - no
need to rely on ``mj.mj_saveLastXML`` (which raises a C-level
``FatalError`` when the model wasn't loaded from an XML file).

The agent-in-the-loop probe at
``/tmp/e2e_agentic_test_85/notebooks/e2e_agentic_test_85.ipynb``
(scenario ``S2_equality``) originally surfaced this by calling
``export_xml`` right after ``replace_scene_mjcf`` and getting an
unhandled exception. These tests cover the critical combinations
(replace+export, patch+export, export-to-file, no-world).
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
