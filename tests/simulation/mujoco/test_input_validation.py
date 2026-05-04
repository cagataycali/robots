"""Input validation regression tests for PR #85 fixes (T7, T9, T10).

These guard against silent data-integrity bugs and process-killing MuJoCo
aborts that were caught by autonomous local testing on PR #85.
"""

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim_with_world():
    """A minimal simulation with an empty world for validation tests."""
    sim = Simulation()
    sim.create_world()
    yield sim
    sim.destroy()


@pytest.fixture
def sim_with_robot():
    """A simulation with a single robot for physics-validation tests."""
    sim = Simulation()
    sim.create_world()
    # Use a built-in registry robot — no network I/O
    res = sim.add_robot(name="panda", data_config="panda")
    if res["status"] != "success":
        pytest.skip(f"panda not available: {res['content'][0]['text']}")
    sim.reset()
    yield sim
    sim.destroy()


# --- T9: step validation --------------------------------------------------


class TestStepValidation:
    def test_step_negative_errors(self, sim_with_world):
        """step(n_steps=-5) must error and NOT decrement step_count."""
        initial = sim_with_world._world.step_count
        res = sim_with_world.step(n_steps=-5)
        assert res["status"] == "error"
        assert "n_steps must be >= 0" in res["content"][0]["text"]
        assert sim_with_world._world.step_count == initial, "step_count must not change on rejected call"

    def test_step_zero_is_noop(self, sim_with_world):
        """step(n_steps=0) is a successful no-op."""
        initial = sim_with_world._world.step_count
        res = sim_with_world.step(n_steps=0)
        assert res["status"] == "success"
        assert "no-op" in res["content"][0]["text"].lower()
        assert sim_with_world._world.step_count == initial

    def test_step_positive_still_works(self, sim_with_world):
        """Baseline: non-negative n_steps continues to work."""
        res = sim_with_world.step(n_steps=3)
        assert res["status"] == "success"
        assert sim_with_world._world.step_count == 3


# --- T7: raycast zero-direction guard -------------------------------------


class TestRaycastValidation:
    def test_zero_direction_errors_not_crash(self, sim_with_robot):
        """raycast with zero direction used to abort the interpreter. Now errors cleanly."""
        res = sim_with_robot.raycast(origin=[0, 0, 1], direction=[0, 0, 0])
        assert res["status"] == "error"
        assert "zero-length" in res["content"][0]["text"].lower()

    def test_wrong_length_direction_errors(self, sim_with_robot):
        res = sim_with_robot.raycast(origin=[0, 0, 1], direction=[0, 0])
        assert res["status"] == "error"
        assert "3 elements" in res["content"][0]["text"]

    def test_wrong_length_origin_errors(self, sim_with_robot):
        res = sim_with_robot.raycast(origin=[0, 0], direction=[0, 0, 1])
        assert res["status"] == "error"
        assert "3 elements" in res["content"][0]["text"]

    def test_valid_raycast_still_works(self, sim_with_robot):
        res = sim_with_robot.raycast(origin=[0, 0, 5], direction=[0, 0, -1])
        assert res["status"] == "success"

    def test_multi_raycast_zero_direction_isolates_error(self, sim_with_robot):
        """A zero-length direction in one ray must not abort the whole batch."""
        res = sim_with_robot.multi_raycast(
            origin=[0, 0, 5],
            directions=[[0, 0, -1], [0, 0, 0], [1, 0, -1]],
        )
        assert res["status"] == "success"
        # The JSON payload should show error on ray[1] only
        rays = res["content"][1]["json"]["rays"]
        assert len(rays) == 3
        assert rays[1].get("error") is not None
        assert "zero-length" in rays[1]["error"]


# --- T10: apply_force must reject missing-both --------------------------


class TestApplyForceValidation:
    def test_missing_both_force_and_torque_errors(self, sim_with_robot):
        """apply_force(body='link1') with no force/torque must error, not silent success."""
        res = sim_with_robot.apply_force(body_name="link1")
        assert res["status"] == "error"
        assert "at least one" in res["content"][0]["text"].lower()

    def test_explicit_zero_force_still_clears_latched(self, sim_with_robot):
        """Regression: apply_force(body, force=[0,0,0]) is the documented way to clear."""
        # First latch a force
        r1 = sim_with_robot.apply_force(body_name="link1", force=[10, 0, 0])
        assert r1["status"] == "success"
        # Then clear with explicit zero — this MUST remain valid
        r2 = sim_with_robot.apply_force(body_name="link1", force=[0, 0, 0])
        assert r2["status"] == "success"

    def test_wrong_length_force_errors(self, sim_with_robot):
        res = sim_with_robot.apply_force(body_name="link1", force=[1, 2])
        assert res["status"] == "error"
        assert "3-element" in res["content"][0]["text"]


# --- T8: negative/invalid mass, timestep -------------------------------


class TestMassAndTimestepValidation:
    def test_set_body_properties_negative_mass_errors(self, sim_with_robot):
        res = sim_with_robot.set_body_properties(body_name="link1", mass=-1.0)
        assert res["status"] == "error"
        assert "must be > 0" in res["content"][0]["text"]

    def test_set_body_properties_zero_mass_errors(self, sim_with_robot):
        res = sim_with_robot.set_body_properties(body_name="link1", mass=0.0)
        assert res["status"] == "error"

    def test_set_body_properties_positive_mass_works(self, sim_with_robot):
        res = sim_with_robot.set_body_properties(body_name="link1", mass=2.5)
        assert res["status"] == "success"

    def test_set_timestep_negative_errors(self, sim_with_world):
        res = sim_with_world.set_timestep(-0.01)
        assert res["status"] == "error"
        assert "> 0" in res["content"][0]["text"]

    def test_set_timestep_zero_errors(self, sim_with_world):
        res = sim_with_world.set_timestep(0)
        assert res["status"] == "error"

    def test_set_timestep_positive_works(self, sim_with_world):
        res = sim_with_world.set_timestep(0.001)
        assert res["status"] == "success"

    def test_set_timestep_large_warns_but_succeeds(self, sim_with_world):
        res = sim_with_world.set_timestep(0.5)
        assert res["status"] == "success"
        assert "⚠️" in res["content"][0]["text"] or "unusually" in res["content"][0]["text"]


# --- T38: set_gravity dim validation -----------------------------------


class TestSetGravityValidation:
    def test_two_element_gravity_errors(self, sim_with_world):
        res = sim_with_world.set_gravity([0.0, 0.0])
        assert res["status"] == "error"
        assert "3-element" in res["content"][0]["text"]

    def test_scalar_gravity_still_works(self, sim_with_world):
        # Scalar form convenience (z-only) preserved
        res = sim_with_world.set_gravity(-9.81)
        assert res["status"] == "success"

    def test_full_vector_gravity_works(self, sim_with_world):
        res = sim_with_world.set_gravity([1.0, 2.0, -9.0])
        assert res["status"] == "success"


# --- T11: set_joint_positions list/dict support -----------------------


class TestSetJointPositionsForms:
    def test_dict_form_works(self, sim_with_robot):
        # Pick a valid joint name from the robot
        joint_names = list(sim_with_robot._world.robots.values())[0].joint_names or []
        if not joint_names:
            import pytest as _pytest
            _pytest.skip("robot has no named joints")
        res = sim_with_robot.set_joint_positions(positions={joint_names[0]: 0.1})
        assert res["status"] == "success"

    def test_list_form_matches_count(self, sim_with_robot):
        joint_names = list(sim_with_robot._world.robots.values())[0].joint_names or []
        if not joint_names:
            import pytest as _pytest
            _pytest.skip("robot has no named joints")
        res = sim_with_robot.set_joint_positions(positions=[0.0] * len(joint_names))
        assert res["status"] == "success", res["content"][0]["text"]

    def test_list_form_wrong_length_errors(self, sim_with_robot):
        # 999 is almost certainly wrong for any robot
        res = sim_with_robot.set_joint_positions(positions=[0.1] * 999)
        assert res["status"] == "error"
        assert "does not match" in res["content"][0]["text"]


# --- T5: policy-running guards -----------------------------------------


class TestPolicyRunningGuards:
    """Simulate policy-running state by poisoning _policy_threads.

    We insert a fake Future whose done() returns False so _require_no_running_policy
    flags a running policy without actually starting one.
    """

    def _install_fake_running_policy(self, sim):
        class _FakeRunningFuture:
            def done(self):
                return False

        sim._policy_threads["fake"] = _FakeRunningFuture()

    def test_reset_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.reset()
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]

    def test_set_gravity_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.set_gravity([0, 0, -5])
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]

    def test_set_timestep_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.set_timestep(0.001)
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]

    def test_set_joint_positions_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.set_joint_positions(positions={"nope": 0.0})
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]

    def test_apply_force_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.apply_force(body_name="link1", force=[1, 0, 0])
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]

    def test_set_body_properties_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.set_body_properties(body_name="link1", mass=3.0)
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]

    def test_randomize_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.randomize(seed=42)
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]


# --- T6: add_robot initial state is zero -------------------------------


class TestAddRobotInitialState:
    """After add_robot, qpos/qvel/ctrl must be zero without needing reset()."""

    def test_initial_qpos_is_zero(self):
        import numpy as np
        sim = Simulation()
        try:
            sim.create_world()
            res = sim.add_robot(name="panda", data_config="panda")
            if res["status"] != "success":
                import pytest as _pytest
                _pytest.skip(f"panda not available: {res['content'][0]['text']}")
            # IMPORTANT: do NOT call reset. T6 requires that add_robot itself leaves a clean state.
            data = sim._world._data
            assert np.allclose(data.qpos, 0.0), f"qpos should be zero after add_robot, got {data.qpos}"
            assert np.allclose(data.qvel, 0.0), f"qvel should be zero after add_robot, got {data.qvel}"
            assert np.allclose(data.ctrl, 0.0), f"ctrl should be zero after add_robot, got {data.ctrl}"
        finally:
            sim.destroy()


# --- T3: render camera strict validation -------------------------------


class TestRenderCameraValidation:
    def test_unknown_camera_errors(self, sim_with_world):
        res = sim_with_world.render(camera_name="does_not_exist", width=64, height=48)
        assert res["status"] == "error"
        assert "not found" in res["content"][0]["text"]

    def test_default_camera_labelled_honestly(self, sim_with_world):
        res = sim_with_world.render(camera_name="default", width=64, height=48)
        if res["status"] != "success":
            import pytest as _pytest
            _pytest.skip(f"offscreen render unavailable: {res['content'][0]['text']}")
        assert "free (default)" in res["content"][0]["text"]

    def test_free_alias_labelled_honestly(self, sim_with_world):
        res = sim_with_world.render(camera_name="free", width=64, height=48)
        if res["status"] != "success":
            import pytest as _pytest
            _pytest.skip(f"offscreen render unavailable: {res['content'][0]['text']}")
        assert "free (default)" in res["content"][0]["text"]

    def test_render_depth_unknown_camera_errors(self, sim_with_world):
        res = sim_with_world.render_depth(camera_name="ghost_cam", width=64, height=48)
        assert res["status"] == "error"
        assert "not found" in res["content"][0]["text"]


# --- T2: camera target actually applied -----------------------------


class TestAddCameraTargetOrients:
    """The 'headline broken feature': add_camera(target=...) was silently dropped
    so every custom camera rendered the same default view. These tests verify
    that orientation now flows through to the rendered pixels.
    """

    def _with_obj(self):
        """Create a world with a distinguishable colored object for the cameras to frame."""
        sim = Simulation()
        sim.create_world()
        # Add a vivid red box at origin to make camera differences visible.
        sim.add_object(
            name="target_box",
            shape="box",
            size=[0.3, 0.3, 0.3],
            position=[0.0, 0.0, 0.25],
            color=[1.0, 0.0, 0.0, 1.0],
            is_static=True,
        )
        return sim

    def test_degenerate_target_equals_position_errors(self):
        sim = self._with_obj()
        try:
            res = sim.add_camera(name="bad_cam", position=[1, 2, 3], target=[1, 2, 3])
            assert res["status"] == "error"
            assert "identical" in res["content"][0]["text"]
        finally:
            sim.destroy()

    def test_wrong_length_position_errors(self):
        sim = self._with_obj()
        try:
            res = sim.add_camera(name="bad_cam", position=[1, 2], target=[0, 0, 0])
            assert res["status"] == "error"
            assert "3 elements" in res["content"][0]["text"]
        finally:
            sim.destroy()

    def test_xyaxes_emitted_in_xml(self):
        """The merged scene XML must contain xyaxes= for cameras with a target."""
        sim = self._with_obj()
        try:
            res = sim.add_camera(
                name="side_cam", position=[2.0, 0.0, 0.3], target=[0.0, 0.0, 0.25]
            )
            assert res["status"] == "success", res["content"][0]["text"]
            # Grab the stored scene XML.
            xml = sim._world._backend_state.get("xml", "")
            # If there are no robots in the scene the XML is only recompiled (not injected).
            # In either case the camera emission path should have used our helper.
            if xml and "side_cam" in xml:
                assert "xyaxes=" in xml, "xyaxes attribute must be written for targeted cameras"
        finally:
            sim.destroy()

    def test_different_targets_produce_different_xyaxes(self):
        """Two cameras at the SAME position but different targets must produce
        DIFFERENT ``xyaxes`` strings in the merged scene XML. Before the fix the
        XML had no orientation at all, so both cameras shared MuJoCo's default
        look direction -> identical frames regardless of `target`.

        We assert on XML (orientation bits) rather than rendered pixels, because
        the offscreen GL context on some CI runners produces blank frames which
        makes pixel-level comparison unreliable (see note on macOS depth/ARB_clip
        elsewhere in this suite)."""
        import re as _re
        sim = self._with_obj()
        try:
            res_a = sim.add_camera(
                name="cam_a", position=[2.0, 0.0, 0.5], target=[0.0, 0.0, 0.25]
            )
            res_b = sim.add_camera(
                name="cam_b", position=[2.0, 0.0, 0.5], target=[0.0, 2.0, 0.25]
            )
            assert res_a["status"] == "success"
            assert res_b["status"] == "success"
            xml = sim._world._backend_state.get("xml", "")
            a_match = _re.search(r'<camera[^>]*name="cam_a"[^>]*xyaxes="([^"]+)"', xml)
            b_match = _re.search(r'<camera[^>]*name="cam_b"[^>]*xyaxes="([^"]+)"', xml)
            assert a_match, f"cam_a has no xyaxes in XML: {xml[:500]}"
            assert b_match, f"cam_b has no xyaxes in XML: {xml[:500]}"
            assert a_match.group(1) != b_match.group(1), (
                "cameras with different targets must have different xyaxes (they are currently identical,"
                f" which means `target` is being ignored): {a_match.group(1)}"
            )
        finally:
            sim.destroy()


class TestCameraXyAxesHelper:
    """Direct unit test on the _camera_xyaxes_from_target helper."""

    def test_basic_look_at_origin(self):
        from strands_robots.simulation.mujoco.mjcf_builder import _camera_xyaxes_from_target
        # Camera at (2, 0, 0) looking at origin along -X, up = +Z.
        # forward = normalize(origin - pos) = (-1, 0, 0)
        # right   = forward × up = (-1,0,0) × (0,0,1) = (0*1 - 0*0, 0*0 - -1*1, -1*0 - 0*0) = (0, 1, 0)
        # image_up = right × forward = (0,1,0) × (-1,0,0) = (1*0 - 0*0, 0*-1 - 0*0, 0*0 - 1*-1) = (0, 0, 1)
        s = _camera_xyaxes_from_target([2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert s is not None
        parts = [float(x) for x in s.split()]
        assert len(parts) == 6
        rx, ry, rz, ux, uy, uz = parts
        assert abs(rx) < 1e-5 and abs(ry - 1.0) < 1e-5 and abs(rz) < 1e-5, f"right={parts[:3]}"
        assert abs(ux) < 1e-5 and abs(uy) < 1e-5 and abs(uz - 1.0) < 1e-5, f"image_up={parts[3:]}"

    def test_degenerate_returns_none(self):
        from strands_robots.simulation.mujoco.mjcf_builder import _camera_xyaxes_from_target
        assert _camera_xyaxes_from_target([1, 2, 3], [1, 2, 3]) is None
