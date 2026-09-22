"""The M3 Pro twin transport: the robot's graph, answered by a sim engine.

Everything here runs with no MuJoCo: one double stands in for the sim engine
and is faithful in the one respect that matters - it **records** every
``send_action`` (the actuator targets and the substeps) and answers
``get_observation`` from a mutable state, so a test can say what the twin
wrote to the model and for how long, not only that the reply said "published".

The driver is the one under ``tests/drivers/test_yahboom_m3pro_driver.py``;
what is graded here is that the SAME driver, on ``transport="twin"``, puts the
same agent verbs onto the model in the model's units - degrees back to
radians, a body-frame twist rotated into the world-frame slides, the watchdog
emulated - and that the twin's graph makes ``get_observation`` a reading. The
real-MuJoCo half is ``tests_integ/simulation/test_yahboom_m3pro_twin.py``.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

from strands_robots._command_gate import COMMAND_ALLOW_ENV
from strands_robots.drivers.yahboom_m3pro import (
    CMD_VEL_WATCHDOG_S,
    GRIPPER_CLOSED_RAD,
    HOME_DEG,
    JOINT_STATES_TOPIC,
    PUBLISH_RATE_HZ,
    REQUIRED_TOPICS,
    YahboomM3ProDriver,
)
from strands_robots.drivers.yahboom_m3pro_twin import (
    BASE_ACTUATORS,
    TWIN_ENDPOINT,
    TWIN_GRAPH,
    M3ProTwinGraph,
    body_to_world,
    world_to_body,
    yaw_to_quaternion,
)

_DT = 0.002


class _FakeEngine:
    """A ``yahboom_m3pro`` sim engine double: records writes, integrates nothing but yaw."""

    def __init__(self, yaw: float = 0.0) -> None:
        self.writes: list[tuple[dict[str, float], int]] = []
        self.state: dict[str, float] = {
            "base_x": 0.0,
            "base_y": 0.0,
            "base_yaw": yaw,
            "arm1": 0.0,
            "arm2": math.pi / 6,
            "arm3": -math.pi / 2,
            "arm4": -math.pi / 2,
            "arm5": 0.0,
            "rlink1": 0.0,
            "llink1": 0.0,
        }
        self.destroyed = False
        self.refuse: str | None = None

    def list_robots(self) -> list[str]:
        return ["yahboom_m3pro"]

    def create_world(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": "world"}]}

    def add_robot(self, name: str, **kwargs: Any) -> dict[str, Any]:
        assert name == "yahboom_m3pro" and kwargs.get("keyframe") == "home"
        return {"status": "success", "content": [{"text": "added"}]}

    def physics_timestep(self) -> float:
        return _DT

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        assert robot_name == "yahboom_m3pro" and skip_images
        obs: dict[str, Any] = dict(self.state)
        obs.update({f"{k}.vel": 0.0 for k in self.state})
        obs["front"] = object()  # an image, which the twin must skip
        return obs

    def send_action(self, action: dict[str, float], robot_name: str, n_substeps: int) -> dict[str, Any]:
        assert robot_name == "yahboom_m3pro"
        if self.refuse:
            return {"status": "error", "content": [{"text": self.refuse}]}
        self.writes.append((dict(action), n_substeps))
        # Position servos arrive; velocity servos integrate over the substeps.
        for key, value in action.items():
            if key in BASE_ACTUATORS:
                self.state[key] += value * n_substeps * _DT
            elif key == "gripper":
                self.state["rlink1"] = value
                self.state["llink1"] = value
            else:
                self.state[key] = value
        return {"status": "success", "content": [{"text": f"Action applied ({len(action)} keys)."}]}

    def destroy(self) -> None:
        self.destroyed = True


@pytest.fixture
def engine() -> _FakeEngine:
    return _FakeEngine()


@pytest.fixture
def twin(engine: _FakeEngine) -> YahboomM3ProDriver:
    driver = YahboomM3ProDriver(transport="twin", sim=engine)
    assert driver.connect_eagerly() is None
    return driver


async def _invoke(driver: YahboomM3ProDriver, **request: Any) -> dict[str, Any]:
    tool_use: Any = {"toolUseId": "t1", "name": driver.tool_name, "input": request}
    return [event async for event in driver.stream(tool_use, {})][0]


def _base_writes(engine: _FakeEngine) -> list[tuple[dict[str, float], int]]:
    return [(a, n) for a, n in engine.writes if set(a) <= set(BASE_ACTUATORS)]


def _arm_writes(engine: _FakeEngine) -> list[tuple[dict[str, float], int]]:
    return [(a, n) for a, n in engine.writes if not set(a) <= set(BASE_ACTUATORS)]


# --------------------------------------------------------------------------- #
# The frames.                                                                 #
# --------------------------------------------------------------------------- #


class TestTheFrameMaths:
    def test_body_to_world_and_back_are_inverses(self) -> None:
        for yaw in (0.0, 0.7, -2.1, math.pi):
            wx, wy = body_to_world(0.3, -0.1, yaw)
            bx, by = world_to_body(wx, wy, yaw)
            assert (bx, by) == pytest.approx((0.3, -0.1))

    def test_a_quarter_turn_swaps_the_axes(self) -> None:
        assert body_to_world(1.0, 0.0, math.pi / 2) == pytest.approx((0.0, 1.0))
        assert body_to_world(0.0, 1.0, math.pi / 2) == pytest.approx((-1.0, 0.0))

    def test_yaw_becomes_a_unit_quaternion_about_z(self) -> None:
        q = yaw_to_quaternion(math.pi / 2)
        assert q["x"] == q["y"] == 0.0
        assert q["z"] ** 2 + q["w"] ** 2 == pytest.approx(1.0)
        assert 2 * math.atan2(q["z"], q["w"]) == pytest.approx(math.pi / 2)


# --------------------------------------------------------------------------- #
# The graph.                                                                  #
# --------------------------------------------------------------------------- #


class TestTheTwinIsTheRobotsGraph:
    def test_the_driver_connects_over_the_twin_without_a_bridge(self, twin: YahboomM3ProDriver) -> None:
        assert twin.is_connected and twin.endpoint == TWIN_ENDPOINT
        status = asyncio.run(twin.get_status())["content"][0]["json"]
        assert status["transport"] == "twin"
        for topic in REQUIRED_TOPICS:
            assert topic in status["topics"]
        assert JOINT_STATES_TOPIC in status["topics"], "the twin has joint state where the board has none"

    def test_the_twin_lists_every_robot_topic(self) -> None:
        names = {name for name, _ in TWIN_GRAPH}
        assert {"/cmd_vel", "/arm6_joints", "/arm_joint", "/odom_raw", "/imu/data_raw", "/scan0", "/scan1"} <= names

    def test_sim_is_only_the_twins_engine(self, engine: _FakeEngine) -> None:
        with pytest.raises(ValueError, match="transport='twin'"):
            YahboomM3ProDriver(sim=engine)
        assert YahboomM3ProDriver().sim is None
        assert YahboomM3ProDriver(transport="twin", sim=engine).sim is engine

    def test_a_build_failure_is_a_named_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.simulation as sim_mod

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise ImportError("mujoco is not installed")

        monkeypatch.setattr(sim_mod, "create_simulation", _boom)
        driver = YahboomM3ProDriver(transport="twin")
        reason = driver.connect_eagerly()
        assert reason is not None and "mujoco is not installed" in reason
        assert not driver.is_connected

    def test_the_gate_is_not_consulted_on_the_twin(
        self, twin: YahboomM3ProDriver, engine: _FakeEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No allowlist, no bypass, no agent: the twin still moves - it is not a physical surface."""
        monkeypatch.delenv(COMMAND_ALLOW_ENV, raising=False)
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        assert twin.move(0.2, duration_s=0.5)["status"] == "success"
        assert _base_writes(engine)


# --------------------------------------------------------------------------- #
# The arm, in the model's units.                                              #
# --------------------------------------------------------------------------- #


class TestTheArmReachesTheModel:
    def test_home_writes_the_keyframe_radians_and_steps_the_move_time(
        self, twin: YahboomM3ProDriver, engine: _FakeEngine
    ) -> None:
        assert twin.home(time_ms=1000)["status"] == "success"
        [(targets, substeps)] = _arm_writes(engine)
        assert targets == pytest.approx(
            {"arm1": 0.0, "arm2": math.pi / 6, "arm3": -math.pi / 2, "arm4": -math.pi / 2, "arm5": 0.0, "gripper": 0.0}
        )
        assert substeps == round(1.0 / _DT)

    def test_degrees_from_the_agent_land_as_radians_in_the_model(
        self, twin: YahboomM3ProDriver, engine: _FakeEngine
    ) -> None:
        result = asyncio.run(_invoke(twin, action="arm", joints=[45, 90, 90, 90, 270, 30], time_ms=500))
        assert result["status"] == "success"
        targets, substeps = _arm_writes(engine)[-1]
        assert targets["arm1"] == pytest.approx(-math.pi / 4)
        assert targets["arm5"] == pytest.approx(math.pi)
        assert targets["gripper"] == pytest.approx(GRIPPER_CLOSED_RAD)
        assert substeps == 250

    def test_get_observation_is_a_reading_on_the_twin(self, twin: YahboomM3ProDriver) -> None:
        before = twin.get_observation()
        assert before["arm3.pos"] == pytest.approx(-math.pi / 2)
        assert before["gripper.pos"] == 0.0 and "base_yaw.pos" in before
        twin.send_action({"arm1.pos": 0.4, "gripper.pos": -1.0})
        after = twin.get_observation()
        # The wire is whole degrees, so the reading is the target quantised to one.
        assert after["arm1.pos"] == pytest.approx(0.4, abs=math.radians(0.5))
        assert after["gripper.pos"] == pytest.approx(-1.0, abs=0.02), (
            "rlink1 reads back under the key send_action takes"
        )

    def test_a_radian_round_trips_through_degrees_within_a_degree(
        self, twin: YahboomM3ProDriver, engine: _FakeEngine
    ) -> None:
        """The wire is whole degrees, so the twin's target is quantised - by no more than that."""
        twin.send_action({"arm2.pos": 0.1234})
        assert _arm_writes(engine)[-1][0]["arm2"] == pytest.approx(0.1234, abs=math.radians(0.5))

    def test_an_engine_refusal_reaches_the_caller(self, twin: YahboomM3ProDriver, engine: _FakeEngine) -> None:
        engine.refuse = "unresolved_keys"
        result = twin.home()
        assert result["status"] == "error" and "unresolved_keys" in result["content"][0]["text"]


# --------------------------------------------------------------------------- #
# The base: body frame in, world frame on the slides, watchdog emulated.      #
# --------------------------------------------------------------------------- #


class TestTheBaseReachesTheModel:
    def test_a_held_move_streams_frames_then_the_watchdog_then_zero(
        self, twin: YahboomM3ProDriver, engine: _FakeEngine
    ) -> None:
        result = twin.move(0.3, 0.0, 0.0, duration_s=1.0)
        assert result["status"] == "success", result
        writes = _base_writes(engine)
        frames = int(PUBLISH_RATE_HZ)
        # The burst: one write per frame, each one publish period long.
        for targets, substeps in writes[:frames]:
            assert targets == pytest.approx({"base_x": 0.3, "base_y": 0.0, "base_yaw": 0.0})
            assert substeps == round(1.0 / PUBLISH_RATE_HZ / _DT)
        # Then the firmware's watchdog hold, then the zero.
        assert writes[frames][0]["base_x"] == pytest.approx(0.3)
        assert writes[frames][1] == round(CMD_VEL_WATCHDOG_S / _DT)
        assert writes[frames + 1][0] == {"base_x": 0.0, "base_y": 0.0, "base_yaw": 0.0}
        # The driver's own trailing stop is a zero twist: it writes zero and holds nothing.
        assert all(all(v == 0.0 for v in a.values()) for a, _ in writes[frames + 2 :])
        assert engine.state["base_x"] == pytest.approx(0.3 * (1.0 + CMD_VEL_WATCHDOG_S))

    def test_a_body_frame_strafe_is_rotated_by_the_current_yaw(self, engine: _FakeEngine) -> None:
        engine.state["base_yaw"] = math.pi / 2
        driver = YahboomM3ProDriver(transport="twin", sim=engine)
        driver.connect_eagerly()
        driver.send_action({"linear.y": 0.2})
        targets, _ = _base_writes(engine)[0]
        # +y in the body frame, facing +y in the world, is -x in the world.
        assert targets == pytest.approx({"base_x": -0.2, "base_y": 0.0, "base_yaw": 0.0}, abs=1e-9)

    def test_a_single_frame_is_a_watchdog_twitch(self, twin: YahboomM3ProDriver, engine: _FakeEngine) -> None:
        twin.send_action({"linear.x": 0.5})
        writes = _base_writes(engine)
        assert [n for _, n in writes] == [round(1.0 / PUBLISH_RATE_HZ / _DT), round(CMD_VEL_WATCHDOG_S / _DT), 1]
        assert writes[-1][0] == dict.fromkeys(BASE_ACTUATORS, 0.0)

    def test_a_zero_twist_does_not_hold_for_the_watchdog(self, twin: YahboomM3ProDriver, engine: _FakeEngine) -> None:
        twin.stop_task()
        writes = _base_writes(engine)
        assert len(writes) == 2  # the frame and the zero; no watchdog hold of a zero
        assert all(all(v == 0.0 for v in a.values()) for a, _ in writes)

    def test_odometry_and_imu_are_read_from_the_base_joints(
        self, twin: YahboomM3ProDriver, engine: _FakeEngine
    ) -> None:
        engine.state.update({"base_x": 1.5, "base_y": -0.25, "base_yaw": math.pi / 2})
        state = twin.read_state()
        odom, imu = state["odom"], state["imu"]
        assert odom["pose"]["pose"]["position"] == {"x": 1.5, "y": -0.25, "z": 0.0}
        assert 2 * math.atan2(
            odom["pose"]["pose"]["orientation"]["z"], odom["pose"]["pose"]["orientation"]["w"]
        ) == pytest.approx(math.pi / 2)
        assert imu["linear_acceleration"]["z"] == pytest.approx(9.80665)
        sensors = asyncio.run(_invoke(twin, action="sensors"))
        assert sensors["status"] == "success" and "odom x 1.5" in sensors["content"][0]["text"]

    def test_the_lidars_are_listed_but_silent(self, twin: YahboomM3ProDriver, engine: _FakeEngine) -> None:
        graph = M3ProTwinGraph(engine)
        reply = graph("echo", topic="/scan0", count=1)
        assert reply["status"] == "success" and "no messages" in reply["content"][0]["text"]

    def test_a_sensor_topic_cannot_be_published_and_a_foreign_one_is_refused(self, engine: _FakeEngine) -> None:
        graph = M3ProTwinGraph(engine)
        assert graph("publish", topic="/odom_raw", fields={})["status"] == "error"
        assert graph("publish", topic="/elsewhere", fields={})["status"] == "error"
        assert graph("echo", topic="/elsewhere")["status"] == "error"
        assert graph("dance")["status"] == "error"


# --------------------------------------------------------------------------- #
# Lifecycle.                                                                  #
# --------------------------------------------------------------------------- #


class TestLifecycle:
    def test_cleanup_stops_and_leaves_a_callers_engine_alive(
        self, twin: YahboomM3ProDriver, engine: _FakeEngine
    ) -> None:
        twin.cleanup()
        assert not twin.is_connected
        assert engine.destroyed is False, "the caller built it, the caller destroys it"
        assert twin.sim is None

    def test_an_engine_the_twin_built_is_destroyed_on_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.simulation as sim_mod

        built = _FakeEngine()
        monkeypatch.setattr(sim_mod, "create_simulation", lambda *a, **k: built)
        driver = YahboomM3ProDriver(transport="twin")
        assert driver.connect_eagerly() is None
        assert driver.sim is built
        driver.cleanup()
        assert built.destroyed is True

    def test_every_agent_verb_runs_on_the_twin(self, twin: YahboomM3ProDriver, engine: _FakeEngine) -> None:
        for request in (
            {"action": "status"},
            {"action": "sensors"},
            {"action": "arm", "joints": list(HOME_DEG), "time_ms": 300},
            {"action": "gripper", "open": False},
            {"action": "home"},
            {"action": "move", "linear_x": 0.1, "angular_z": 0.2, "duration_s": 0.3},
            {"action": "stop"},
        ):
            result = asyncio.run(_invoke(twin, **request))
            assert result["status"] == "success", (request, result)
        assert len(_arm_writes(engine)) == 3
        assert any(a.get("base_yaw") == pytest.approx(0.2) for a, _ in _base_writes(engine))
