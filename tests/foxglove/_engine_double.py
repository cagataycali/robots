"""A minimal concrete ``SimEngine`` for exercising the telemetry seam without MuJoCo."""

from __future__ import annotations

from typing import Any

from strands_robots.simulation.base import SimEngine


class TelemetryEngine(SimEngine):
    """One robot, one observation dict, the telemetry helper and nothing else."""

    def __init__(
        self, observation: dict[str, Any], *, joint_names: list[str] | None = None, **bridge_kwargs: Any
    ) -> None:
        self._obs = observation
        self._joint_names = ["shoulder_pan", "elbow"] if joint_names is None else joint_names
        self.tool_name_str = "double"
        self._init_ros_bridge(**bridge_kwargs)

    def list_robots(self) -> list[str]:
        return ["so101"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._joint_names)

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        if skip_images:
            return {k: v for k, v in self._obs.items() if not hasattr(v, "ndim")}
        return self._obs

    def set_joint_positions(self, positions: Any, robot_name: str | None = None, hold: bool = False) -> dict[str, Any]:
        self.last_command = (robot_name, dict(positions), hold)
        return {"status": "success", "content": [{"text": f"Set {len(positions)} joint positions"}]}

    def create_world(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def destroy(self) -> dict[str, Any]:
        raise NotImplementedError

    def reset(self) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def add_robot(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def remove_robot(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def add_object(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def remove_object(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def render(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def send_action(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def physics_timestep(self) -> float:
        return 0.002
