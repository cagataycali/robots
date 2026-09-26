"""One stand-in for the simulation engine the RL stack drives.

Every RL surface - ``SimEnv``, ``GymSimEnv``, ``VecSimEnv``, the trainers and
``evaluate`` - reaches its backend through five methods: ``list_robots``,
``robot_action_keys``, ``get_observation``, ``reset`` and ``send_action``.
:class:`~strands_robots.simulation.base.SimEngine` publishes all five, and
``robot_action_keys`` is the one that is NOT abstract: the base defines it to
mirror ``robot_joint_names``, which is what every backend whose actuator set
matches its joint set inherits.

A duck-typed double does not inherit that default, so it has to restate it -
and a double that restates a signature is bound to nothing: a renamed keyword
or a narrowed argument on the seam leaves it green. This stand-in subclasses
the seam instead, so mypy checks every override against the published
signature, the action-key default is the production one, and a method the RL
stack does not reach raises instead of quietly answering.

The integrator is the one the RL tests need: each joint advances by
``gain * command`` plus ``drift`` per step, and ``<joint>.vel`` reports the
last advance. The refusals are the backend rules rather than test-local
inventions - a vector whose width is not the action-key count is refused with
the message the backends publish, and a non-finite command is refused rather
than clamped (``SimEngine._coerce_action``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from strands_robots.simulation.base import SimEngine


class EngineStandIn(SimEngine):
    """A joint-integrating engine bound to the ``SimEngine`` seam.

    Args:
        robots: Names ``list_robots`` publishes. Empty models a world with no
            registered robot, where ``SimEnv`` must be given ``action_dim``.
        joints: Joint names. Each is observed as ``<name>``, its velocity as
            ``<name>.vel``.
        action_keys: The actuator vocabulary ``send_action`` binds a vector
            against. ``None`` keeps the inherited default, which mirrors the
            joint names; a list models a backend whose actuators are spelled
            differently or are fewer than its joints (a tendon gripper).
        gain: Command coupling - a step advances each joint by ``gain`` times
            its commanded value.
        drift: Free-running advance per step, applied whatever the command is.
            With ``gain=0`` the joints are a step counter at constant velocity.
        extra_obs: Extra observation entries, e.g. a privileged key a critic
            reads in sim and a deployed policy cannot.
        refuse: When set, every ``send_action`` is refused with this text.
    """

    def __init__(
        self,
        *,
        robots: Sequence[str] = ("fake",),
        joints: Sequence[str] = ("J",),
        action_keys: Sequence[str] | None = None,
        gain: float = 0.1,
        drift: float = 0.0,
        extra_obs: Mapping[str, float] | None = None,
        refuse: str | None = None,
    ) -> None:
        self._robots = list(robots)
        self._joints = list(joints)
        self._action_keys = None if action_keys is None else list(action_keys)
        self._gain = float(gain)
        self._drift = float(drift)
        self._extra_obs = dict(extra_obs or {})
        self._refuse = refuse
        #: Every seam method the caller reached, in order.
        self.calls: list[str] = []
        #: Every action vector that arrived, refused ones included.
        self.sent: list[list[float]] = []
        #: The status returned for each, so a refusal is observable.
        self.statuses: list[str] = []
        #: Widths refused for not matching the action-key count.
        self.refused_widths: list[int] = []
        self.resets = 0
        self._reset_joints()

    def _reset_joints(self) -> None:
        self._position = dict.fromkeys(self._joints, 0.0)
        self._velocity = dict.fromkeys(self._joints, self._drift)

    def _vocabulary(self) -> list[str]:
        """The action keys, read without recording a call the caller did not make.

        Mirrors what the inherited ``robot_action_keys`` default resolves, so
        ``send_action`` binds a vector against the same list a caller sees.
        """
        return list(self._joints if self._action_keys is None else self._action_keys)

    # -- the surface the RL stack reaches -- #

    def list_robots(self) -> list[str]:
        self.calls.append("list_robots")
        return list(self._robots)

    def robot_joint_names(self, robot_name: str) -> list[str]:
        self.calls.append("robot_joint_names")
        return list(self._joints)

    def robot_action_keys(self, robot_name: str) -> list[str]:
        self.calls.append("robot_action_keys")
        if self._action_keys is None:
            # The production default, inherited rather than restated.
            return super().robot_action_keys(robot_name)
        return list(self._action_keys)

    def reset(self) -> dict[str, Any]:
        self.calls.append("reset")
        self.resets += 1
        self._reset_joints()
        return {"status": "success"}

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        self.calls.append("get_observation")
        obs: dict[str, Any] = dict(self._position)
        obs.update({f"{name}.vel": value for name, value in self._velocity.items()})
        obs.update(self._extra_obs)
        return obs

    def send_action(
        self,
        action: dict[str, Any] | Sequence[float],
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> dict[str, Any]:
        self.calls.append("send_action")
        keys = self._vocabulary()
        values = [float(action[key]) for key in keys] if isinstance(action, Mapping) else [float(v) for v in action]
        self.sent.append(values)

        if self._refuse is not None:
            return self._error(self._refuse)
        if len(values) != len(keys):
            self.refused_widths.append(len(values))
            return self._error(
                f"send_action: action vector length {len(values)} does not match robot "
                f"'{self._robots[0] if self._robots else ''}' action-key count {len(keys)}."
            )
        if not all(math.isfinite(value) for value in values):
            return self._error("send_action: a non-finite command is refused, not clamped.")

        for key, value in zip(keys, values, strict=True):
            advance = self._drift + self._gain * value
            # A key that names an actuator driving no joint of its own name
            # (a tendon) moves nothing here, exactly as it moves no joint of
            # that name on a backend.
            if key in self._position:
                self._position[key] += advance
                self._velocity[key] = advance
        self.statuses.append("success")
        return {"status": "success", "content": [{"text": f"Action applied ({len(values)} keys)."}]}

    @property
    def applied(self) -> list[list[float]]:
        """The vectors that were applied - the arrivals that were not refused."""
        return [values for values, status in zip(self.sent, self.statuses, strict=True) if status == "success"]

    def _error(self, text: str) -> dict[str, Any]:
        self.statuses.append("error")
        return {"status": "error", "content": [{"text": text}]}

    # -- the surface the RL stack does not reach -- #
    #
    # Concrete so the class is instantiable, and refusing so a new production
    # call cannot be answered by a fake nobody taught the behaviour. Permissive
    # signatures keep them Liskov-safe without restating each contract.

    def _unreached(self, name: str) -> dict[str, Any]:
        raise AssertionError(f"the RL stack does not reach SimEngine.{name}")

    def create_world(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._unreached("create_world")

    def destroy(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._unreached("destroy")

    def get_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._unreached("get_state")

    def add_robot(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._unreached("add_robot")

    def remove_robot(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._unreached("remove_robot")

    def add_object(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._unreached("add_object")

    def remove_object(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._unreached("remove_object")

    def render(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._unreached("render")

    def step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._unreached("step")
