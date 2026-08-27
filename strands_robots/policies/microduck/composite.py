"""Hot-swap FSM over several Microduck ONNX policies.

A single Microduck robot ships many skills as separate ONNX files
(``alpha_walking``, ``alpha_stand``, ``roulade``, ``ball_kick_*`` ...). A
:class:`MicroduckPolicyBundle` holds several :class:`MicroduckPolicy` instances
warm and delegates each tick to the ACTIVE one, so a controller can switch skill
mid-rollout (walk -> kick -> walk) without tearing down and rebuilding sessions.

Switching is explicit: the caller names the next skill via
``get_actions(select=...)`` (or :meth:`switch`). When ``switch_on_velocity`` is
set, the bundle also auto-selects between a ``move_key`` and an ``idle_key`` by
the magnitude of the twist command - the same walking<->standing gate Pollen's
``infer_policy.py`` uses. The previously active child's ``last_action`` history
is left intact so returning to a skill resumes cleanly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from strands_robots.policies.base import Policy

from .policy import MicroduckPolicy


class MicroduckPolicyBundle(Policy):
    """A named collection of Microduck skills with a single active policy.

    Args:
        policies: Mapping of skill name -> :class:`MicroduckPolicy`.
        active: The initially selected skill name. Defaults to the first key.
        switch_on_velocity: If set, auto-switch between ``move_key`` and
            ``idle_key`` by ``|twist|`` against this threshold each tick.
        move_key / idle_key: Skill names for the velocity gate (default
            ``"walk"`` / ``"stand"`` when those keys exist).
    """

    requires_images = False

    def __init__(
        self,
        policies: dict[str, MicroduckPolicy],
        *,
        active: str | None = None,
        switch_on_velocity: float | None = None,
        move_key: str = "walk",
        idle_key: str = "stand",
    ) -> None:
        if not policies:
            raise ValueError("MicroduckPolicyBundle requires at least one policy.")
        for name, pol in policies.items():
            if not isinstance(pol, MicroduckPolicy):
                raise TypeError(
                    f"MicroduckPolicyBundle: policy {name!r} is {type(pol).__name__}, expected MicroduckPolicy."
                )
        self._policies = dict(policies)
        first = next(iter(self._policies))
        self._active = active or first
        if self._active not in self._policies:
            raise ValueError(
                f"MicroduckPolicyBundle: active skill {self._active!r} is not one of {list(self._policies)}."
            )
        self._switch_on_velocity = float(switch_on_velocity) if switch_on_velocity is not None else None
        self._move_key = move_key
        self._idle_key = idle_key

    @property
    def provider_name(self) -> str:
        """Registry key for this provider (``"microduck_bundle"``)."""
        return "microduck_bundle"

    @property
    def active(self) -> str:
        """The currently selected skill name."""
        return self._active

    @property
    def children(self) -> tuple[Policy, ...]:
        """Every held skill, so a capability probe can walk to the leaf policy."""
        return tuple(self._policies.values())

    def switch(self, name: str) -> None:
        """Select ``name`` as the active skill."""
        if name not in self._policies:
            raise ValueError(f"MicroduckPolicyBundle: unknown skill {name!r}; have {list(self._policies)}.")
        self._active = name

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Forward the robot's joint list to every held skill."""
        for pol in self._policies.values():
            pol.set_robot_state_keys(robot_state_keys)

    def set_control_frequency(self, hz: float) -> None:
        """Forward the control rate to every held skill."""
        super().set_control_frequency(hz)
        for pol in self._policies.values():
            pol.set_control_frequency(hz)

    def reset(self, seed: int | None = None) -> None:
        """Reset every held skill's per-episode state."""
        for pol in self._policies.values():
            pol.reset(seed)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Delegate this tick to the active skill, after any requested switch."""
        select = kwargs.get("select")
        if select is not None:
            self.switch(str(select))
        elif self._switch_on_velocity is not None:
            self._auto_switch(kwargs.get("target_velocity"))
        return await self._policies[self._active].get_actions(observation_dict, instruction, **kwargs)

    def _auto_switch(self, target_velocity: Any) -> None:
        """Gate move<->idle by twist magnitude, when both keys exist."""
        if self._move_key not in self._policies or self._idle_key not in self._policies:
            return
        if target_velocity is None or self._switch_on_velocity is None:
            return
        mag = float(np.linalg.norm(np.asarray(target_velocity, dtype=np.float32).reshape(-1)[:3]))
        self._active = self._move_key if mag >= self._switch_on_velocity else self._idle_key
