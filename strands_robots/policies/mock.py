"""Mock policy for testing - generates smooth sinusoidal trajectories."""

import logging
import math
from typing import Any, ClassVar

from strands_robots.policies.base import Policy
from strands_robots.utils import name_list_error, sequence_length

logger = logging.getLogger(__name__)


class MockPolicy(Policy):
    """Mock policy for testing - generates smooth sinusoidal trajectories."""

    def __init__(self, **kwargs: Any) -> None:
        self.robot_state_keys: list[str] = []
        self._step = 0
        self._ctrl_bounds: dict[str, tuple[float, float]] = {}
        logger.info("Mock Policy initialized")

    @property
    def provider_name(self) -> str:
        """Provider name for identification (always ``"mock"``)."""
        return "mock"

    @property
    def requires_images(self) -> bool:
        """Mock policy only consumes joint state - skip camera rendering."""
        return False

    #: ``False``: every joint follows a sinusoid; ``instruction`` is never read.
    reads_instruction: ClassVar[bool] = False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Record the ordered joint keys used to name the sinusoidal action dict.

        Raises:
            ValueError: If ``robot_state_keys`` is not an ordered list of
                distinct non-blank names, per
                :func:`~strands_robots.utils.name_list_error`. A single name
                passed as a bare string is the mistake this catches: ``str`` is
                iterable per character, so it would bind one joint per letter.
        """
        if robot_state_keys and (
            error := name_list_error(robot_state_keys, "robot_state_keys", "set_robot_state_keys")
        ):
            raise ValueError(error)
        self.robot_state_keys = robot_state_keys

    def set_sim_context(self, model: Any, namespace: str) -> None:
        """Learn each driven actuator's ctrlrange so the sinusoid stays inside it.

        Called by the MuJoCo engine's ``bind_policy_sim_context`` right after
        :meth:`set_robot_state_keys`, with the compiled ``MjModel`` and the
        robot's namespace prefix (``"so100/"``). The mock's ±0.5 rad sinusoid
        was written for a generic joint; on a real model some actuators do not
        span it - the SO-100 ``Pitch`` ctrlrange is ``[-3.32, 0.174]`` and its
        ``Jaw`` is ``[-0.174, 1.75]`` - so MuJoCo clamped the value and the
        engine warned that the commanded trajectory was NOT reproduced. That
        warning was the first thing ``examples/01_sim_hello_world.py`` printed.
        Knowing the ranges, the mock clips its own output so what it commands
        is what the actuator does. Unlimited actuators and names that resolve
        to no actuator are left alone; an error while reading the model leaves
        the policy exactly as configured, never fails the rollout.
        """
        bounds: dict[str, tuple[float, float]] = {}
        try:
            import mujoco  # noqa: PLC0415 - optional sim dependency

            for key in self.robot_state_keys:
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{namespace}{key}")
                if act_id < 0 or not bool(model.actuator_ctrllimited[act_id]):
                    continue
                lo, hi = (float(v) for v in model.actuator_ctrlrange[act_id])
                if hi > lo:
                    bounds[key] = (lo, hi)
        except Exception as exc:  # noqa: BLE001 - best-effort, mirrors the engine's binding
            logger.debug("MockPolicy.set_sim_context could not read ctrlranges: %s", exc)
            return
        self._ctrl_bounds = bounds

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Return smooth sinusoidal actions.

        Canonical reference for the per-tick action value convention
        documented on :meth:`Policy.get_actions`: every value is a python
        ``float`` (single-DOF joint target), never a raw ``np.ndarray``.
        """
        if not self.robot_state_keys:
            if "observation.state" in observation_dict:
                state = observation_dict["observation.state"]
                # ``sequence_length`` rather than a ``hasattr(state,
                # "__len__")`` probe: a 0-d array declares ``__len__`` and
                # raises from it, so the probe passes and ``len()`` escapes
                # past the ``else`` written for exactly this value - a state
                # that does not carry a width (#1883, the rule from #1844).
                n_components = sequence_length(state)
                dim = 6 if n_components is None else n_components
            else:
                dim = 6
            self.robot_state_keys = [f"joint_{i}" for i in range(dim)]

        mock_actions = []
        for i in range(8):
            action_dict = {}
            t = (self._step + i) * 0.02
            for j, key in enumerate(self.robot_state_keys):
                freq = 0.3 + j * 0.15
                phase = j * math.pi / 3
                value = 0.5 * math.sin(2 * math.pi * freq * t + phase)
                bound = self._ctrl_bounds.get(key)
                if bound is not None:
                    value = min(max(value, bound[0]), bound[1])
                action_dict[key] = value
            mock_actions.append(action_dict)

        self._step += len(mock_actions)
        return mock_actions
