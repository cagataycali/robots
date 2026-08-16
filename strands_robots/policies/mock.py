"""Mock policy for testing - generates smooth sinusoidal trajectories."""

import logging
import math
from typing import Any

from strands_robots.policies.base import Policy
from strands_robots.utils import name_list_error, sequence_length

logger = logging.getLogger(__name__)


class MockPolicy(Policy):
    """Mock policy for testing - generates smooth sinusoidal trajectories."""

    def __init__(self, **kwargs: Any) -> None:
        self.robot_state_keys: list[str] = []
        self._step = 0
        logger.info("Mock Policy initialized")

    @property
    def provider_name(self) -> str:
        """Provider name for identification (always ``"mock"``)."""
        return "mock"

    @property
    def requires_images(self) -> bool:
        """Mock policy only consumes joint state - skip camera rendering."""
        return False

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
                action_dict[key] = 0.5 * math.sin(2 * math.pi * freq * t + phase)
            mock_actions.append(action_dict)

        self._step += len(mock_actions)
        return mock_actions
