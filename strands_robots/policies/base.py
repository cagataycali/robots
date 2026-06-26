"""Abstract base class for robot policies (VLA, motion planners, MPC, scripted).

The :class:`Policy` ABC is intentionally agnostic about *how* actions are
produced.  Built-in providers (`mock`, `groot`, `lerobot_local`) are VLA-style,
but the same interface is the right shape for:

* **Classical motion planners** - cuRobo, MoveIt2, OMPL, RRT*: take a goal
  pose and joint state, return a collision-free trajectory.
* **Model-predictive controllers** (MPC) - solve a finite-horizon optimal
  control problem each tick.
* **Scripted / pure-IK trajectories** - analytic IK followed by interpolation;
  zero learning involved.

Non-VLA implementations typically set :attr:`Policy.requires_images` to
``False`` to skip camera rendering (~10x throughput win at 500Hz) and read
their goal from the well-known ``**kwargs`` keys documented on
:meth:`Policy.get_actions` rather than parsing the natural-language
``instruction`` string.

See ``MockPolicy`` (``strands_robots/policies/mock.py``) for the canonical
non-VLA reference implementation.
"""

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


class Policy(ABC):
    """Abstract base class for robot policies (VLA, motion planners, MPC, scripted).

    All policies implement async :meth:`get_actions`.  For convenience, a
    synchronous wrapper :meth:`get_actions_sync` is provided.

    The interface is general enough to cover both **VLA-style** providers
    (consume images + instruction, output joint targets) and **non-VLA**
    providers such as classical motion planners (cuRobo, MoveIt2),
    model-predictive controllers, and pure-IK / scripted trajectories.
    Non-VLA providers typically set :attr:`requires_images` to ``False``
    and read their goal from the well-known ``**kwargs`` keys documented
    on :meth:`get_actions`.

    All providers MUST honour the per-tick **action value convention**
    documented on :meth:`get_actions`: each action value is a python
    ``float`` (single-DOF) or ``list[float]`` (multi-DOF group), never a
    raw ``np.ndarray``, so downstream consumers handle every provider's
    output uniformly regardless of its internal compute backend. See
    ``MockPolicy`` for the canonical reference.
    """

    @abstractmethod
    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Get actions from policy given observation and instruction.

        Args:
            observation_dict: Robot observation (cameras + state).  VLA
                providers consume both ``observation.images.*`` and
                ``observation.state``.  Non-VLA providers typically
                consume ``observation.state`` only and set
                :attr:`requires_images` to ``False`` to skip camera
                rendering.
            instruction: Natural language instruction.  Required by the
                signature for VLA providers; non-VLA providers (motion
                planners, MPC, scripted) may ignore it and read the goal
                from ``**kwargs`` instead.
            **kwargs: Provider-specific parameters.  The following keys
                are **well-known** and SHOULD be honoured by non-VLA
                providers when present so callers don't have to JSON-encode
                goals into the ``instruction`` string:

                - ``target_pose: list[float]`` - Cartesian goal as
                  ``[x, y, z, qw, qx, qy, qz]`` (position in metres,
                  orientation as a unit quaternion in the robot base frame).
                - ``target_joints: dict[str, float]`` - joint-space goal
                  keyed by joint name; values are in radians (revolute) or
                  metres (prismatic).
                - ``world_update: dict | None`` - per-call world refresh
                  for collision-aware planners (e.g. point cloud / depth
                  image / mesh updates).  ``None`` means "reuse the world
                  configured at init time".

                Providers MUST ignore unknown ``**kwargs`` rather than
                raising, so callers can pass shared keys across providers.

        Returns:
            List of action dicts for robot execution.  Each dict maps a
            robot state key (joint/actuator name) to its **target value**
            for that tick.

            Values MUST be **JSON / python-native**: a python ``float`` for
            a single-DOF actuator, or a ``list[float]`` for a multi-DOF
            actuator group.  Implementations MUST NOT return raw
            ``np.ndarray`` objects -- coerce with ``.tolist()`` /
            ``float(...)`` before returning -- so downstream consumers can
            treat every provider's output uniformly (e.g. ``float(v)`` on a
            scalar, ``len(v)`` on a group) regardless of the policy's
            internal compute backend.

            The list length is the action-chunk horizon; consumers execute
            it at a fixed control rate (e.g. 50Hz).
        """
        pass

    def get_actions_sync(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Synchronous convenience wrapper around get_actions().

        Safe to call from sync code, event loops, or notebooks.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.get_actions(observation_dict, instruction, **kwargs),
                ).result()
        else:
            return asyncio.run(self.get_actions(observation_dict, instruction, **kwargs))

    @abstractmethod
    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Configure the policy with robot state keys."""
        pass

    def reset(self, seed: int | None = None) -> None:
        """Reset per-episode policy state.

        Default implementation is a no-op. Policies that hold per-episode
        state (e.g. diffusion sampler RNG, action chunk caches, KV-caches)
        should override to apply the reset.

        For SERVICE-mode policies (e.g. ``Gr00tPolicy(host=...)`` over
        ZMQ), the override forwards the call to the server so its
        per-episode RNG state can be re-initialised - without this,
        ``set_eval_seed`` only seeds the client-side process, leaving
        the server's diffusion sampler RNG drifting across calls and
        breaking reproducibility (#187).

        Args:
            seed: Optional master seed forwarded to the policy's
                random-number generators. When ``None``, implementations
                may apply a default seed or leave RNG state untouched.
        """
        # Default no-op. Concrete policies override to apply per-episode
        # state reset (RNG seeding, action-cache flush, server-side
        # reset endpoint call, etc.).
        return None

    @property
    def requires_images(self) -> bool:
        """Whether this policy needs camera frames in its observation.

        Default ``True`` (most VLA policies do). Subclasses that only
        consume joint state (e.g. ``MockPolicy``, classical motion planners
        such as cuRobo / MoveIt2, MPC, pure-IK controllers, scripted
        trajectories) can return ``False`` to let the simulation skip
        expensive camera rendering - a ~10x throughput win at 500Hz when
        no cameras are needed.
        """
        return True

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name for identification."""
        pass


@runtime_checkable
class ChunkedPolicy(Protocol):
    """Introspection contract for policies that emit ACTION CHUNKS.

    A *chunked* policy returns more than one action per
    :meth:`Policy.get_actions` call: a model trained for N-step open-loop replay
    (ACT, diffusion, pi0, SmolVLA, MolmoAct2) emits a length-N chunk that a
    consumer executes before re-querying. The chunk PRODUCER is the existing
    async :meth:`Policy.get_actions` - this protocol deliberately does NOT add a
    second chunk-producing method (that would split one contract across two
    code paths); it only surfaces the metadata a consumer needs to drive an
    already-produced chunk correctly.

    Every consumer of a chunk (the single-policy runner, the multi-episode eval
    loop, and the synchronized multi-robot loop) must size the chunk the same
    way - see :func:`resolve_chunk_length`. Routing all of them through one
    helper that reads this contract keeps a chunk-emitting policy from being
    truncated differently depending on which loop happens to drive it.

    The protocol is ``runtime_checkable`` so a consumer can branch on
    ``isinstance(policy, ChunkedPolicy)`` and a type checker rejects a
    non-chunked policy where a chunked one is required.

    Attributes:
        actions_per_step: Number of actions the policy intends a consumer to
            execute open-loop from one ``get_actions`` chunk before re-querying
            (the policy's trained chunk length). Truncating below this drops the
            chunk tail and forces an out-of-distribution re-query.
        supports_rtc: Whether the policy blends chunk seams internally via
            Real-Time Chunking - it carries prev-chunk state across re-queries
            so consecutive chunks join smoothly. Introspection only; a consumer
            never has to drive RTC, the policy does it inside ``get_actions``.
    """

    actions_per_step: int
    supports_rtc: bool


def resolve_chunk_length(policy: "Policy", action_horizon: int) -> int:
    """Effective number of actions to consume from one ``get_actions`` chunk.

    Centralizes the single chunk-length rule every consumer must apply
    identically: consume ``max(action_horizon, policy.actions_per_step)``
    actions before re-querying. A policy trained for N-step open-loop replay
    (``actions_per_step == N``) must have its FULL chunk consumed; clamping to a
    smaller ``action_horizon`` drops the chunk tail and forces an
    out-of-distribution re-query. Policies that do not declare
    ``actions_per_step`` (single-action providers such as ``MockPolicy``) behave
    as a 1-action chunk, so the result is just ``max(action_horizon, 1)``.

    Before this helper existed each consumer inlined the same
    ``max(action_horizon, getattr(policy, "actions_per_step", 1))`` expression,
    and they drifted: the synchronized multi-robot loop truncated to
    ``action_horizon`` alone, silently dropping a chunk-emitting policy's tail
    while the single-policy runner consumed the full chunk.

    Args:
        policy: Any policy. Its chunk length is read from the optional
            :class:`ChunkedPolicy` ``actions_per_step`` attribute; a policy that
            does not declare it is treated as single-action.
        action_horizon: Consumer-requested actions per chunk (clamped to >= 1).

    Returns:
        The number of leading chunk actions to execute before re-querying.
    """
    intended = getattr(policy, "actions_per_step", 1)
    try:
        intended_int = int(intended)
    except (TypeError, ValueError):
        intended_int = 1
    if intended_int < 1:
        intended_int = 1
    return max(int(action_horizon), 1, intended_int)
