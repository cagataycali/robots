# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared rule for reading a joint-state ordering out of an observation.

A ``Policy`` whose caller declared no ``robot_state_keys`` has to infer which
observation keys form this step's state vector, and the only ordering available
is the observation's own insertion order. The sim backends emit a velocity
companion beside every joint position (:mod:`strands_robots.simulation.mujoco.rendering`
writes ``obs[jnt_name] = qpos`` and then ``obs[f"{jnt_name}.vel"] = qvel``, calling
the second an "Additive key ... existing position-only" consumer contract), so that
order alternates ``[pos0, vel0, pos1, vel1, ...]``. Feeding it to a policy
trained on positions puts a velocity in every other slot and, once the vector is
truncated to the model's state dim, drops the trailing joints entirely - a wrong
state vector with no error raised.

:func:`drop_velocity_siblings` restores the producer's additive contract for
every consumer that infers an ordering this way. It is shared rather than
re-derived per provider because two providers conforming to the same ``Policy``
contract must read the same observation as the same state vector: the LeRobot
provider already filtered here and the Cosmos 3 provider did not, so one
interleaved velocities into its 7-joint request while the other did not, from
the same observation.

Explicitly configured ``robot_state_keys`` are never filtered - see the function
docstring for why an operator naming ``elbow.vel`` is stating the model's input.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from typing import Any

#: Suffix the sim backends append to a joint's additive velocity companion.
VELOCITY_SUFFIX = ".vel"

#: Flat state vector a direct-API caller / a dataset row carries.
FLAT_STATE_KEY = "observation.state"


def drop_velocity_siblings(scalar_keys: list[str]) -> list[str]:
    """Drop each ``<joint>.vel`` whose ``<joint>`` position companion is present.

    Used only for an OBSERVATION-DERIVED state ordering, never for an ordering
    the caller declared. An operator naming ``elbow.vel`` in
    ``robot_state_keys`` is stating the model's input; this only cleans up an
    ordering inferred from whatever the observation happened to contain.

    Pairing is decided per key, not by suffix alone. A ``.vel`` key with NO
    position companion is KEPT, because some embodiments legitimately declare
    velocity state and dropping it would corrupt those instead:
    ``embodiments.json`` gives LeKiwi body-frame base velocities ``x.vel`` /
    ``y.vel`` / ``theta.vel`` with no ``x`` / ``y`` / ``theta`` position key.

    Args:
        scalar_keys: Candidate state keys in observation insertion order.

    Returns:
        The same list, order preserved, minus the paired velocity siblings.
    """
    present = set(scalar_keys)
    return [k for k in scalar_keys if not (k.endswith(VELOCITY_SUFFIX) and k[: -len(VELOCITY_SUFFIX)] in present)]


def observation_joint_keys(observation: Mapping[str, Any], robot_state_keys: Sequence[str] = ()) -> list[str]:
    """The per-joint scalar keys of an observation, in the order they form a vector.

    ``robot_state_keys`` wins when the caller declared one and the observation
    carries every name in it. Otherwise the ordering is the observation's own
    insertion order over its numeric scalars, minus the velocity siblings -
    the rule :func:`drop_velocity_siblings` states.

    Args:
        observation: The observation dict handed to ``get_actions``.
        robot_state_keys: Ordering declared through ``set_robot_state_keys``.

    Returns:
        The joint keys in vector order; empty when the observation carries none.
    """
    names = list(robot_state_keys)
    if names and all(name in observation for name in names):
        return names
    return drop_velocity_siblings(
        [k for k, v in observation.items() if isinstance(v, numbers.Real) and not isinstance(v, bool)]
    )


def joint_positions_from_observation(
    observation: Mapping[str, Any], robot_state_keys: Sequence[str] = ()
) -> list[float] | None:
    """Read this step's joint positions out of an observation.

    One state arrives in two shapes, and a provider that reads only one of them
    reads the other as no state at all: the flat ``observation.state`` vector a
    direct-API caller passes, and the per-joint scalars the sim backends emit
    (:mod:`strands_robots.simulation.mujoco.rendering` writes ``obs[joint] =
    qpos`` beside ``obs[f"{joint}.vel"] = qvel``, and writes no flat vector).
    The planner providers read only the flat key, so every simulated rollout
    planned from no start state at all.

    The flat vector wins when present; otherwise the scalars are read in
    :func:`observation_joint_keys` order.

    Args:
        observation: The observation dict handed to ``get_actions``.
        robot_state_keys: Ordering declared through ``set_robot_state_keys``.

    Returns:
        The joint positions in that order, or ``None`` when the observation
        carries neither shape.

    Raises:
        TypeError: If the flat value is not iterable, or a value under it is
            not a number. The caller decides whether that degrades or refuses.
        ValueError: If a value cannot be read as a float.
    """
    flat = observation.get(FLAT_STATE_KEY)
    if flat is not None:
        if hasattr(flat, "tolist"):
            flat = flat.tolist()
        return [float(x) for x in flat]
    names = observation_joint_keys(observation, robot_state_keys)
    if not names:
        return None
    return [float(observation[k]) for k in names]
