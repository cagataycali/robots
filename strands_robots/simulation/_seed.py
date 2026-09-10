# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The seed domain every randomization and rollout surface shares.

A leaf on purpose: :mod:`strands_robots.simulation.base` re-exports these two
names for its backends, and :mod:`strands_robots.policies._rng` reaches them
directly, so the policies and simulation packages no longer import each other
to agree on what a seed is. Its only import from strands_robots is
:mod:`strands_robots.utils`, itself a leaf, for the refusal renderer every
guard in the package shares.
"""

from __future__ import annotations

import numbers
from typing import Any

from strands_robots.utils import refusal_repr

MAX_EVAL_SEED = 2**32 - 1


def randomization_seed_error(
    value: Any, context: str, *, max_seed: int | None = None, allow_none: bool = True
) -> str | None:
    """Return why a value cannot seed a reproducible randomization stream.

    The seed reaches ``numpy.random.default_rng``, which accepts only
    non-negative integers (and a few RNG objects the ``int | None`` annotations
    on these methods do not advertise). A float or string seed raises there -
    on the sensor-noise path not until the first observation is drawn, long
    after the configuring call reported success - so it is rejected at the call
    that supplied it.

    Two families share this domain, and they share it because the failure is
    the same: the ``seed`` of ``randomize`` / ``set_obs_noise``, which drives
    the domain-randomization streams, and the ``seed`` of a policy rollout or
    evaluation (``run_policy`` / ``eval_policy`` / ``start_policy`` /
    ``evaluate_benchmark``), which pins the client RNGs a stochastic policy
    samples from. The name reads for the first family and is accurate for both:
    a rollout seed exists precisely to make the policy's randomization
    reproducible.

    Their appliers are not equally wide, so the accepted domain is not either.
    ``randomize`` / ``set_obs_noise`` reach ``default_rng``, which takes a
    non-negative integer of any width. A rollout seed is applied through
    :func:`~strands_robots.simulation.policy_runner.set_eval_seed`, which also
    reseeds the legacy NumPy global RNG (``numpy.random.seed``) - the one most
    policies draw from - and that refuses anything above :data:`MAX_EVAL_SEED`.
    ``max_seed`` carries that ceiling, so the rollout surfaces refuse a value
    they could not apply while the randomization surfaces keep the width they
    can honor. One rule with an explicit bound per destination is what stops
    the accepted domain drifting from the applier in either direction.

    ``allow_none`` is the same idea at the other end of the domain. ``None`` is
    a legitimate *parameter* value for most callers - it selects fresh entropy
    for ``randomize`` / ``set_obs_noise`` and means "do not seed" at the rollout
    facades - but it is not a *seed*, so an applier that has to hand one to an
    RNG cannot honor it: ``random`` and NumPy would reseed from entropy while
    ``torch.manual_seed`` refuses it, leaving a process-wide RNG side effect on
    a rollout that asked for none. ``allow_none=False`` refuses it there and
    drops ``None`` from the messages, so the reason a caller is given always
    describes the domain that caller actually has.

    Args:
        value: The candidate seed (``None`` selects fresh entropy).
        context: Method name to prefix the message with.
        max_seed: Largest value the caller's applier can honor, or ``None``
            when the non-negative-integer rule is the only bound.
        allow_none: Whether ``None`` is a value this caller can honor. True for
            a parameter where it selects fresh entropy or means "do not seed";
            False for an applier that has to hand a seed to an RNG, which has
            nothing to apply. When False the messages stop advertising ``None``
            too, so a caller is never offered a value this destination refuses.

    Returns:
        ``None`` when the seed is usable, otherwise the reason as a string.
    """
    none_clause = " or None" if allow_none else ""
    entropy_hint = " (None draws fresh entropy)" if allow_none else ""
    if value is None:
        if allow_none:
            return None
        return (
            f"{context}: seed is required; None is the absence of a seed, not a seed to apply. "
            f"To leave the RNGs untouched, do not call {context} - reseeding them from entropy "
            "is a global side effect an unseeded rollout must not acquire."
        )
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return f"{context}: seed must be a non-negative integer{none_clause}, got {refusal_repr(value)}{entropy_hint}"
    if int(value) < 0:
        return f"{context}: seed must be a non-negative integer{none_clause}, got {refusal_repr(value)}{entropy_hint}"
    if max_seed is not None and int(value) > max_seed:
        return (
            f"{context}: seed must be an integer in [0, {max_seed}]{none_clause}, got {refusal_repr(value)} "
            "(a rollout seed is applied to the legacy NumPy global RNG, which refuses a larger value)"
        )
    return None
