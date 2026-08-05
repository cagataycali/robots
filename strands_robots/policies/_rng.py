"""Shared RNG reseed helper for Policy providers.

#331: ``Gr00tPolicy.reset`` reseeds Python ``random``, NumPy, torch CPU + CUDA,
and toggles cuDNN determinism, while ``Cosmos3Policy.reset`` only mutated the
global NumPy RNG. Two providers conforming to the same ``Policy`` contract must
behave identically for ``set_eval_seed``-style reproducibility (#187). This
module is the single source of truth for the client-side reseed so both
providers stay in parity.

Parity covers which seeds are *accepted* as well as which RNGs are reseeded.
The appliers here do not share one domain - Python ``random`` takes a string,
NumPy's legacy global RNG takes a non-negative integer below 2**32 - so the
narrowest of them decides what this helper can honor, and that is the same
bound :func:`~strands_robots.simulation.policy_runner.set_eval_seed` applies
for the same reason.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def reseed_client_rngs(seed: int | None) -> None:
    """Reseed the client-side RNGs for per-episode reproducibility.

    Seeds Python ``random``, NumPy, and (if importable) torch CPU + CUDA, and
    toggles cuDNN into deterministic mode. ``None`` is a no-op. Best-effort:
    a missing torch is skipped silently (it is an optional dependency for the
    policy providers); any other failure is logged and swallowed because
    ``reset`` is a soft reproducibility hint, not a hard requirement.

    A seed outside the domain is refused up front, which is a different case
    from the best-effort clause above rather than an exception to it. That
    clause covers a *failure of an applier* - torch absent, a runtime hiccup -
    and swallowing one leaves every RNG consistently unseeded. An unusable
    value is a caller mistake no applier can accept, and swallowing it left
    the process *half* seeded: the appliers run in sequence and only the
    second bounds the domain, so Python ``random`` was reseeded and NumPy was
    not for every seed NumPy refuses - anything negative, fractional,
    non-integral, or above
    :data:`~strands_robots.simulation.base.MAX_EVAL_SEED`. ``reset`` still
    returned normally, so a caller was told the episode was seeded while half
    the streams a policy draws from were not, with the reason only at
    ``INFO``. Checking before the first applier runs makes the reseed
    all-or-nothing for every value it accepts.

    The domain is the one its sibling applier
    :func:`~strands_robots.simulation.policy_runner.set_eval_seed` already
    enforces, down to the ``MAX_EVAL_SEED`` ceiling that
    :func:`~strands_robots.simulation.base.randomization_seed_error` carries
    for exactly this destination - both reseed the legacy NumPy global RNG,
    so both can honor exactly the same seeds. A rollout that reseeds through
    a provider's ``reset`` therefore accepts what one seeding through
    ``set_eval_seed`` accepts.

    Args:
        seed: Master per-episode seed, or ``None`` to leave RNGs untouched.

    Raises:
        ValueError: If *seed* is neither ``None`` nor an integer in
            ``[0, MAX_EVAL_SEED]``. Raising rather than logging is what the
            caller needs to distinguish "this episode is reproducible" from
            "it is not": the alternative is a rollout that reports success
            having seeded some of its RNGs, which is the state this guard
            exists to make unreachable.
    """
    if seed is None:
        return

    # Deferred import: strands_robots.simulation.base imports policy_runner at
    # module level and policy_runner imports policies.base, so reaching the
    # shared seed domain from module scope here would close that ring. Both
    # callers of this helper import it the same way, for the same reason.
    from strands_robots.simulation.base import MAX_EVAL_SEED, randomization_seed_error

    # ``None`` returned above, so a seed is required past this line;
    # ``allow_none=False`` keeps the reason from offering ``None`` as a remedy
    # for a value that already got here.
    if error := randomization_seed_error(seed, "reseed_client_rngs", max_seed=MAX_EVAL_SEED, allow_none=False):
        raise ValueError(error)

    try:
        import random as _random

        _random.seed(seed)

        import numpy as _np

        _np.random.seed(seed)

        try:
            import torch as _torch

            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
        except ImportError:
            # torch is optional for the policy providers (mock / service-only
            # installs); no torch RNG state to seed when it is not present.
            pass
    except Exception as exc:  # noqa: BLE001 - reset is best-effort
        logger.info("reseed_client_rngs: reseed failed (seed=%r): %s", seed, exc)
