"""Shared, defense-in-depth input validation for the training backends.

Every concrete :class:`~strands_robots.training.base.Trainer` translates a
:class:`~strands_robots.training.base.TrainSpec` into its backend's native
config object and runs it IN-PROCESS (imported and called as a library - no
subprocess). The ``train_policy`` ``@tool`` lets an agent (LLM) populate that
``TrainSpec`` directly, so the path fields and the free-form ``extra`` dict are
*untrusted input that reaches backend internals*. Per ``AGENTS.md`` > Review
Learnings (#92) > "LLM Input Safety", those values MUST be validated before they
can become a config field, a Hydra override, or a token in a backend's
argv-parity helper: a value beginning with ``-`` could read as a *new flag*, and
an arbitrary ``extra`` key could set an arbitrary config attribute / override.

:func:`validate_train_inputs` is the single source of that check. It is invoked
from every backend's :meth:`Trainer.validate`, which each backend's
:meth:`Trainer.train` calls (fail-closed) before building any config - so no
run can start with unvalidated input regardless of the call path.

:func:`run_size_problems` is the second shared gate, on a different axis: the
*run size* numerics. ``steps`` and ``global_batch_size`` are the two factors of
how much training a spec asks for, and both are read straight into a backend's
loop bound / dataloader. They live in their own gate rather than in
:func:`validate_train_inputs` because :class:`TrainSpec` documents that a
backend "reads the fields it supports and ignores the rest": the RL trainers
drive training from ``total_timesteps`` / ``batch_size`` and never read either
field, so reporting a problem for them there would be a false rejection of a
field that backend does not use.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from strands_robots.tools._path_validation import validate_save_path
from strands_robots.utils import positive_count_error

if TYPE_CHECKING:
    from strands_robots.training.base import TrainSpec

# ``extra`` keys are interpolated into argv as ``--{key}=...`` (lerobot/groot)
# or ``{key}=...`` (cosmos hydra). Allowlist the key FORMAT only: lowercase,
# dotted (lerobot ``dataset.episodes`` / cosmos ``model.x.y``), no leading dash,
# no ``=``, no whitespace or shell metacharacters. We deliberately do NOT try to
# enumerate every valid backend flag - that allowlist is impossible to keep
# current and would break the documented ``extra`` escape hatch.
_EXTRA_KEY_RE = re.compile(r"^[a-z][a-z0-9_.]*$")

# Scalars that are interpolated as the value of a single argv flag
# (e.g. ``--dataset.root={dataset_root}``). A leading ``-`` is the injection
# vector: ``base_model="--config_path=/etc/passwd"`` would otherwise parse as a
# separate flag. An interior ``=`` is harmless (the token stays single, no
# shell) and is legitimate for HF revision refs, so it is NOT rejected.
_FLAG_BOUND_FIELDS = ("dataset_root", "output_dir", "base_model", "embodiment", "dataset_repo_id")

# Path-like fields additionally get the audited filesystem check (null bytes,
# ``..`` traversal, protected system directories).
_PATH_FIELDS = ("dataset_root", "output_dir")


def validate_train_inputs(spec: TrainSpec) -> list[str]:
    """Return a list of input-safety problems for a :class:`TrainSpec`.

    An empty list means every agent-supplied value is safe to interpolate into
    a backend config / argv-parity helper. Pure and side-effect-free
    (read-only ``realpath`` only),
    so it is safe to call from :meth:`Trainer.validate`.
    """
    problems: list[str] = []

    # Path fields: reuse the audited validator used by the other write-path tools.
    for label in _PATH_FIELDS:
        val = getattr(spec, label, None)
        if val:
            try:
                validate_save_path(str(val), label=label)
            except ValueError as e:
                problems.append(str(e))

    # Flag-bound scalars must not smuggle an argv flag via a leading dash.
    for label in _FLAG_BOUND_FIELDS:
        val = getattr(spec, label, None)
        if isinstance(val, str) and val.startswith("-"):
            problems.append(f"{label} must not start with '-' (would parse as a stray flag)")

    # ``extra`` keys become backend-native flags - allowlist the key format.
    for key in spec.extra or {}:
        if not _EXTRA_KEY_RE.match(str(key)):
            problems.append(
                f"extra key {key!r} is not allowed "
                f"(must match {_EXTRA_KEY_RE.pattern}: lowercase, "
                f"no leading dash, no '=', no whitespace)"
            )

    return problems


def run_size_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return run-size problems for a :class:`TrainSpec`.

    ``steps`` and ``global_batch_size`` are the two factors of the amount of
    training a spec asks for, and each is consumed directly as a discrete
    count: ``steps`` bounds the backend's optimizer loop (lerobot iterates
    ``range(step, cfg.steps)``) and ``global_batch_size`` becomes a
    ``DataLoader`` batch size / a ``--global_batch_size`` flag. Only a positive
    integer can be honored, which is why both are checked against the one
    shared :func:`~strands_robots.utils.positive_count_error` domain rather
    than a local comparison: a bare ``value <= 0`` test admits every value that
    is not comparably non-positive, so ``True`` reads as a silent run of one
    step, a fractional or non-finite value reaches ``range()`` and raises
    there, and a string raises out of the comparison itself - inside a
    :meth:`Trainer.validate` that is documented to *return* problems.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        One problem per unusable field; empty when both are usable counts.
    """
    problems: list[str] = []
    for param, value in (("steps", spec.steps), ("global_batch_size", spec.global_batch_size)):
        error = positive_count_error(value, param, context)
        if error is not None:
            problems.append(error)
    return problems
