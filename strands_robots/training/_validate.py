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
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from strands_robots.tools._path_validation import validate_save_path

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
_FLAG_BOUND_FIELDS = ("dataset_root", "output_dir", "base_model", "embodiment")

# Path-like fields additionally get the audited filesystem check (null bytes,
# ``..`` traversal, protected system directories).
_PATH_FIELDS = ("dataset_root", "output_dir")

# Path-typed ``extra`` keys whose VALUE is a filesystem WRITE target reaching
# ``os.makedirs`` in a backend (cosmos ``export_dir`` -> export() output dir,
# ``dcp_path`` -> prepare() DCP base dir). The key-FORMAT allowlist above passes
# ``"export_dir"`` as a key, but does nothing for its value, so without this an
# agent-supplied ``extra["export_dir"]="/etc/cron.d/x"`` would reach makedirs with
# the protected-dir / ``..``-traversal guard turned off. Read-only path extras
# (``cosmos_root``/``groot_root``/``sft_toml``/``modality_config_path``) are
# deliberately excluded: the protected-dir block is a WRITE guard and would
# reject legitimate reads from system locations.
_PATH_VALUE_EXTRA_KEYS = ("export_dir", "dcp_path")


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

    # Path-typed ``extra`` VALUES that reach a backend write path get the same
    # audited filesystem guard as ``_PATH_FIELDS`` (the key-format check above
    # only guards the key, not the path value).
    extra = spec.extra or {}
    for key in _PATH_VALUE_EXTRA_KEYS:
        val = extra.get(key)
        if val:
            try:
                validate_save_path(str(val), label=f"extra[{key!r}]")
            except ValueError as e:
                problems.append(str(e))

    return problems
