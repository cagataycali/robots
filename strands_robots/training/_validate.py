"""Shared input validation for the training backends.

Two kinds of gate live here.

:func:`validate_train_inputs` is the input-safety gate. An agent populates
:class:`~strands_robots.training.base.TrainSpec` directly, so its path fields,
its flag-bound scalars and its free-form ``extra`` keys are untrusted values
that reach backend internals: they must be safe to interpolate into a config
field, a Hydra override or an argv token before any config is built. Every
concrete :meth:`~strands_robots.training.base.Trainer.validate` calls it, and
every ``train`` calls ``validate`` fail-closed.

The ``*_problems(spec, *, context)`` gates are read-only domain preflights.
Each returns one message per field of *spec* it cannot honor and an empty list
when the spec is usable; *context* is the caller's
:attr:`~strands_robots.training.base.Trainer.provider_name`, prefixed to every
message so a problem names the backend that refused the value. A gate reports
only on fields its caller reads - :class:`TrainSpec` documents that a backend
"reads the fields it supports and ignores the rest", so a backend that never
reads a field must not call the gate that bounds it. Fields are grouped per
gate by the axis they configure, not per backend.

The domains are the shared ones from :mod:`strands_robots.utils`:

* count - a positive ``int``; ``bool`` and integral ``float`` are refused
  (:func:`~strands_robots.utils.positive_count_error`).
* cadence - a non-negative whole number of steps
  (:func:`~strands_robots.utils.step_cadence_error`).
* rate - a positive finite real
  (:func:`~strands_robots.utils.positive_finite_number_error`).
* weight - any finite real (:func:`~strands_robots.utils.finite_number_error`).
* clip bound - a positive real, where ``inf`` spells "do not clip"
  (:func:`_clip_bound_error`).
* closed unit - a real in ``[0, 1]`` (:func:`_closed_unit_interval_error`);
  half-open unit - a real in ``(0, 1]``
  (:func:`_half_open_unit_interval_error`).
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from strands_robots.tools._path_validation import validate_save_path
from strands_robots.utils import (
    boolean_flag_error,
    finite_number_error,
    non_negative_count_error,
    positive_count_error,
    positive_finite_number_error,
    step_cadence_error,
    torch_device_error,
)

if TYPE_CHECKING:
    from strands_robots.training.base import TrainSpec

# ``extra`` keys are interpolated into argv as ``--{key}=...`` (lerobot/groot)
# or ``{key}=...`` (cosmos hydra). The key FORMAT is allowlisted, not the key
# set: lowercase and dotted, no leading dash, no ``=``, no whitespace.
_EXTRA_KEY_RE = re.compile(r"^[a-z][a-z0-9_.]*\Z")

# Scalars interpolated as the value of a single argv flag (e.g.
# ``--dataset.root={dataset_root}``). A leading ``-`` is the injection vector:
# ``base_model="--config_path=/etc/passwd"`` would parse as a separate flag.
# An interior ``=`` keeps the token single and is legitimate for HF refs.
_FLAG_BOUND_FIELDS = ("dataset_root", "output_dir", "base_model", "embodiment", "dataset_repo_id")

# Path-like fields additionally get the audited filesystem check (null bytes,
# ``..`` traversal, protected system directories).
_PATH_FIELDS = ("dataset_root", "output_dir")


def validate_train_inputs(spec: TrainSpec) -> list[str]:
    """Return input-safety problems for *spec*; empty when every value is safe.

    Checks the path fields for traversal and protected directories, refuses a
    flag-bound scalar that starts with ``-``, and refuses an ``extra`` key
    outside the allowlisted format.
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
    """Report ``steps`` and ``global_batch_size``, each a count."""
    problems: list[str] = []
    for param, value in (("steps", spec.steps), ("global_batch_size", spec.global_batch_size)):
        error = positive_count_error(value, param, context)
        if error is not None:
            problems.append(error)
    return problems


def rl_run_size_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``total_timesteps`` and ``rollout_steps``, each a count.

    ``num_envs`` is not here: which *counts* are usable differs between the
    backends that read it, but being a count *at all* is not per-backend, so
    the shared factor is graded by the caller that reads it.
    """
    problems: list[str] = []
    for param in ("total_timesteps", "rollout_steps"):
        error = positive_count_error(getattr(spec, param, 1), param, context)
        if error is not None:
            problems.append(error)
    return problems


def torch_device_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``device`` when stated: one torch can place tensors on.

    Empty when the field is unstated or torch is not importable.
    """
    device = getattr(spec, "device", None)
    if not device:
        return []
    problem = torch_device_error(device, "device", context)
    return [] if problem is None else [problem]


def rl_replay_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``buffer_size``, ``batch_size`` and ``gradient_steps``, each a count.

    A zero ``gradient_steps`` takes zero gradient updates for the whole run and
    still reports success with a written checkpoint.
    """
    problems: list[str] = []
    for param in ("buffer_size", "batch_size", "gradient_steps"):
        error = positive_count_error(getattr(spec, param, 1), param, context)
        if error is not None:
            problems.append(error)
    return problems


def launch_topology_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``num_gpus`` and ``num_nodes``, each a count.

    Both become a ``torchrun`` ``nproc_per_node`` / ``nnodes``.
    """
    problems: list[str] = []
    for param, value in (("num_gpus", spec.num_gpus), ("num_nodes", spec.num_nodes)):
        error = positive_count_error(value, param, context)
        if error is not None:
            problems.append(error)
    return problems


def learning_rate_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``learning_rate`` when supplied: a rate. Empty when ``None``."""
    if spec.learning_rate is None:
        return []
    error = positive_finite_number_error(spec.learning_rate, "learning_rate", context)
    return [error] if error is not None else []


def seed_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``seed`` when supplied: a non-negative count. Empty when ``None``."""
    if spec.seed is None:
        return []
    error = non_negative_count_error(spec.seed, "seed", context)
    return [] if error is None else [error]


def checkpoint_cadence_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``save_freq``: a cadence in steps."""
    error = step_cadence_error(spec.save_freq, "save_freq", context)
    return [] if error is None else [error]


def validation_episodes_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``val_episodes`` when supplied: a count. Empty when ``None``.

    The count becomes a real-valued split fraction whose ceiling lerobot takes,
    so a fractional value reserves a different number of episodes than asked.
    """
    if spec.val_episodes is None:
        return []
    error = positive_count_error(spec.val_episodes, "val_episodes", context)
    return [] if error is None else [error]


def _posture_flag_problems(spec: TrainSpec, fields: Sequence[str], *, context: str) -> list[str]:
    """One problem per field in *fields* that is not a ``bool``, in that order."""
    problems: list[str] = []
    for param in fields:
        error = boolean_flag_error(getattr(spec, param), param, context)
        if error is not None:
            problems.append(error)
    return problems


def resume_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``resume``: a ``bool``."""
    return _posture_flag_problems(spec, ("resume",), context=context)


def streaming_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``streaming``: a ``bool``."""
    return _posture_flag_problems(spec, ("streaming",), context=context)


def lora_hyperparameter_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``lora_r`` and ``lora_alpha``, each a count when supplied.

    Empty unless ``method == "lora"`` - the only strategy that reads them.
    """
    if spec.method != "lora":
        return []
    problems: list[str] = []
    for param, value in (("lora_r", spec.lora_r), ("lora_alpha", spec.lora_alpha)):
        if value is None:
            continue
        error = positive_count_error(value, param, context)
        if error is not None:
            problems.append(error)
    return problems


def _closed_unit_interval_error(value: Any, param: str, context: str) -> str | None:
    """Error text when *value* is not a real number in ``[0, 1]``; else ``None``."""
    error = finite_number_error(value, param, context)
    if error is not None:
        return error
    if not 0.0 <= float(value) <= 1.0:
        return f"{context}: {param} must be in [0, 1], got {value!r}."
    return None


def discount_factor_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``gamma``: a closed unit. Every RL backend reads it."""
    error = _closed_unit_interval_error(getattr(spec, "gamma", 0.0), "gamma", context)
    return [error] if error is not None else []


def gae_lambda_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``lam``: a closed unit.

    The trace decays by the product ``gamma * lam``, so bounding one factor
    does not bound the trace; only the on-policy backend estimates a trace.
    """
    error = _closed_unit_interval_error(getattr(spec, "lam", 0.0), "lam", context)
    return [error] if error is not None else []


def optimization_epochs_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``num_learning_epochs``: a count.

    It is the ``range()`` bound enclosing every ``optimizer.step()``, so a
    non-positive value takes no gradient step while the run still reports
    success and writes an untrained checkpoint.
    """
    error = positive_count_error(getattr(spec, "num_learning_epochs", 1), "num_learning_epochs", context)
    return [error] if error is not None else []


def temperature_learning_rate_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``alpha_lr``: a rate. Empty unless ``autotune_alpha`` is set."""
    if not getattr(spec, "autotune_alpha", False):
        return []
    error = positive_finite_number_error(getattr(spec, "alpha_lr", 3e-4), "alpha_lr", context)
    return [error] if error is not None else []


def initial_temperature_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``init_alpha``: a rate, since the temperature is stored as its log."""
    error = positive_finite_number_error(getattr(spec, "init_alpha", 1.0), "init_alpha", context)
    return [error] if error is not None else []


def target_entropy_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``target_entropy`` when stated: a finite real.

    ``None`` is the sentinel for the heuristic default, not a missing value.
    """
    value = getattr(spec, "target_entropy", None)
    if value is None:
        return []
    error = finite_number_error(value, "target_entropy", context)
    return [error] if error is not None else []


def _clip_bound_error(value: Any, param: str, context: str) -> str | None:
    """Error text when *value* is not a clip bound; ``None`` when it is.

    A clip bound is a positive real, and ``inf`` is the spelling of "do not
    clip" that both consumers honor.
    """
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            is_no_clip = float(value) == math.inf
        except Exception:
            # A ``numbers.Real`` registration owes this no working
            # ``__float__``, and a real past the float64 range raises
            # ``OverflowError`` (``10**400``, ``Fraction(10**400, 3)``).
            # Neither is the no-clip spelling; the domain below answers both.
            is_no_clip = False
        if is_no_clip:
            return None
    return positive_finite_number_error(value, param, context)


def gradient_clip_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``max_grad_norm``: a clip bound."""
    error = _clip_bound_error(getattr(spec, "max_grad_norm", 1.0), "max_grad_norm", context)
    return [error] if error is not None else []


def loss_weight_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``value_loss_coef`` and ``entropy_coef``, each a weight.

    Zero and negative are real configurations for both; the domain bounds only
    what no reading makes usable.
    """
    defaults = {"value_loss_coef": 1.0, "entropy_coef": 0.0}
    problems = []
    for param, default in defaults.items():
        error = finite_number_error(getattr(spec, param, default), param, context)
        if error is not None:
            problems.append(error)
    return problems


def clip_range_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``clip_param``: a clip bound, the half-width of the trust region."""
    error = _clip_bound_error(getattr(spec, "clip_param", 0.2), "clip_param", context)
    return [error] if error is not None else []


def policy_delay_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``policy_delay``: a count.

    It is the modulus of an ``update_count % policy_delay`` test, so a value
    that never satisfies it trains the critics while the deployable actor never
    takes a gradient step.
    """
    error = positive_count_error(getattr(spec, "policy_delay", 1), "policy_delay", context)
    return [error] if error is not None else []


def td3_noise_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``exploration_noise_std``, ``target_noise_std`` and ``target_noise_clip``.

    Each is a rate: Gaussian noise is symmetric, so a negative scale is the
    identical distribution and zero removes the mechanism entirely.
    """
    problems: list[str] = []
    for param, default in (
        ("exploration_noise_std", 0.1),
        ("target_noise_std", 0.2),
        ("target_noise_clip", 0.5),
    ):
        error = positive_finite_number_error(getattr(spec, param, default), param, context)
        if error is not None:
            problems.append(error)
    return problems


def network_width_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``hidden_dims``: a sequence of counts, each width named by index."""
    widths = getattr(spec, "hidden_dims", ())
    if isinstance(widths, str) or not isinstance(widths, Sequence):
        return [f"{context}: hidden_dims must be a sequence of positive int layer widths, got {widths!r}"]
    return [
        error
        for index, width in enumerate(widths)
        if (error := positive_count_error(width, f"hidden_dims[{index}]", context)) is not None
    ]


def rl_checkpoint_interval_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``log_interval``: a cadence in iterations."""
    error = step_cadence_error(getattr(spec, "log_interval", 0), "log_interval", context)
    return [] if error is None else [error]


def _half_open_unit_interval_error(value: Any, param: str, context: str) -> str | None:
    """Error text when *value* is not a real number in ``(0, 1]``; else ``None``."""
    error = finite_number_error(value, param, context)
    if error is not None:
        return error
    if not 0.0 < float(value) <= 1.0:
        return f"{context}: {param} must be in (0, 1], got {value!r}."
    return None


def polyak_coefficient_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Report ``tau``: a half-open unit - a zero target update never moves."""
    error = _half_open_unit_interval_error(getattr(spec, "tau", 0.005), "tau", context)
    return [error] if error is not None else []
