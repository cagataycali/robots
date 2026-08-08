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

:func:`learning_rate_problems` is the third, on the optimization axis. It lives
in its own gate for the opposite reason to :func:`run_size_problems`: *every*
backend reads ``learning_rate`` -- the three supervised ones map it onto their
config's optimizer field, and the RL trainers hand it straight to
``torch.optim.Adam`` -- so there is no backend for which reporting on it would
be a false rejection, and :class:`~strands_robots.training.rl.base_algo.RLTrainSpec`
documents the field as one of the "universal" ones. It is separate from
:func:`validate_train_inputs` because that gate answers a different question
(is this value safe to interpolate into a config or an argv token) from this one
(can this value be honored at all).

:func:`launch_topology_problems` is the fourth, on the *launch topology* axis:
``num_gpus`` and ``num_nodes``, the two process counts every distributed launch
is sized from. It is scoped like :func:`run_size_problems` rather than like
:func:`learning_rate_problems` - only the three supervised backends read either
field (they become a ``torchrun``/``elastic_launch`` ``nproc_per_node`` /
``nnodes``), so a backend that ignores them must not report on them.

:func:`seed_problems` is the fifth, on the reproducibility axis, and
:func:`validation_episodes_problems` the sixth, on the *evaluation* axis:
``val_episodes``, the episode count a caller reserves as a held-out validation
set. It is scoped like :func:`run_size_problems` - only the LeRobot backend
reads the field (GR00T, Cosmos and the RL trainers never do), so a backend that
ignores it must not report on it. What makes a shared gate the right home
rather than a local test is the conversion: the count becomes a real-valued
split fraction whose ceiling lerobot takes, so a comparison admits values that
reserve a different number of episodes than the one asked for.

:func:`lora_hyperparameter_problems` is the seventh, on the *adapter* axis:
``lora_r`` and ``lora_alpha``, the rank and the scaling numerator of a LoRA
fine-tune. It is scoped like :func:`run_size_problems` and narrowed once more -
only the LeRobot backend reads either field, and only on its ``method == "lora"``
branch, so a value a run's own strategy never reads must not be reported.

:func:`discount_factor_problems` is the eighth, on the *return* axis:
``gamma``, the discount factor of the return the algorithm optimizes. It is
scoped like :func:`learning_rate_problems` rather than like
:func:`run_size_problems` - it is the one
:class:`~strands_robots.training.rl.base_algo.RLTrainSpec` coefficient that
*every* RL backend reads (PPO discounts the GAE recursion with it, FastSAC
discounts its target-Q bootstrap), so there is no RL backend for which
reporting on it would be a false rejection.

:func:`gae_lambda_problems` is the ninth, on the same *return* axis and for the
sibling factor: ``lam``, the GAE trace-decay coefficient. It is a separate gate
from :func:`discount_factor_problems` because the two fields are scoped
differently - every RL backend reads ``gamma``, but only the on-policy backend
estimates an advantage trace, so per :class:`TrainSpec` FastSAC must not report
on a field it never reads. They are nonetheless one contract: the trace decays
by the *product* ``gamma * lam``, so bounding one factor does not bound the
trace.

:func:`optimization_epochs_problems` is the tenth, on the *optimization* axis:
``num_learning_epochs``, the number of passes the on-policy update makes over
each rollout batch. It is the loop bound of the entire optimizer step
(``for _ in range(spec.num_learning_epochs)`` wraps every ``optimizer.step()``),
so a non-positive value takes no gradient step at all while the run still
collects its rollouts, writes a deployable checkpoint and reports success. It is
scoped like :func:`gae_lambda_problems`: only the on-policy backend has an epoch
loop, so FastSAC must not report on a field it never reads.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from strands_robots.tools._path_validation import validate_save_path
from strands_robots.utils import (
    finite_number_error,
    non_negative_count_error,
    positive_count_error,
    positive_finite_number_error,
)

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


def launch_topology_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return launch-topology problems for a :class:`TrainSpec`.

    ``num_gpus`` and ``num_nodes`` are the two process counts a distributed run
    is sized from. Each is consumed as a discrete count in three places: a
    ``spec.num_gpus > 1`` / ``spec.num_nodes > 1`` test that selects between the
    single-process and the multi-process launch path, a ``nproc_per_node`` /
    ``nnodes`` argument to torch's ``elastic_launch``, and a
    ``--nproc_per_node=`` / ``--nnodes=`` / ``--num_gpus=`` argv token. Only a
    positive integer can be honored, and each of the three ways a bad value
    fails is silent or late:

    * ``0``, a negative, ``nan`` and ``True`` all read as *not* greater than one
      -- ``nan`` compares false against everything -- so the selector routes
      them to the single-process path and the run proceeds on one process under
      a successful result. The topology the caller asked for is simply not the
      one that ran, and for ``num_nodes`` that also slips past the multi-node
      refusal the backends raise for an unsupported topology.
    * ``2.7`` and ``inf`` *are* greater than one, so they select the
      multi-process path and reach ``elastic_launch`` as the worker count.
      ``LaunchConfig`` accepts both without complaint, so nothing downstream
      rejects them either.
    * A string, ``None`` or a list raises ``TypeError`` out of the comparison
      itself -- from inside a :meth:`Trainer.validate` that is documented to
      *return* problems.

    Both are therefore checked against the one shared
    :func:`~strands_robots.utils.positive_count_error` domain, the same one
    :func:`run_size_problems` uses, rather than by a local comparison.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        One problem per unusable field; empty when both are usable counts.
    """
    problems: list[str] = []
    for param, value in (("num_gpus", spec.num_gpus), ("num_nodes", spec.num_nodes)):
        error = positive_count_error(value, param, context)
        if error is not None:
            problems.append(error)
    return problems


def learning_rate_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return optimizer learning-rate problems for a :class:`TrainSpec`.

    ``learning_rate`` is the one numeric on a :class:`TrainSpec` that decides
    whether a run *learns* rather than how much work it does, and every backend
    reads it: the supervised three assign it to their config's optimizer field
    (LeRobot ``policy.optimizer_lr``, GR00T ``FinetuneConfig.learning_rate``,
    Cosmos ``optimizer.lr``) and the RL trainers pass it directly to
    ``torch.optim.Adam(..., lr=...)``.

    Only a positive finite value can be honored, and the two ends of the domain
    fail *silently* rather than loudly, which is why this is a preflight rather
    than something the backend can be left to notice:

    * ``0`` (and ``False``, which is ``0`` to every consumer) runs the full
      ``steps`` x ``global_batch_size`` of work and updates no weight, so the
      run reports success and writes a checkpoint identical to its
      initialisation. That is the pathology :func:`run_size_problems` exists to
      prevent, reached by a different route and at full cost.
    * ``inf`` diverges on the first optimizer step, so the checkpoint is all
      ``NaN`` -- again under a successful result.
    * ``True`` is a silent learning rate of ``1.0``, four orders of magnitude
      above a typical fine-tuning preset.

    A negative or ``nan`` value *is* refused by ``torch.optim.Adam``
    (``ValueError: Invalid learning rate``), but only once the dataset and model
    are already loaded -- after the point :meth:`Trainer.validate` documents
    itself as running before ("it powers a ``plan`` advisor that runs *before*
    anything expensive starts").

    ``None`` is the documented sentinel for "use the backend's own default" and
    is therefore not a problem. It is checked against the shared
    :func:`~strands_robots.utils.positive_finite_number_error` domain rather
    than a local comparison because a bare ``value <= 0`` test admits ``nan``
    (every comparison against it is ``False``), admits a ``bool``, and raises
    out of the comparison itself for a non-numeric value -- inside a method
    documented to *return* problems.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single problem when ``learning_rate`` is supplied and unusable;
        empty when it is usable or left at ``None``.
    """
    if spec.learning_rate is None:
        return []
    error = positive_finite_number_error(spec.learning_rate, "learning_rate", context)
    return [error] if error is not None else []


def seed_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return reproducibility-seed problems for a :class:`TrainSpec`.

    ``seed`` is the field a caller sets to make a run reproducible, and the four
    backends that read it apply it through appliers that disagree about what a
    single value means:

    * The RL trainers hand it to ``torch.manual_seed``, which reduces it modulo
      ``2**64``. A negative seed is therefore *silently a different seed*:
      ``manual_seed(-1)`` and ``manual_seed(2**64 - 1)`` draw the identical
      stream, so two seeds the caller means to be distinct collapse onto one and
      the run is reproducible under a number nobody asked for. ``True`` is
      likewise a silent seed of ``1`` and ``2.7`` a silent seed of ``2``.
    * LeRobot assigns it to ``cfg.seed``, which reaches lerobot's ``set_seed``:
      ``random.seed`` first, then ``numpy.random.seed``. NumPy is far narrower
      than torch - it refuses a negative value and a float or string outright -
      but only *after* ``random.seed`` has run, so a refused seed leaves the
      process RNG reseeded by a call that failed.
    * Cosmos interpolates it into a ``trainer.seed=`` Hydra override, and
      LeRobot's argv-parity path into a ``--seed=`` token. There every value
      renders - ``nan``, ``2.7``, ``[7]`` - and fails, if at all, inside the
      run after the dataset and model are already loaded.

    So the same ``seed=-1`` is silently rewritten by one backend and refused with
    a bare third-party message by the next. Only a non-negative integer can be
    honored by all of them, so it is checked against the one shared
    :func:`~strands_robots.utils.non_negative_count_error` domain: the same
    non-negative-integer rule, whose ``0`` is first-class here too (seed ``0`` is
    a seed), and which rejects ``bool`` explicitly because a bare ``value < 0``
    test lets ``True`` through as a silent seed of one.

    ``None`` is the documented sentinel for "use the backend's own default"
    (LeRobot's is ``1000``) and is therefore not a problem, exactly as it is not
    one for :func:`learning_rate_problems`.

    One boundary this does not decide: the appliers also disagree about the
    upper end - torch accepts up to ``2**64 - 1`` while NumPy's legacy seeder
    stops at ``2**32 - 1`` - so a per-backend ceiling is a separate question from
    the floor and type checked here.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single problem when ``seed`` is supplied and unusable; empty otherwise.
    """
    if spec.seed is None:
        return []
    error = non_negative_count_error(spec.seed, "seed", context)
    return [] if error is None else [error]


def validation_episodes_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return held-out-validation-set problems for a :class:`TrainSpec`.

    ``val_episodes`` is the count of episodes a caller reserves from the tail of
    the dataset to validate on. It is not read straight into a loop bound like
    :func:`run_size_problems`' fields: the LeRobot backend converts it into
    lerobot's ``dataset.eval_split`` FRACTION via
    :func:`~strands_robots.utils.validation_split_fraction`, and lerobot then
    holds out ``ceil(episodes_in_task * eval_split)``. That conversion is what
    makes a local comparison unsafe in both directions at once:

    * A non-positive value is *silently dropped*. The fraction is only computed
      for a count in ``(0, total)``, so ``val_episodes=0`` (or a negative)
      produces no ``eval_split`` and no ``eval_steps`` at all: the run trains on
      the whole dataset, records no validation loss, and reports no problem. The
      caller asked for a validation set and got a run without one.
    * A value that merely *compares* as positive is silently rewritten, because
      the fraction is real-valued and lerobot takes its ceiling: ``True``
      reserves 1 episode and ``2.7`` reserves 3 - a whole number the caller never
      named. ``0.5`` is the sharpest of these: it clears the ``0 < count <
      total`` test, so it emits ``eval_split=0.0`` - a held-out set of zero
      episodes - *together with* an ``eval_steps`` cadence, asking lerobot to
      validate periodically on nothing.
    * A non-numeric value raises out of the comparison itself, from a
      :meth:`~strands_robots.training.base.Trainer.validate` documented to
      *return* problems.

    Only a positive integer strictly below the dataset's episode count can be
    honored, so the type and floor are checked here against the same shared
    :func:`~strands_robots.utils.positive_count_error` domain that
    :func:`run_size_problems` uses. The upper bound is dataset-dependent (it needs
    ``total_episodes`` from ``meta/info.json``) and stays with the backend that
    reads the metadata, which also owns the per-task-fraction refusal in
    :func:`~strands_robots.utils.validation_split_error`.

    ``None`` is the documented sentinel for "train on every episode, no held-out
    set" and is therefore not a problem, exactly as it is not one for
    :func:`seed_problems` or :func:`learning_rate_problems`.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single problem when ``val_episodes`` is supplied and unusable as a
        count; empty otherwise.
    """
    if spec.val_episodes is None:
        return []
    error = positive_count_error(spec.val_episodes, "val_episodes", context)
    return [] if error is None else [error]


def lora_hyperparameter_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return LoRA adapter-hyperparameter problems for a :class:`TrainSpec`.

    ``lora_r`` and ``lora_alpha`` are the rank and the scaling numerator of a
    LoRA fine-tune: peft builds a rank-``r`` adapter and applies its update
    scaled by ``lora_alpha / r``. The two fields fail in opposite ways, and only
    one of them fails loudly:

    * ``lora_r`` is refused by peft, but only from inside
      ``get_peft_model`` - after the base model has been downloaded and loaded.
      A non-positive rank raises ``ValueError: `r` should be a positive integer
      value``, and a ``bool``/float/string one raises out of torch's tensor
      allocation with a message naming neither the field nor the run.
    * ``lora_alpha`` is **accepted for every unusable value**. It is only ever a
      numerator, so nothing downstream compares it: ``lora_alpha=0`` builds the
      adapter, reports its trainable parameters and trains them with a scaling
      of ``0.0``, so the adapter provably cannot change the model's output - the
      fine-tune runs to completion, writes checkpoints, and has learned nothing
      that can ever be applied. A negative value applies the negation of what
      the adapter learned, and ``True`` is a silent alpha of one.

    The two paths that carry these fields also disagree about a fractional
    value. In-process, peft accepts ``lora_alpha=2.7`` and scales by
    ``2.7 / r``; on the argv-parity path the same value reaches lerobot's
    ``PeftConfig``, whose ``r`` and ``lora_alpha`` are declared ``int``, and
    draccus refuses it. So one spelling of one run honors a value the other
    rejects.

    A positive integer is therefore the only thing both paths can honor, and it
    is checked against the same shared
    :func:`~strands_robots.utils.positive_count_error` domain
    :func:`run_size_problems` uses - the domain that also rejects ``bool``,
    which a bare ``value < 1`` test would let through as a silent rank or alpha
    of one.

    ``None`` is the documented sentinel for "omit the option and keep peft's own
    default" and is therefore not a problem, exactly as it is not one for
    :func:`seed_problems` or :func:`validation_episodes_problems`.

    Both fields are read only on the ``method == "lora"`` branch, so a spec that
    carries them under another strategy reports nothing: the fields are inert
    there, and refusing a value the run never reads would be a false rejection -
    the same reason this is a separate gate from :func:`learning_rate_problems`
    rather than part of it.

    ``lora_target_modules`` is out of scope: it is a module-name string rather
    than a count, and :func:`validate_train_inputs` already owns what may be
    interpolated into a config field or an argv token.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        One problem per supplied-and-unusable adapter hyperparameter; empty when
        the spec does not request LoRA or both values are usable.
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
    """Error text when *value* is not a real number in the closed range [0, 1].

    Numeric-ness, ``bool`` rejection and finiteness are delegated to the shared
    :func:`~strands_robots.utils.finite_number_error` domain, so those refusals
    read identically to every other numeric field's. The only thing decided here
    is the interval, which no shared domain expresses: ``utils`` carries
    open-ended families (positive, non-negative) rather than a bounded one.

    Both endpoints are inside the domain and neither is a degenerate spelling of
    "disabled", which is why the interval is closed rather than half-open.

    Args:
        value: The caller-supplied value.
        param: Field name for the message.
        context: Caller label the message is prefixed with.

    Returns:
        The error text, or None when *value* is a real number in [0, 1].
    """
    error = finite_number_error(value, param, context)
    if error is not None:
        return error
    if not 0.0 <= float(value) <= 1.0:
        return f"{context}: {param} must be in [0, 1], got {value!r}."
    return None


def discount_factor_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return discount-factor problems for an RL :class:`TrainSpec`.

    ``gamma`` weights every future reward in the return the algorithm optimizes,
    and it is the one coefficient both RL backends read: PPO discounts the GAE
    recursion with it (twice - the single-env and vectorized rollout paths), and
    FastSAC discounts its target-Q bootstrap. A discounted return is a geometric
    series, so the domain is not a matter of taste:

    * ``gamma > 1`` makes that series **diverge**. The advantages grow without
      bound in the rollout horizon rather than being merely large - over a
      24-step rollout of unit rewards, ``gamma=1.5`` inflates the largest
      advantage from 12.9 to 1.2e4, and ``gamma=5`` to 4.6e15. Nothing refuses
      it: the run trains on those advantages, reports success, and writes a
      checkpoint.
    * ``gamma < 0`` alternates the sign of each successive reward, so the trace
      no longer accumulates future return at all - the same rollout collapses
      the largest advantage to the immediate reward, 1.0.
    * ``nan``/``inf`` make every advantage non-finite, which surfaces only once
      the update samples the action distribution: ``ValueError: Expected
      parameter loc ... of distribution Normal ... to satisfy the constraint
      Real()``, a torch message that names neither the field nor the run, raised
      after the env, the networks and a full rollout have been built. That is
      exactly the "deep stack trace" a read-only preflight exists to replace.
    * ``True`` is a silent ``gamma`` of one, because a bare comparison against
      the interval bounds accepts it - ``bool`` is an ``int`` subclass.

    Both endpoints are legitimate and standard: ``gamma=1`` is the undiscounted
    episodic return, ``gamma=0`` a myopic agent that optimizes the immediate
    reward only. So the domain is the *closed* interval [0, 1], checked through
    :func:`_closed_unit_interval_error`.

    The sibling FastSAC preflight already bounds its own interval coefficient
    this way (``tau`` must be in ``(0, 1]``), which is the shape this gate
    generalizes: an interval coefficient is checked against its interval rather
    than left to the arithmetic that consumes it.

    ``lam``, the other factor of the trace-decay product, has its own gate for
    that same scoping reason - see :func:`gae_lambda_problems`, and
    ``num_learning_epochs`` likewise in :func:`optimization_epochs_problems`.
    The remaining PPO coefficients (``clip_param``, ``entropy_coef``,
    ``value_loss_coef``, ``max_grad_norm``, ``init_noise_std``) are out of scope
    in all three: they weight loss terms and bound gradients rather than the
    return, and for each of them zero has a candidate reading as a *disabled
    mode* that this repository has not settled - ``entropy_coef`` is shipped
    defaulting to ``0.0``, so zero is already a supported configuration;
    ``max_grad_norm=0`` reads as "no clipping" in several RL codebases; and
    ``init_noise_std=0`` is refused by ``torch`` itself, which rejects a
    ``Normal`` of zero scale. Those are contract questions rather than defects,
    so they are left to a gate that settles them.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single-element list when ``gamma`` cannot be honored; empty otherwise.
    """
    error = _closed_unit_interval_error(getattr(spec, "gamma", 0.0), "gamma", context)
    return [error] if error is not None else []


def gae_lambda_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return GAE-lambda problems for an on-policy RL :class:`TrainSpec`.

    ``lam`` is the second factor of the advantage trace's decay. The GAE
    recursion carries it forward as ``last_adv = delta + gamma * lam *
    (1 - done) * last_adv``, so the trace decays by the **product**
    ``gamma * lam`` and :func:`discount_factor_problems` bounding ``gamma``
    alone does not bound it: with a ``gamma`` of ``0.99`` - comfortably inside
    that gate's closed interval - a ``lam`` of ``1.5`` gives a decay factor of
    ``1.485`` and the same divergence, measured on this backend's own
    ``compute_gae`` over a rollout of unit rewards:

    ======  =========  =========  =========  =========
    ``lam``  ``T=12``   ``T=24``   ``T=48``   ``T=96``
    ======  =========  =========  =========  =========
    0.95      8.8        13.0       15.9       16.8
    1.5     235.1      2.7e+04    3.6e+08    6.3e+16
    1e6       inf        inf        inf        inf
    ======  =========  =========  =========  =========

    The largest advantage grows without bound in the rollout horizon rather than
    being merely large, and nothing refuses it: the run trains on those
    advantages, reports success, and writes a checkpoint.

    The remaining values outside the interval fail in three further ways:

    * ``lam < -1 / gamma`` diverges as well, because the trace decays by
      ``|gamma * lam|`` - ``lam=-2`` reaches ``1.0e+28`` by ``T=96`` - while a
      ``lam`` merely below zero (``-0.5``) collapses the trace to the immediate
      reward, so the estimator stops accumulating future advantage at all.
    * ``nan``/``inf`` make every advantage non-finite, which surfaces only once
      the update samples the action distribution - a torch constraint error
      naming neither the field nor the run, after the env, the networks and a
      full rollout have been built.
    * ``True`` is a silent ``lam`` of one, because a bare comparison against the
      interval bounds accepts it: ``bool`` is an ``int`` subclass. That is a
      different estimator from the one the caller asked for - Monte-Carlo return
      rather than a bootstrapped trace.

    Both endpoints are legitimate and standard, which is why the domain is the
    *closed* interval [0, 1]: ``lam=1`` is the Monte-Carlo advantage (no
    bootstrapping, 61.9 at ``T=96`` above) and ``lam=0`` is TD(0), the
    one-step advantage.

    Unlike ``gamma`` this is read by the on-policy backend only, so it is scoped
    like :func:`run_size_problems`: FastSAC has no advantage trace and must not
    report on a field it never reads.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single-element list when ``lam`` cannot be honored; empty otherwise.
    """
    error = _closed_unit_interval_error(getattr(spec, "lam", 0.0), "lam", context)
    return [error] if error is not None else []


def optimization_epochs_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return optimization-epoch problems for an on-policy RL :class:`TrainSpec`.

    ``num_learning_epochs`` is the number of passes the update makes over each
    rollout batch, and it is consumed as a bare loop bound around the whole
    optimizer step - ``for _ in range(spec.num_learning_epochs)`` encloses every
    ``optimizer.step()`` in the PPO update. So the field does not merely scale
    how much optimization happens; a non-positive value removes *all* of it, and
    nothing downstream notices:

    ==========================  =========  ==============  ==============
    ``num_learning_epochs``     verdict    optimizer       reported
                                           steps taken     losses
    ==========================  =========  ==============  ==============
    5 (the shipped default)     honored    24              real values
    0                           accepted   **0**           all ``0.0``
    -3                          accepted   **0**           all ``0.0``
    ==========================  =========  ==============  ==============

    Measured on this backend over a 60-step run: ``0`` and ``-3`` both report
    ``status="success"``, take **zero** gradient steps, and write a checkpoint
    whose parameters are bit-identical to each other - the untrained
    initialisation. The losses read ``0.0`` rather than blank because the update
    averages its accumulators through ``max(1, n_updates)``, so an epoch count
    that ran no minibatch reports plausible metrics for a run that learned
    nothing. A caller therefore gets a deployable-looking checkpoint, a
    successful result and a metrics dict, with no signal anywhere that the
    optimizer never ran.

    The remaining values outside the domain fail in two further ways:

    * ``True`` is a silent single epoch, because ``range(True)`` is
      ``range(1)`` - the same run takes 12 optimizer steps instead of 24. That
      is a different amount of optimization from the one requested, reported as
      success.
    * ``2.7``/``nan``/``inf``/``"5"``/``None`` raise a bare ``TypeError:
      'float' object cannot be interpreted as an integer`` out of ``range()``,
      naming neither the field nor the run, and only after the environment, the
      networks and a full rollout have been built - exactly the deep stack trace
      a read-only preflight exists to replace. No checkpoint is written at all.

    The domain is therefore a positive integer, checked by
    :func:`~strands_robots.utils.positive_count_error`, which is already the
    domain this repository uses for a value consumed as a ``range()`` bound: an
    integral float is not usable there (``range(2.0)`` raises) and ``bool`` must
    be rejected rather than silently read as one.

    Unlike ``gamma`` this is read by the on-policy backend only - FastSAC
    optimizes per gradient step from a replay buffer and has no epoch loop over a
    rollout batch - so it is scoped like :func:`gae_lambda_problems`: per
    :class:`TrainSpec` a backend ignores the fields it does not support, so
    reporting on one it never reads would be a false rejection.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single-element list when ``num_learning_epochs`` cannot be honored;
        empty otherwise.
    """
    error = positive_count_error(getattr(spec, "num_learning_epochs", 1), "num_learning_epochs", context)
    return [error] if error is not None else []
