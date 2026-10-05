"""Policy factory - create_policy() and runtime registration."""

import difflib
import importlib
import inspect
import logging
import os
import warnings
from collections.abc import Callable, Iterable, Mapping
from types import ModuleType
from typing import Any

from strands_robots import refusal_codes
from strands_robots.policies.base import Policy
from strands_robots.registry import (
    get_policy_provider,
    list_policy_aliases,
    list_policy_providers,
    resolve_policy,
)

# The one canonicalisation rule, shared rather than restated: a decision keyed
# on a provider name has to resolve the caller's spelling first, and a second
# copy of that rule here is a second thing to keep in step with policies.json.
from strands_robots.registry.policies import (
    _canonical_provider_name,
    _matches_declared_url_pattern,
    _url_scheme_refusal,
    _with_lowercase_url_scheme,
    removed_provider_error,
)

logger = logging.getLogger(__name__)

#
# Runtime registration (for user-defined providers not in JSON)
#

_runtime_registry: dict[str, Callable[[], type[Policy]]] = {}
_runtime_aliases: dict[str, str] = {}


def register_policy(
    name: str,
    loader: Callable[[], type[Policy]],
    aliases: list[str] | None = None,
):
    """Register a custom policy provider at runtime.

    Use this to add providers without editing policies.json.

    Example::

        from strands_robots.policies import register_policy

        register_policy("my_provider", lambda: MyPolicy, aliases=["my"])
        policy = create_policy("my_provider", ...)
    """
    _runtime_registry[name] = loader
    if aliases:
        for alias in aliases:
            _runtime_aliases[alias] = name


def list_providers() -> list[str]:
    """List all available policy provider names (JSON + runtime)."""
    names = list_policy_providers()
    names.extend(_runtime_registry.keys())
    names.extend(_runtime_aliases.keys())
    return sorted(set(names))


def list_aliases() -> dict[str, str]:
    """Return every provider alias and the canonical name it resolves to.

    :func:`create_policy` accepts a provider's declared aliases and
    shorthands as readily as its canonical name, but
    :func:`list_providers` reports the canonical names from the JSON
    registry. Together the two surfaces enumerate every spelling the
    registries hold::

        registered = set(list_providers()) | set(list_aliases())

    That is every *registered* spelling, not every spelling
    :func:`create_policy` resolves.
    :func:`import_policy_class` falls back
    to auto-discovery, so a module under ``strands_robots.policies`` that
    exports a :class:`~strands_robots.policies.base.Policy` subclass resolves
    under its own module name with no registry entry. Two ship, and neither is
    a registry provider because each wraps a policy the caller already holds
    rather than building one from config:

    * ``composite``
      (:class:`~strands_robots.policies.composite.CompositePolicy`) builds
      through this factory -- ``create_policy("composite", lower=..., upper=...)``
      -- and is the one spelling ``registered`` above omits.
    * ``persistent``
      (:class:`~strands_robots.policies.persistent.PersistentPolicy`) resolves
      but cannot be built here: its first parameter is named ``provider``,
      which :func:`create_policy` has already bound, so it is constructed
      directly. :func:`create_policy` refuses it with a ``TypeError`` that
      says so.

    Covers both registries, matching the union :func:`list_providers`
    reports: aliases declared in ``policies.json`` and aliases passed to
    :func:`register_policy` at runtime. A runtime alias shadows a JSON
    alias of the same name, which is the precedence
    :func:`create_policy` applies.

    Returns:
        Mapping of alias to the canonical provider name it resolves to.
    """
    return {**list_policy_aliases(), **_runtime_aliases}


class UntrustedRemoteCodeError(RuntimeError):
    """Raised when a HF model requires trust_remote_code but the user has not opted in.

    Carries a stable machine-readable :attr:`code`
    (:data:`~strands_robots.refusal_codes.TRUST_REMOTE_CODE_REQUIRED`) and the
    :attr:`subject` provider, so a consumer offering the operator the opt-in
    classifies on identity instead of matching the message text. The message
    is unchanged by this. See :mod:`strands_robots.refusal_codes`.

    Args:
        message: The operator-facing reason, unchanged by the code.
        code: A member of :data:`~strands_robots.refusal_codes.REFUSAL_CODES`.
        subject: The policy provider the gate refused.

    Attributes:
        code: The stable identifier for this refusal, or ``None``.
        subject: The policy provider the gate refused, or ``None``.
    """

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        subject: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.subject = subject


def construction_failure_keeps_its_raise(exc: BaseException) -> bool:
    """Whether a raise out of a provider constructor must travel on as an exception.

    Every rollout surface that builds a policy for a caller (the simulation's
    ``run_policy`` / ``eval_policy`` / ``evaluate_benchmark``, the ``run_policy``
    agent tool) answers a refused configuration as a ``status=error`` envelope,
    and a constructor is the last judge of its configuration: a checkpoint id
    that is not on the Hub (``FileNotFoundError``), a server nobody listens on
    (``ConnectionError``), a checkpoint directory without its ONNX
    (``RuntimeError``), a port outside 1-65535 (``ValueError``), a keyword the
    constructor does not bind (``TypeError``). Each of those is this
    configuration's verdict and belongs in the envelope.

    Two raises do not. Both name their remedy already and hold for every call
    on this process rather than for this configuration, so turning them into an
    error string a caller may retry past would hide a decision the process
    needs to make once:

    * the remote-code gate, :class:`UntrustedRemoteCodeError` (a security
      decision, not a configuration one);
    * a missing optional dependency: an ``ImportError``, or a provider's own
      error wrapping one as its cause (``raise RuntimeError(...) from e``, the
      shape the ``wbc`` provider uses when ``onnxruntime`` is absent).

    Args:
        exc: The exception the constructor raised.

    Returns:
        ``True`` when the caller must re-raise ``exc``; ``False`` when it is a
        refusal of this configuration and belongs in the caller's envelope.
    """
    if isinstance(exc, UntrustedRemoteCodeError | ImportError):
        return True
    cause = exc.__cause__
    return isinstance(cause, ImportError)


# Providers whose HuggingFace model loading path calls ``trust_remote_code=True``.
# Any provider that downloads and executes code from a model repository
# **must** be listed here so users are forced to explicitly opt in.
#: Providers announced for removal in 0.7, each with what replaces it. The
#: announcement is a warning from :func:`create_policy`, one minor ahead of
#: the cut, so a caller learns the replacement before the provider is gone.
_REMOVED_IN_0_7: dict[str, str] = {
    "curobo": "simulation.motion_primitives with mink IK for a sim reach, or Isaac cuMotion for GPU planning",
    "moveit2": "a MoveIt goal sent as a ROS 2 action through the use_ros or use_rosbridge tool",
    "kimodo": "a motion generated offline and replayed as joint targets (nothing in-tree)",
    "protomotions": "the wbc provider for Unitree G1 whole-body control",
}


_HF_REMOTE_CODE_PROVIDERS: frozenset[str] = frozenset(
    {
        "lerobot_local",
        "kimodo",
    }
)


def _check_trust_remote_code(provider: str) -> None:
    """Enforce the trust-remote-code gate for HuggingFace-backed providers.

    Only providers listed in ``_HF_REMOTE_CODE_PROVIDERS`` are gated.
    These providers load models with ``trust_remote_code=True``, which
    allows **arbitrary code execution** from the model repository.

    Set the environment variable ``STRANDS_TRUST_REMOTE_CODE=1`` to opt in.
    """
    if provider not in _HF_REMOTE_CODE_PROVIDERS:
        return

    opted_in = os.environ.get("STRANDS_TRUST_REMOTE_CODE", "").strip()
    if opted_in in ("1", "true", "yes"):
        return

    raise UntrustedRemoteCodeError(
        f"Policy provider '{provider}' loads HuggingFace models with "
        f"trust_remote_code=True, which allows arbitrary code execution "
        f"from the model repository.\n\n"
        f"Only load models from organisations you trust.\n\n"
        f"To acknowledge this risk and proceed, set the environment variable:\n"
        f"    export STRANDS_TRUST_REMOTE_CODE=1\n",
        code=refusal_codes.TRUST_REMOTE_CODE_REQUIRED,
        subject=provider,
    )


def _is_smart_string(provider: str) -> bool:
    """Whether ``provider`` has the shape of an address or a checkpoint :func:`resolve_policy` interprets.

    Three shapes qualify: a ``scheme://`` URL (an undeclared scheme is refused
    downstream by name), a scheme-less address some provider's
    ``url_patterns`` declares, and a checkpoint - a HuggingFace ``org/repo`` or
    a filesystem path with a name in it. Anything else is read as a provider
    name, so a typo such as ``"wbc/"``, ``":"`` or ``"protomotions:"`` reaches
    the did-you-mean lookup instead of being forwarded to ``lerobot_local`` as
    a checkpoint id the caller never named.
    """
    spelling = provider.strip()
    if "://" in spelling or _matches_declared_url_pattern(_with_lowercase_url_scheme(spelling)):
        return True
    if "/" not in spelling:
        return False
    names = [part for part in spelling.split("/") if part.strip(".~")]
    return len(names) >= 2 or (bool(names) and spelling.startswith(("/", "./", "../", "~/")))


def provider_can_be_created(provider: Any) -> bool:
    """Whether :func:`create_policy` could resolve ``provider`` - without importing it.

    A pre-flight check refuses a provider *before* spending something expensive
    on it (energizing an arm, asking an operator), so it must answer exactly the
    question :func:`create_policy` answers, from the same three stages
    :func:`_resolve_policy_class` walks - the runtime registry that the public
    :func:`register_policy` API fills (a name or one of its aliases), a smart
    string :func:`resolve_policy` interprets, then the shipped registry and the
    ``strands_robots.policies.<name>`` auto-discovery that
    :func:`~strands_robots.registry.policies.policy_provider_resolves` mirrors.
    Asking only the last stage refused every runtime-registered provider as
    unknown at every hardware entry point, while ``create_policy`` built it.

    Optimistic where resolution is: a smart string is reported as resolving
    unless its URL scheme is one no provider declares (the one refusal that
    needs neither the network nor the Hub), and a registered loader is never
    invoked here.

    Args:
        provider: Any spelling a caller may supply. ``None``/empty/non-string
            resolves to nothing.

    Returns:
        True when ``create_policy(provider)`` would get past provider lookup.
    """
    if not provider or not isinstance(provider, str):
        return False
    if _runtime_aliases.get(provider, provider) in _runtime_registry:
        return True
    if removed_provider_error(provider) is not None:
        return False
    if _is_smart_string(provider):
        return _url_scheme_refusal(provider) is None
    from strands_robots.registry.policies import policy_provider_resolves

    return policy_provider_resolves(provider)


def _provider_import_error(provider: str, exc: ImportError, extra: str | None) -> ImportError:
    """Translate a failed provider-module import into an actionable error.

    A policy provider's module may import an optional dependency at import time
    (e.g. ``lerobot_local`` imports ``torch``). When that dependency is absent
    the import machinery raises a bare ``ModuleNotFoundError: No module named
    'torch'`` which names neither the provider the caller asked for nor the way
    to fix it -- so a caller who asked for one provider is left holding an error
    about a package they never mentioned.

    Every other provider defers its heavy import and reports the remedy through
    :func:`~strands_robots.utils.require_optional` /
    :func:`~strands_robots.utils.require_optionals`, which name the extra that
    ships the dependency. This is the same report for the providers whose
    dependency is needed to import the module at all, so the remedy does not
    depend on WHERE a provider happens to import its dependency.

    Args:
        provider: Canonical provider name the caller asked for.
        exc: The ``ImportError`` raised while importing the provider's module.
        extra: ``pyproject.toml`` extras group that ships the dependency, as
            declared by the provider's ``extra`` field in ``policies.json``.
            ``None`` when the provider declares none, in which case the missing
            module is named without an install command for a specific extra.

    Returns:
        An ``ImportError`` naming the provider, the missing module and the
        remedy. The caller should ``raise ... from exc`` to keep the original
        traceback.
    """
    missing = getattr(exc, "name", None) or "an optional dependency"
    if extra:
        remedy = f"Install the extra that ships it:\n  uv pip install 'strands-robots[{extra}]'"
    else:
        remedy = f"Install {missing!r} (or the strands-robots extra that ships it) and retry."
    return ImportError(
        f"Policy provider {provider!r} needs an optional dependency that is not installed:\n  {exc}\n\n{remedy}"
    )


def import_policy_class(provider: str) -> type:
    """Dynamically import and return the Policy class for a provider.

    Uses the module + class paths from policies.json.  Falls back to
    auto-discovery (strands_robots.policies.<name>) if not in JSON.

    Args:
        provider: Canonical provider name.

    Returns:
        The Policy subclass.

    Raises:
        ValueError: If the provider does not exist, or was removed - a removed
            spelling (``groot``) is refused with the sentence
            :data:`~strands_robots.registry.policies.REMOVED_PROVIDERS` holds
            for it, never rerouted to another provider.
        ImportError: If the provider exists but its module cannot be imported,
            naming the provider, the missing module and the remedy (see
            :func:`_provider_import_error`). A provider whose module is present
            but whose optional dependency is missing reports that rather than
            being misreported as an unknown provider.
    """
    if (removed := removed_provider_error(provider)) is not None:
        raise ValueError(removed)
    config = get_policy_provider(provider)
    if config:
        # get_policy_provider already keyed the lookup on the canonical name,
        # so config IS the canonical entry; the name is needed for the report.
        canonical = _canonical_provider_name(provider)
        try:
            mod = importlib.import_module(config["module"])
        except ImportError as exc:
            # A provider whose module needs an optional dependency at import
            # time (lerobot_local imports torch) otherwise raises a bare
            # "No module named 'torch'" naming neither this provider nor the
            # remedy - the dead end _provider_import_error exists to close.
            raise _provider_import_error(canonical, exc, config.get("extra")) from exc
        return getattr(mod, config["class"])

    # Auto-discovery fallback, for spellings that can name a module at all.
    module_name = f"strands_robots.policies.{provider}"
    discovered: ModuleType | None = None
    try:
        if provider.isidentifier():
            discovered = importlib.import_module(module_name)
    except ImportError as exc:
        # Distinguish "this provider does not exist" from "it exists but its
        # optional dependency is missing". Only the former is an unknown
        # provider; reporting the latter that way sends the caller to check a
        # name that was correct.
        if getattr(exc, "name", None) != module_name:
            raise _provider_import_error(provider, exc, None) from exc
    if discovered is not None:
        class_name = f"{provider.capitalize()}Policy"
        if hasattr(discovered, class_name):
            return getattr(discovered, class_name)
        for attr_name in dir(discovered):
            attr = getattr(discovered, attr_name)
            if isinstance(attr, type) and issubclass(attr, Policy) and attr is not Policy:
                return attr

    # Offer the nearest registered spellings, the way Robot() does for a robot
    # name: case and dash are folded, and 0.6 is Robot()'s cutoff, which is
    # what lets NVIDIA's own spelling ``gr00t`` find ``groot``. The search pool
    # mixes canonical names and their aliases/shorthands so a near miss of a
    # shorthand lands too; the hint is then collapsed back to canonical names
    # so a single typo doesn't show two spellings of the same provider (``gtp``
    # alongside ``protomotions``) and -- the sharper edge -- never leaks a
    # shorthand that routes to a *different* canonical (``text2motion`` ->
    # ``kimodo``) into the suggestions for a typo of ``protomotions``.
    folded = provider.lower().replace("-", "_")
    close = difflib.get_close_matches(folded, [*list_providers(), *list_aliases()], n=3, cutoff=0.6)
    seen: set[str] = set()
    deduped: list[str] = []
    for name in close:
        canonical = _canonical_provider_name(name)
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(canonical)
    hint = f" Did you mean: {', '.join(map(repr, deduped))}?" if deduped else ""
    raise ValueError(f"Unknown policy provider: '{provider}'.{hint} Available: {list_policy_providers()}")


def _spell_model_path_as_the_provider_does(provider: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The wire's generic ``model_path`` under the key the provider declares for it.

    ``model_path`` is the one checkpoint key the mesh carries (validated for
    traversal on the robot host, contained under the checkpoint homes by the
    dashboard). A provider that spells that key ``checkpoint`` in its
    ``config_keys`` (``wbc``, ``wbc_gait``) and does not declare ``model_path``
    used to receive it through ``**kwargs`` and drop it, so a G1 asked to walk
    over the wire failed at load with "no checkpoint" while the caller had sent
    one. Mapped here, once, for every caller of ``create_policy``; an explicit
    ``checkpoint`` in ``kwargs`` wins.
    """
    if "model_path" not in kwargs:
        return dict(kwargs)
    canonical = _canonical_provider_name(provider)
    info = get_policy_provider(canonical) or {}
    keys = set(info.get("config_keys") or [])
    if "checkpoint" not in keys or "model_path" in keys:
        return dict(kwargs)
    out = dict(kwargs)
    value = out.pop("model_path")
    out.setdefault("checkpoint", value)
    return out


def _provider_type_error(provider: object, param: str) -> str | None:
    """Describe why a non-string ``provider`` names nothing, or ``None`` for a string.

    Resolution strips and indexes the registry with this value, so a non-string
    otherwise surfaces as a bare ``AttributeError``/``TypeError`` naming neither
    the parameter nor the problem. A :class:`Policy` (instance or class) is the
    likeliest wrong value, so the message points at ``policy_object``.
    """
    if isinstance(provider, str):
        return None
    hint = (
        " A pre-built policy is passed as policy_object= instead."
        if isinstance(provider, Policy) or (isinstance(provider, type) and issubclass(provider, Policy))
        else ""
    )
    return (
        f"{param} must be a string, got {type(provider).__name__} ({provider!r}). "
        "Pass a provider name (list_providers() reports them), a HuggingFace "
        f"model ID, or a server URL.{hint}"
    )


def _resolve_policy_class(provider: str, /, **kwargs) -> tuple[str, type[Policy], dict]:
    """Resolve ``provider`` to its policy class WITHOUT instantiating it.

    Imports the class and computes the effective constructor kwargs using the
    same three-stage lookup as :func:`create_policy` (runtime registry, smart
    string, then ``policies.json``), but never calls the constructor and never
    enforces the trust-remote-code gate. This lets callers inspect or run a
    class-level :meth:`Policy.preflight` check before paying the cost (and,
    for remote-code providers, the risk) of construction.

    Args:
        provider: Provider name, HF model ID, or server URL.
        **kwargs: Provider-specific parameters.

    Returns:
        ``(canonical_provider_name, PolicyClass, resolved_kwargs)``.

    Raises:
        TypeError: If ``provider`` is not a string.
        ImportError / ValueError: Propagated from the underlying class import
            or smart-string resolution when the provider cannot be resolved.
    """
    if (type_error := _provider_type_error(provider, "provider")) is not None:
        raise TypeError(type_error)
    # 1. Runtime registry (user-registered providers).
    resolved_name = _runtime_aliases.get(provider, provider)
    if resolved_name in _runtime_registry:
        return resolved_name, _runtime_registry[resolved_name](), dict(kwargs)
    kwargs = _spell_model_path_as_the_provider_does(provider, kwargs)

    # 2. Smart string (HF ID, URL, etc.).
    if _is_smart_string(provider):
        try:
            resolved_provider, resolved_kwargs = resolve_policy(provider, **kwargs)
        except ImportError:
            pass  # not installed as a smart string; fall through to the registry lookup
        else:
            if resolved_provider:
                return resolved_provider, import_policy_class(resolved_provider), dict(resolved_kwargs)

    # 3. Standard lookup from policies.json. The name returned is the canonical
    #    one, not the caller's spelling: create_policy keys the
    #    trust-remote-code gate on it and that gate membership-tests a set of
    #    canonical names, so returning a declared alias would skip the gate for
    #    every spelling but one. Stages 1 and 2 already canonicalise (the
    #    runtime alias map, and resolve_policy's shorthand stage); this is the
    #    third.
    return _canonical_provider_name(provider), import_policy_class(provider), dict(kwargs)


# ``policy_config`` (and the per-call ``policy_kwargs``) are opaque provider
# keyword bags: callers hand them to ``create_policy`` / ``get_actions``, which
# splat them with ``**``. A non-mapping value therefore fails inside CPython's
# call machinery with a bare ``TypeError`` naming this module's internals, which
# tells the caller nothing about which parameter to fix. Callers validate the
# value against this helper first and wrap the message in their own error
# envelope, mirroring ``VideoConfig.validation_error``.
_POLICY_MAPPING_HINTS: dict[str, str] = {
    "policy_config": (
        "provider kwargs forwarded to create_policy, e.g. policy_config={'host': '127.0.0.1', 'port': 5555}"
    ),
    "policy_kwargs": ("per-call kwargs forwarded to policy.get_actions, e.g. policy_kwargs={'target_pose': [...]}"),
}


def policy_mapping_error(value: object, param: str = "policy_config") -> str | None:
    """Describe why ``value`` cannot be used as a provider keyword mapping.

    ``policy_config`` / ``policy_kwargs`` are free-form dicts with no signature
    to bounce off, so a value of the wrong *shape* - a ``"host=1"`` string, a
    list of pairs, a JSON blob an agent forgot to parse - is only detected when
    CPython splats it, far from the call the caller made.

    Args:
        value: The caller-supplied value, or ``None`` (always accepted: the
            parameter is optional).
        param: Parameter name to quote in the message; also selects the
            example shown. Unknown names fall back to a generic hint.

    Returns:
        A single-sentence explanation naming the parameter, the type received
        and a correct example, or ``None`` when ``value`` is usable as ``**``
        keyword arguments.
    """
    if value is None or isinstance(value, Mapping):
        return None
    hint = _POLICY_MAPPING_HINTS.get(param, "keyword arguments")
    return f"{param} must be a dict of {hint}; got {type(value).__name__} ({value!r})."


def policy_object_error(value: object, param: str = "policy_object") -> str | None:
    """Describe why ``value`` cannot be driven as a pre-built policy.

    ``policy_object`` is the sibling of the keyword bags
    :func:`policy_mapping_error` guards, and it fails the same way for the same
    reason: it is an opaque parameter with no signature to bounce off, so a
    value of the wrong shape is only detected when the rollout reaches for a
    method on it. That happens well after the call the caller made, and on
    ``start_policy`` it happens on a worker thread whose result nothing reads -
    so the caller is handed ``status="success"`` for a rollout that never
    produced an action.

    Unlike the bags, this parameter is *bypass* rather than configuration: a
    ``policy_object`` is driven directly, so it skips provider resolution and
    the provider's ``preflight`` hook. Nothing downstream can turn the value
    into a policy, which is why the domain is checked here.

    A ``Policy`` SUBCLASS is called out separately: passing the class instead of
    an instance is the likeliest version of this mistake, and it is the one
    whose unguarded failure is least legible (attribute access on a class
    reaches unbound descriptors rather than a missing attribute).

    Args:
        value: The caller-supplied value, or ``None`` (always accepted: the
            parameter is optional and a provider is named instead).
        param: Parameter name to quote in the message.

    Returns:
        A single-sentence explanation naming the parameter, what arrived and how
        to obtain a usable value, or ``None`` when ``value`` can be driven.
    """
    if value is None or isinstance(value, Policy):
        return None
    if isinstance(value, type) and issubclass(value, Policy):
        return (
            f"{param} must be a Policy instance; got the class {value.__name__} itself. "
            f"Instantiate it ({value.__name__}()), or omit {param} and name policy_provider "
            "to have one built."
        )
    return (
        f"{param} must be a Policy instance; got {type(value).__name__} ({value!r}). It is driven "
        "directly, so it bypasses provider resolution and nothing downstream can turn this value "
        f"into a policy. Pass an instance (create_policy(provider, **config) returns one), or omit "
        f"{param} and name policy_provider to have one built."
    )


# A residual keyword scoring at least this against a declared constructor
# parameter is a misspelling of it, not another option. The cutoff is the one
# ``simulation.base.reject_misspelled_kwargs`` uses for engine kwargs, so a typo
# is judged the same way whichever sink it lands in; not imported from there
# because the policies package does not depend on the simulation package.
_MISSPELLING_RATIO = 0.8


def _constructor_keywords(PolicyClass: type) -> tuple[tuple[str, ...], bool]:
    """The keyword names a provider's constructor binds, and whether it has a sink.

    Returns:
        ``(accepted, tolerates_unknown)`` - the parameters a caller can spell by
        keyword (``self`` and the sinks omitted, in declaration order) and
        whether the constructor declares ``**kwargs``. Empty and ``True`` when
        the class has no introspectable signature, which turns screening into
        a no-op rather than refusing every keyword.
    """
    try:
        params = inspect.signature(PolicyClass).parameters
    except (TypeError, ValueError):
        return (), True
    accepted = tuple(
        name
        for name, p in params.items()
        if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL, p.POSITIONAL_ONLY)
    )
    tolerates_unknown = any(p.kind is p.VAR_KEYWORD for p in params.values())
    return accepted, tolerates_unknown


def _mistyped_in_place(name: str, candidate: str) -> bool:
    """Whether ``name`` is ``candidate`` with a character mistyped in place.

    One wrong character, or two adjacent characters in each other's place - the
    two typos that leave a name's length alone, and the only ones
    :data:`_MISSPELLING_RATIO` cannot see. ``difflib``'s ratio is
    ``2 * matches / total``: a dropped or doubled character costs one match but
    also changes the total (0.857 and 0.889 against a four-letter name, and
    higher for every longer one), while a wrong character costs a match with the
    total unchanged, scoring ``2 * (n - 1) / 2n`` - 0.750 at four characters,
    0.800 at five. So the cutoff screens every length-changing typo of every
    parameter this package declares, and leaves exactly one class open: a wrong
    character in a four-letter name. Five parameters are four characters -
    ``host``, ``port``, ``mode``, ``seed``, ``walk`` - and ``host`` and ``port``
    are the two the most providers declare.

    Args:
        name: The keyword the caller spelled.
        candidate: A parameter the constructor binds.

    Returns:
        Whether one typo in ``candidate`` produces ``name``.
    """
    if len(name) != len(candidate):
        return False
    differing = [i for i, (a, b) in enumerate(zip(name, candidate, strict=True)) if a != b]
    if len(differing) == 1:
        return True
    if len(differing) == 2:
        first, second = differing
        # Adjacency is implied by the swap identity below (the character between
        # two non-adjacent differences matches, which the identity contradicts),
        # and stated because it is the invariant a reader needs.
        return second == first + 1 and name[first] == candidate[second] and name[second] == candidate[first]
    return False


def _misspelling_of(name: str, accepted: tuple[str, ...]) -> str | None:
    """The accepted parameter ``name`` misspells, or ``None``.

    Two tests, because neither covers the other. A close match at
    :data:`_MISSPELLING_RATIO` catches a name off a parameter by a character it
    dropped, doubled, or by several characters. :func:`_mistyped_in_place`
    catches the one class the ratio scores too low to see - a wrong character
    in a four-letter name: ``hoat`` and ``hots`` for ``host``, ``porr`` and
    ``prot`` for ``port`` all score 0.750. A provider whose constructor has a
    ``**kwargs`` sink drops such a name silently, which is byte-identical to
    omitting the argument, so the policy dials the default host and reports
    success.

    Nothing legitimate is one typo from a parameter the same constructor binds:
    a pass-through option is another subsystem's name, not a near-miss of this
    one's. Measured over the parameters of every registered provider, no name
    any of them declares is one typo from a parameter of another.
    """
    match = difflib.get_close_matches(name, list(accepted), n=1, cutoff=_MISSPELLING_RATIO)
    if match:
        return match[0]
    for candidate in accepted:
        if _mistyped_in_place(name, candidate):
            return candidate
    return None


def policy_kwargs_error(provider: str, PolicyClass: type, kwargs: Mapping[str, Any]) -> str | None:
    """Why ``kwargs`` cannot be handed to ``PolicyClass`` as written, or ``None``.

    One rule for every provider, applied before construction. Pre-fix each
    provider had its own: a constructor with ``**kwargs`` dropped
    ``create_policy("moveit2", hots="x")`` silently (the client dialled the
    default host under ``status="success"``), ``remote`` logged
    "ignoring unexpected constructor kwarg(s)" where no agent reads it,
    and a constructor without a sink raised CPython's
    ``__init__() got an unexpected keyword argument 'acton_space'`` - which
    names neither the provider nor the parameter meant.

    A name the constructor binds passes. A name that misspells one it binds
    (:data:`_MISSPELLING_RATIO`) is refused naming the parameter meant: no
    call can intend it, and a sink makes it byte-identical to omitting the
    argument. A name that is neither is refused when the constructor has no
    sink (it would have raised anyway - this names the provider and lists
    what it does accept) and tolerated when it has one, because a provider's
    ``**kwargs`` is its documented pass-through (model-loader options,
    another provider's keys on a shared ``policy_config``), logged at DEBUG so
    it is visible somewhere.

    Args:
        provider: The canonical provider name, quoted in the report.
        PolicyClass: The class about to be constructed.
        kwargs: The resolved constructor kwargs.

    Returns:
        The refusal, or ``None`` when every name is usable.
    """
    accepted, tolerates_unknown = _constructor_keywords(PolicyClass)
    if not accepted:
        return None
    if (collision := _provider_collision_error(provider, PolicyClass)) is not None:
        return collision
    # A provider may know a name that its sink would otherwise swallow: a field
    # that belongs on another object (lerobot_local's ``state_units`` is an
    # embodiment field, #4164). Its ``misplaced_kwargs_error`` says where the
    # name goes, and runs here so the caller learns it before the trust gate.
    misplaced_error = getattr(PolicyClass, "misplaced_kwargs_error", None)
    if callable(misplaced_error) and (misplaced := misplaced_error(kwargs)) is not None:
        return str(misplaced)
    owner = f"{PolicyClass.__name__} (policy provider {provider!r})"
    misspelled: list[str] = []
    unknown: list[str] = []
    for name in kwargs:
        if name in accepted:
            continue
        meant = _misspelling_of(name, accepted)
        if meant is not None:
            misspelled.append(f"{name!r} (did you mean {meant!r}?)")
        else:
            unknown.append(name)
    if misspelled:
        return (
            f"{owner} does not accept {', '.join(misspelled)}. A misspelling of a parameter it does read "
            "cannot be a pass-through option, so it is refused rather than dropped - dropped, it would be "
            "byte-identical to omitting the argument and the policy would run on the default. Fix the "
            f"spelling, or drop the argument. It accepts: {', '.join(accepted)}."
        )
    if unknown and not tolerates_unknown:
        names = ", ".join(repr(n) for n in unknown)
        return (
            f"{owner} does not accept {names}: its constructor declares no **kwargs, so there is nothing "
            f"to forward them to. It accepts: {', '.join(accepted)}. Drop the argument, or check the "
            "provider's docs for the name it uses."
        )
    if unknown:
        logger.debug(
            "%s forwarded %s to its **kwargs: no parameter of that name, and no close match to one. "
            "Expected for a pass-through option; otherwise it is an unsupported name.",
            owner,
            sorted(unknown),
        )
    if missing := [name for name in _required_keywords(PolicyClass) if name not in kwargs]:
        return f"{owner} requires {', '.join(repr(n) for n in missing)}. It accepts: {', '.join(accepted)}."
    return None


def _required_keywords(PolicyClass: type) -> tuple[str, ...]:
    """The constructor parameters a caller must pass by keyword (no default)."""
    try:
        params = inspect.signature(PolicyClass).parameters
    except (TypeError, ValueError):
        return ()
    return tuple(
        name
        for name, p in params.items()
        if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    )


def _provider_collision_error(provider: str, PolicyClass: type) -> str | None:
    """Why ``create_policy`` can never build ``PolicyClass``, or ``None``.

    ``create_policy`` binds its own first parameter, ``provider``, to the name
    being resolved. A constructor that requires a parameter of that name
    (:class:`~strands_robots.policies.persistent.PersistentPolicy` wraps an
    inner provider) can therefore receive no value for it from any keyword: the
    call fails as ``missing 1 required positional argument`` without one and
    ``got multiple values for argument 'provider'`` with one, and neither names
    the provider asked for or the way to build it.
    """
    if "provider" not in _required_keywords(PolicyClass):
        return None
    name = PolicyClass.__name__
    return (
        f"policy provider {provider!r} cannot be built by create_policy: {name} requires its own 'provider' "
        "argument, and create_policy has already bound that name to the provider being resolved. Construct "
        f"it directly: from {PolicyClass.__module__} import {name}; {name}(provider='mock', **config)."
    )


def create_policy(provider: str, /, **kwargs) -> Policy:
    """Create a policy instance.

    Accepts either a provider name or a smart string:

    - Provider name: ``create_policy("lerobot_local", pretrained_name_or_path="lerobot/act_aloha_sim")``
    - Server URL: ``create_policy("ws://gpu-box:8765")``
    - Checkpoint: ``create_policy("lerobot/act_aloha_sim")`` or a path such as
      ``create_policy("outputs/train/act/checkpoints/last/pretrained_model")``
    - Shorthand: ``create_policy("mock")``

    Any other spelling is a provider name; one carrying stray punctuation
    (``"wbc/"``, ``"protomotions:"``) is refused as an unknown provider with the
    nearest names, not forwarded to ``lerobot_local`` as a checkpoint id.

    All provider definitions live in ``registry/policies.json``.

    Args:
        provider: Provider name, HF model ID, or server URL. Positional-only,
            so a provider keyword named ``provider`` reaches the policy's
            constructor (and its refusal) instead of colliding with this one.
        **kwargs: Provider-specific parameters.

    Returns:
        Policy instance ready for get_actions().

    Warns:
        DeprecationWarning: If ``provider`` is in ``_REMOVED_IN_0_7``,
            naming its replacement.

    Raises:
        TypeError: If ``provider`` is not a string (a pre-built policy goes to
            ``policy_object=`` instead), or if a keyword misspells one the provider's constructor
            binds, names one it cannot bind at all (no ``**kwargs``), omits one
            it requires, or the provider needs its own ``provider`` argument
            (``persistent``, which is constructed directly) - see
            :func:`policy_kwargs_error`. Raised before construction and before
            the trust-remote-code gate, so no model is downloaded, no server
            dialled and no opt-in asked for on a typo.
        UntrustedRemoteCodeError: If the provider loads HF models with
            ``trust_remote_code=True`` and ``STRANDS_TRUST_REMOTE_CODE``
            is not set.
    """
    canonical, PolicyClass, resolved_kwargs = _resolve_policy_class(provider, **kwargs)
    if (replacement := _REMOVED_IN_0_7.get(canonical)) is not None:
        warnings.warn(
            f"policy provider {canonical!r} is removed in 0.7; instead use {replacement}",
            DeprecationWarning,
            stacklevel=2,
        )
    # The kwargs check is pure, so it runs first: a typo must not send the caller
    # to opt in to remote code only to learn about the typo on the second call.
    if (kwargs_error := policy_kwargs_error(canonical, PolicyClass, resolved_kwargs)) is not None:
        raise TypeError(kwargs_error)
    _check_trust_remote_code(canonical)
    return PolicyClass(**resolved_kwargs)


def preflight_policy(provider: str, observation_keys: set[str], **kwargs) -> None:
    """Run a provider's class-level :meth:`Policy.preflight` check, if any.

    Resolves ``provider`` to its policy class WITHOUT instantiating it (so no
    model weights are downloaded) and invokes the class's ``preflight`` hook
    with the runtime ``observation_keys`` and the provider kwargs. Providers
    that do not override :meth:`Policy.preflight` are a no-op.

    This is the fail-fast seam used by ``SimEngine.run_policy`` /
    ``eval_policy`` to catch a misconfiguration (e.g. sim camera names that
    cannot be routed to the model's declared image inputs) BEFORE the
    expensive ``create_policy`` download, instead of crashing deep inside the
    first inference. Resolution failures are swallowed (the matching error is
    surfaced authoritatively by the subsequent ``create_policy``); only the
    provider's own ``preflight`` ``ValueError`` propagates.

    Args:
        provider: Provider name, HF model ID, or server URL (as passed to
            ``create_policy``).
        observation_keys: Keys the runtime observation will contain (joint
            names + camera names).
        **kwargs: Provider-specific parameters (the policy_config).

    Raises:
        ValueError: When the resolved provider's ``preflight`` rejects the
            configuration.
    """
    try:
        _canonical, PolicyClass, resolved_kwargs = _resolve_policy_class(provider, **kwargs)
    except Exception as e:
        # Resolution problems (unknown provider, missing optional dep) are not
        # this hook's concern - create_policy raises the authoritative error.
        logger.debug("preflight_policy: could not resolve '%s' (%s); skipping", provider, e)
        return

    if not _overrides_preflight(PolicyClass):
        # Provider did not override the default no-op preflight.
        return
    PolicyClass.preflight(set(observation_keys), **resolved_kwargs)


def preflight_reason(
    provider: str,
    read_observation_keys: Callable[[], Iterable[str]],
    /,
    **kwargs: Any,
) -> str | None:
    """Why ``provider`` refuses this configuration, or ``None``.

    The whole pre-build check in one call, so the three entry points that owe it
    - the simulation engine, the physical arm and a native driver's task verb -
    read one rule instead of keeping three copies of it in step:

    * the observation is read only when the resolved class actually overrides
      :meth:`Policy.preflight` (:func:`policy_overrides_preflight`). That read is
      not cheap - the sim renders every camera in the scene, an arm warms and
      grabs a frame from each configured camera, a driver crosses the wire - and
      for every shipped provider but ``lerobot_local`` the result is gathered
      only to be discarded.
    * a read that fails, or answers nothing, is not a verdict on the policy
      configuration and does not become one here: the check is skipped and that
      read stays the caller's own to report.
    * the provider's ``ValueError`` comes back as text, because two of the three
      callers answer a refusal envelope rather than raise.

    Args:
        provider: Provider name, HF model ID, or server URL (as passed to
            :func:`create_policy`). Positional-only, as is the reader below, so
            a policy kwarg spelled either way reaches the hook instead of
            binding here.
        read_observation_keys: Answers the keys the runtime observation will
            carry (joint names plus camera names). Called at most once, and only
            when there is a hook to feed.
        **kwargs: Provider-specific parameters (the policy_config), judged as
            the mapping :func:`create_policy` will be given.

    Returns:
        The provider's refusal text, or ``None`` when the configuration passes,
        when there is no hook to run, or when the observation could not be read.
    """
    if not policy_overrides_preflight(provider, **kwargs):
        return None
    try:
        keys = read_observation_keys()
    except Exception as exc:  # noqa: BLE001 - a read the caller cannot serve is the caller's to report
        logger.debug("preflight skipped: observation unavailable for '%s' (%s)", provider, exc)
        return None
    if not keys:
        return None
    try:
        preflight_policy(provider, set(keys), **kwargs)
    except ValueError as exc:
        return str(exc)
    return None


def _overrides_preflight(PolicyClass: type) -> bool:
    """Whether ``PolicyClass`` replaces the default no-op :meth:`Policy.preflight`.

    The single implementation of that rule, shared by :func:`preflight_policy`
    (which runs the hook) and :func:`policy_overrides_preflight` (which lets a
    caller find out before paying to build the hook's argument).
    """
    hook = getattr(PolicyClass, "preflight", None)
    base_hook = getattr(Policy.preflight, "__func__", Policy.preflight)
    return not (hook is None or getattr(hook, "__func__", hook) is base_hook)


def policy_overrides_preflight(provider: str, **kwargs) -> bool:
    """Whether ``provider`` has a real :meth:`Policy.preflight` to run.

    Resolves ``provider`` to its policy class WITHOUT instantiating it (so no
    model weights are downloaded) and reports whether that class overrides the
    default no-op :meth:`Policy.preflight`.

    This exists so a caller can find out whether :func:`preflight_policy` will
    read its ``observation_keys`` argument BEFORE paying to produce it. That
    argument is not always cheap: ``SimEngine._preflight_policy_config`` sources
    it from ``get_observation``, which renders every camera in the scene. For
    the providers that leave ``preflight`` alone - every shipped provider except
    ``lerobot_local`` - those frames are gathered only to be discarded, once per
    ``run_policy`` / ``eval_policy`` / ``start_policy``.

    Args:
        provider: Provider name, HF model ID, or server URL (as passed to
            ``create_policy``).
        **kwargs: Provider-specific parameters (the policy_config), which can
            select the class that answers (a smart-string provider resolves
            through them).

    Returns:
        ``True`` when the resolved class overrides ``preflight``; ``False`` when
        it leaves the default no-op in place, and ``False`` when ``provider``
        cannot be resolved at all - :func:`preflight_policy` swallows resolution
        failures and degrades to a no-op for such a name, so there is likewise
        no hook to feed here.
    """
    try:
        _canonical, PolicyClass, _resolved_kwargs = _resolve_policy_class(provider, **kwargs)
    except Exception as e:
        # Same degrade-to-no-op as preflight_policy: resolution problems are
        # create_policy's to report authoritatively, not this hook's.
        logger.debug("policy_overrides_preflight: could not resolve '%s' (%s); skipping", provider, e)
        return False
    return _overrides_preflight(PolicyClass)


def policy_provider_error(provider: str, /, **kwargs) -> str | None:
    """Return why ``provider`` cannot be resolved to a policy class, or ``None``.

    Probes the SAME resolution path :func:`create_policy` uses, without
    instantiating anything, so every spelling that provider accepts -- a
    registered name, a HuggingFace model ID, a ``ws://`` / ``cosmos3://`` URL --
    resolves here too. Only a name no spelling can reach yields a reason. A
    scheme-less ``host:port`` is one of those unless a provider declares a
    scheme-less ``url_patterns`` entry for it (no shipped provider does).

    This is the agent-tool companion to :func:`preflight_policy`, which
    deliberately swallows resolution failures on the stated grounds that
    "create_policy raises the authoritative error". That premise holds for a
    library caller, which sees the raise. It does not hold for the simulation's
    agent-tool surfaces: a raise out of ``run_policy`` / ``eval_policy``
    escapes the ``status=error`` envelope those tools are documented to return,
    and ``start_policy`` builds the policy on a worker thread, so the raise is
    never surfaced at all and the caller is told the policy started. Returning
    the reason lets each surface report it on its own channel instead.

    The returned message names every registered provider, so a caller that
    guessed a name gets the available set back rather than a traceback.

    A non-string ``provider`` is refused here too: resolution indexes the
    registry with it, so it would otherwise arrive as a bare ``TypeError``
    naming neither the parameter nor the problem.

    Args:
        provider: Provider name, HF model ID, or server URL (as passed to
            ``create_policy``).
        **kwargs: Provider-specific parameters (the policy_config), forwarded
            so resolution sees exactly what ``create_policy`` will.

    Returns:
        The resolution failure message, or ``None`` when ``provider`` resolves.
    """
    if (type_error := _provider_type_error(provider, "policy_provider")) is not None:
        return type_error
    try:
        canonical, PolicyClass, _ = _resolve_policy_class(provider, **kwargs)
    except ValueError as e:
        # ValueError is the unresolvable-NAME verdict. A missing optional
        # dependency (ImportError) and the trust-remote-code gate are separate
        # concerns with their own reporting, and are deliberately not caught.
        return str(e)
    # No keyword can satisfy this one, so it is a property of the provider, not
    # of the config: report it on the same channel as an unknown name.
    return _provider_collision_error(canonical, PolicyClass)
