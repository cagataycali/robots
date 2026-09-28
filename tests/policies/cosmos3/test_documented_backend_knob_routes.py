"""Every knob the cosmos3 pages tell a reader to pass must be passable.

The in-process ``diffusers`` backend is configured on
:class:`~strands_robots.policies.cosmos3.policy_diffusers.Cosmos3DiffusersBackend`,
but the page's worked examples build the policy - :class:`Cosmos3Policy` /
``create_policy("cosmos3", ...)`` - and that constructor forwards only
``embodiment``, ``model`` and ``mode``. It takes no ``**kwargs``, so a keyword
the page tells the reader to pass and that constructor does not declare is a
``TypeError`` rather than a slower path, and the registry route drops it: the
provider's ``config_keys`` filter keeps only the keys the constructor names, so
``build_policy_kwargs`` returns without it and the run proceeds with the
opposite setting reported as success.

So the rule graded here is that an instruction to pass a keyword has to name a
receiver the page also names. That is checked against ``inspect.signature`` of
the classes the page actually mentions rather than against a copied list, so a
knob promoted onto the policy later, or a newly documented one, is graded
without touching this file.

The pages' Python fences are graded the same way in the other direction: every
keyword they pass must be accepted by the call's own receiver.

One page documents the policy now, ``learn/policies/cosmos3.md``; the old
provider and in-process backend pages both redirect there. Its constructor
table is the ``{{providers:kwargs:cosmos3}}`` token that ``docs/hooks/providers.py``
expands at build time, so the page is read through that hook, and every keyword
the table names is an instruction to the reader as much as a ``pass `kw=` ``
sentence is. The worked example reaches the policy through the registry route
(``sim.run_policy(policy_provider="cosmos3", policy_config={...})``), so each
``policy_config`` key is graded through ``build_policy_kwargs``: a key that
route drops is exactly the silent failure described above.
"""

import ast
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.cosmos3 import Cosmos3DiffusersBackend, Cosmos3Policy
from strands_robots.policies.cosmos3.client import Cosmos3WebsocketClient
from strands_robots.policies.cosmos3.embodiments import get_embodiment
from strands_robots.registry import build_policy_kwargs
from strands_robots.simulation import create_simulation

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCS = (_REPO_ROOT / "docs" / "learn" / "policies" / "cosmos3.md",)
_DOC_NAMES = " + ".join(p.name for p in _DOCS)
_PROVIDERS_HOOK = _REPO_ROOT / "docs" / "hooks" / "providers.py"
_FENCE = re.compile(r"```python[^\n]*\n(.*?)```", re.S)

# Classes the page can name as the receiver of a documented keyword.
_RECEIVERS: dict[str, type] = {
    "Cosmos3Policy": Cosmos3Policy,
    "Cosmos3DiffusersBackend": Cosmos3DiffusersBackend,
    "Cosmos3WebsocketClient": Cosmos3WebsocketClient,
}

# A clean sweep must mean the page is right, not that nothing was graded.
_MINIMUM_INSTRUCTIONS = 1
_MINIMUM_FENCE_KEYWORDS = 15


def _providers_hook() -> Any:
    """Load ``docs/hooks/providers.py`` from its file, the way mkdocs does."""
    spec = importlib.util.spec_from_file_location("docs_providers_hook_cosmos3", _PROVIDERS_HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _page() -> str:
    """Return the cosmos3 page as the build renders it (providers tokens expanded)."""
    hook = _providers_hook()
    rendered = "\n".join(hook.substitute(p.read_text(encoding="utf-8"), p.name) for p in _DOCS)
    assert "{{providers:" not in rendered, f"the providers hook left a token unexpanded on {_DOC_NAMES}"
    return rendered


def _named_receivers(page: str) -> dict[str, type]:
    """Receivers the page mentions by name, plus the factory that forwards to one."""
    named = {name: cls for name, cls in _RECEIVERS.items() if name in page}
    if "create_policy(" in page and '"cosmos3"' in page:
        named['create_policy("cosmos3")'] = Cosmos3Policy
    return named


def _params(cls: type) -> set[str]:
    """Keyword names ``cls(...)`` accepts."""
    return set(inspect.signature(cls).parameters)


def _instructed_keywords(page: str) -> set[str]:
    """Keywords the page tells the reader to pass or set.

    Three shapes count: a ``pass `kw=` `` or ``set `kw=` `` sentence, a
    backticked ``kw="value"`` the prose hands the reader, and a row of the
    Constructor keywords table (``| `kw` | type | default |``).
    """
    found: set[str] = set()
    for pattern in (
        r"(?:pass|set)\s+`([a-z_][a-z0-9_]*)\s*=",
        r"\(`([a-z_][a-z0-9_]*)\s*=[^`]*`",
        r"`([a-z_][a-z0-9_]*)=[^`]+`",
    ):
        found |= set(re.findall(pattern, page))
    found |= _table_keywords(page)
    return found


def _table_keywords(page: str) -> set[str]:
    """Keyword names the Constructor keywords table lists."""
    section = re.search(r"## Constructor keywords\n(.*?)(?:\n## |\Z)", page, re.S)
    if section is None:
        return set()
    return set(re.findall(r"^\| `([a-z_][a-z0-9_]*)` \|", section.group(1), re.M))


def _listed_backend_knobs(page: str) -> set[str]:
    """Knob names the page presents as carried by the backend object route."""
    block = re.search(r"load and sampling knobs\s*\n?>?\s*\(([^)]*)\)", page, re.S)
    if block is None:
        return set()
    return set(re.findall(r"`([a-z_][a-z0-9_]*)`", block.group(1)))


def _fence_keywords(page: str) -> dict[str, set[str]]:
    """Keyword arguments each Python fence passes, grouped by the call's receiver."""
    per_call: dict[str, set[str]] = {}
    for block in _FENCE.findall(page):
        try:
            tree = ast.parse(block)
        except SyntaxError:  # a fence may be an excerpt
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = ast.unparse(node.func)
                for keyword in node.keywords:
                    if keyword.arg:
                        per_call.setdefault(target, set()).add(keyword.arg)
    return per_call


def _registry_route_configs(page: str) -> list[dict[str, Any]]:
    """Every ``policy_config`` literal a fence hands to ``policy_provider="cosmos3"``."""
    configs: list[dict[str, Any]] = []
    for block in _FENCE.findall(page):
        try:
            tree = ast.parse(block)
        except SyntaxError:  # a fence may be an excerpt
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            keywords = {k.arg: k.value for k in node.keywords if k.arg}
            provider = keywords.get("policy_provider")
            config = keywords.get("policy_config")
            if isinstance(provider, ast.Constant) and provider.value == "cosmos3" and isinstance(config, ast.Dict):
                configs.append(ast.literal_eval(config))
    return configs


def _sim_receiver(target: str) -> Any:
    """The bound method a ``sim.<name>(...)`` fence call reaches on the MuJoCo engine."""
    engine = create_simulation("mujoco", mesh=False)
    return getattr(type(engine), target.split(".", 1)[1])


class TestTheDocumentedKnobsNameAReachableReceiver:
    """An instruction to pass a keyword must name a receiver that accepts it."""

    def test_every_instructed_keyword_is_a_parameter_of_a_named_receiver(self) -> None:
        page = _page()
        instructed = _instructed_keywords(page)
        assert len(instructed) >= _MINIMUM_INSTRUCTIONS, (
            f"premise: found {len(instructed)} 'pass `kw=`' instructions in {_DOC_NAMES}; "
            "a clean sweep would prove nothing"
        )
        named = _named_receivers(page)
        reachable: set[str] = set()
        for cls in named.values():
            reachable |= _params(cls)
        unreachable = sorted(k for k in instructed if k not in reachable)
        assert not unreachable, (
            f"{_DOC_NAMES} tells the reader to pass {unreachable}, which no receiver it "
            f"names accepts. Named receivers: {sorted(named)}. "
            "Name the class the keyword belongs to, or promote the keyword."
        )

    def test_the_constructor_table_is_on_the_page_and_agrees_with_the_signature(self) -> None:
        """The generated table is what the reader passes; it must match ``Cosmos3Policy``."""
        source = "\n".join(p.read_text(encoding="utf-8") for p in _DOCS)
        assert "{{providers:kwargs:cosmos3}}" in source, f"{_DOC_NAMES} must carry the constructor table token"
        listed = _table_keywords(_page())
        accepted = _params(Cosmos3Policy)
        assert listed, f"the Constructor keywords table on {_DOC_NAMES} rendered no rows"
        assert listed == accepted, (
            f"{_DOC_NAMES} constructor table lists {sorted(listed ^ accepted)} differently from "
            "inspect.signature(Cosmos3Policy); docs/hooks/providers.py read the wrong __init__"
        )

    def test_the_grader_reports_a_planted_instruction(self) -> None:
        """A clean sweep means the page is right, not that the grader accepts anything."""
        planted = _page() + "\nInstall it and pass `not_a_parameter_of_anything=True` to enable it.\n"
        assert "not_a_parameter_of_anything" in _instructed_keywords(planted)

    @pytest.mark.parametrize("keyword", sorted(_instructed_keywords(_page())))
    def test_each_instructed_keyword_resolves_to_exactly_the_class_that_declares_it(self, keyword: str) -> None:
        page = _page()
        owners = sorted(name for name, cls in _named_receivers(page).items() if keyword in _params(cls))
        assert owners, f"{keyword!r} is instructed by {_DOC_NAMES} but declared by no receiver it names"


class TestTheInstructionRoundTrips:
    """Applying the page's instruction to the receiver it names reaches the backend."""

    def test_a_backend_knob_reaches_the_policy_through_the_object_route_the_page_names(self) -> None:
        """The page carries backend knobs on ``diffusers_backend``, not on the JSON route.

        The constructor table names ``diffusers_backend`` typed as
        ``Cosmos3DiffusersBackend``; a knob that class declares and the policy
        does not (``enable_safety_checker``) must reach the running backend
        through that object.
        """
        page = _page()
        assert "diffusers_backend" in _table_keywords(page), f"premise: {_DOC_NAMES} no longer names the object route"
        assert "Cosmos3DiffusersBackend" in page, f"premise: {_DOC_NAMES} no longer names the backend class"
        assert "enable_safety_checker" not in _params(Cosmos3Policy), "the knob moved onto the policy; regrade"
        owners = [cls for cls in _named_receivers(page).values() if "enable_safety_checker" in _params(cls)]
        assert owners, "the page names no receiver that declares enable_safety_checker"

        backend = Cosmos3DiffusersBackend(
            embodiment=get_embodiment("droid"),
            enable_safety_checker=True,
            pipeline=_stub_pipeline(),
            condition_cls=_StubCondition,
        )
        policy = Cosmos3Policy(embodiment="droid", backend="diffusers", diffusers_backend=backend)
        assert policy.last_rollout is None or isinstance(policy.last_rollout, dict)
        reached = getattr(getattr(policy, "_diffusers", None), "enable_safety_checker", None)
        assert reached is True, f"the flag did not reach the backend the policy runs (got {reached!r})"

    def test_any_documented_knob_list_names_only_real_backend_parameters(self) -> None:
        """A list of knobs the object route carries must not name a missing one.

        Silent when the page lists none, so this grades the claim rather than
        requiring one; the planted case below keeps that from being vacuous.
        """
        backend_params = _params(Cosmos3DiffusersBackend)
        listed = _listed_backend_knobs(_page())
        unknown = sorted(listed - backend_params)
        assert not unknown, f"{_DOC_NAMES} lists {unknown} as backend knobs, which Cosmos3DiffusersBackend has not"

        planted = "load and sampling knobs\n> (`resolution_tier`, `not_a_backend_parameter`)"
        assert _listed_backend_knobs(planted) - backend_params == {"not_a_backend_parameter"}


class TestThePolicySurfaceIsUnchanged:
    """Controls: the routes that already worked still answer the same way."""

    def test_every_fenced_keyword_is_accepted_by_its_own_receiver(self) -> None:
        page = _page()
        per_call = _fence_keywords(page)
        graded = 0
        offenders: list[str] = []
        for target, keywords in per_call.items():
            cls: type | None = None
            accepted: set[str]
            if target.startswith("create_policy") or target == "Cosmos3Policy":
                cls = Cosmos3Policy
            elif target in _RECEIVERS:
                cls = _RECEIVERS[target]
            if cls is not None:
                accepted = _params(cls)
            elif target.startswith("sim."):
                accepted = set(inspect.signature(_sim_receiver(target)).parameters) - {"self"}
            elif target == "create_simulation":
                accepted = set(inspect.signature(create_simulation).parameters) | {"mesh"}
            else:
                continue
            graded += len(keywords)
            offenders += [f"{target}({k}=)" for k in sorted(keywords) if k not in accepted]
        assert graded >= _MINIMUM_FENCE_KEYWORDS, f"premise: graded only {graded} fenced keywords in {_DOC_NAMES}"
        assert not offenders, f"{_DOC_NAMES} passes keywords its own receiver refuses: {offenders}"

    def test_every_policy_config_key_survives_the_registry_route(self) -> None:
        """The worked example's ``policy_config`` must not carry a key the route drops."""
        configs = _registry_route_configs(_page())
        assert configs, f'premise: {_DOC_NAMES} has no policy_provider="cosmos3" fence with a policy_config'
        for config in configs:
            built = build_policy_kwargs("cosmos3", **config)
            dropped = sorted(set(config) - set(built))
            assert not dropped, f"{_DOC_NAMES} passes policy_config keys the registry route drops: {dropped}"
            refused = sorted(set(config) - _params(Cosmos3Policy))
            assert not refused, f"{_DOC_NAMES} passes policy_config keys Cosmos3Policy refuses: {refused}"

    def test_the_forwarded_subset_still_reaches_the_backend(self) -> None:
        """``model`` and ``mode`` are the knobs the policy itself carries."""
        backend = Cosmos3DiffusersBackend(
            embodiment=get_embodiment("droid"),
            model="nvidia/Cosmos3-Nano",
            mode="inverse_dynamics",
            pipeline=_stub_pipeline(),
            condition_cls=_StubCondition,
        )
        policy = Cosmos3Policy(embodiment="droid", backend="diffusers", diffusers_backend=backend)
        assert policy._diffusers is not None
        assert policy._diffusers.model == "nvidia/Cosmos3-Nano"
        assert policy._diffusers.mode == "inverse_dynamics"

    def test_the_registry_route_carries_only_the_declared_config_keys(self) -> None:
        """Why the page names an object route: the JSON route filters the flag out."""
        built = build_policy_kwargs("cosmos3", backend="diffusers", mode="policy", enable_safety_checker=True)
        assert built["backend"] == "diffusers"
        assert built["mode"] == "policy"
        assert "enable_safety_checker" not in built


class _StubCondition:
    """Stand-in for ``diffusers.CosmosActionCondition`` (records its kwargs)."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _stub_pipeline() -> Any:
    """A ``Cosmos3OmniPipeline``-shaped callable returning one action chunk."""
    embodiment = get_embodiment("droid")
    chunk = np.zeros((embodiment.action_chunk_size, embodiment.raw_action_dim), dtype=np.float32)

    class _Pipeline:
        def __call__(self, **kwargs: Any) -> Any:
            import types

            return types.SimpleNamespace(action=[chunk], video=None, sound=None)

    return _Pipeline()
