"""Every constructor keyword a caller can set from JSON is declared in ``config_keys``.

``policies.json`` documents ``config_keys`` as "the keywords the provider
UNDERSTANDS" (``registry/policies.py``, ``hardware_robot.py``), and three
consumers act on that reading: ``build_policy_kwargs`` drops any key outside it,
the dashboard renders the policy form from it (``dashboard/policy_fit.py``,
``dashboard/config_api.py``) and ``validate_scope`` reads it for the preflight's
reach. ``tests/registry/test_config_keys_agree_with_constructors.py`` pins one
direction - every declared key names a constructor parameter. Nothing pinned the
other, and the policy-matrix audit of 2026-09-28 measured the drift: 13
``LerobotLocalPolicy`` keywords (``rtc_enabled``, ``camera_key_map``,
``obs_rename_override``, ``revision``, ...), 10 ``Gr00tPolicy`` keywords and
``Cosmos3Policy(pretrained_name_or_path=)`` were bindable by the constructor,
documented on the provider's page (the ``{{providers:kwargs}}`` table is read
from the same ``__init__``), and absent from the registry - so the dashboard
could not offer them and ``build_policy_kwargs`` silently dropped them.

The guard is the reverse inclusion with an explicit, reasoned exclusion list:
a keyword that takes an injected OBJECT (a pre-built client, a live session, a
planner instance) cannot travel as JSON and does not belong in a form, and a
GR00T local-mode keyword is being removed in 0.7 (RULING R-P1), so advertising
it would point the form at a path that is going away. Anything else a
constructor binds by name is a key the registry must know.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICIES_JSON = REPO_ROOT / "strands_robots" / "registry" / "policies.json"

#: Constructor keywords deliberately NOT advertised as config keys, with the reason.
EXCLUDED: dict[str, dict[str, str]] = {
    "cosmos3": {
        "client": "an injected Cosmos3WebsocketClient object (tests), not JSON",
        "transport": "deprecated compatibility knob; the only transport is 'raw'",
        "diffusers_backend": "an injected Cosmos3DiffusersBackend object (tests), not JSON",
    },
    "groot": {
        "model_path": "GR00T local mode, removed in 0.7 (R-P1); use lerobot_local policy_type='groot'",
        "embodiment_tag": "GR00T local mode, removed in 0.7 (R-P1)",
        "device": "GR00T local mode, removed in 0.7 (R-P1)",
    },
    "curobo": {
        "motion_gen": "an injected cuRobo MotionGen instance, not JSON",
    },
    "wbc_latent": {
        "decoder": "a prebuilt SonicDecoder object (tests, shared sessions), not JSON",
        "session": "an injected decoder session object, not JSON",
    },
}


def _providers() -> dict[str, dict]:
    return json.loads(POLICIES_JSON.read_text(encoding="utf-8"))["providers"]


def _constructor_keywords(spec: dict) -> list[str] | None:
    try:
        module = importlib.import_module(spec["module"])
    except ImportError:
        return None
    cls = getattr(module, spec["class"])
    params = list(inspect.signature(cls.__init__).parameters.values())[1:]
    return [p.name for p in params if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]


@pytest.mark.parametrize("provider", sorted(_providers()))
def test_every_bindable_keyword_is_declared_or_excluded_with_a_reason(provider: str) -> None:
    spec = _providers()[provider]
    keywords = _constructor_keywords(spec)
    if keywords is None:
        pytest.skip(f"{provider}: optional dependency not installed")
    declared = set(spec.get("config_keys") or [])
    excluded = set(EXCLUDED.get(provider, {}))
    undeclared = [k for k in keywords if k not in declared and k not in excluded]
    assert not undeclared, (
        f"{provider}.__init__ binds {undeclared} but policies.json config_keys does not declare them: "
        "the dashboard form cannot offer them and build_policy_kwargs drops them. Add them to config_keys, "
        "or list them in EXCLUDED here with the reason they cannot travel as JSON."
    )


@pytest.mark.parametrize("provider", sorted(EXCLUDED))
def test_every_exclusion_still_names_a_constructor_keyword(provider: str) -> None:
    # A stale exclusion would hide a keyword the constructor no longer binds, or
    # one the registry has since started declaring.
    spec = _providers()[provider]
    keywords = _constructor_keywords(spec)
    if keywords is None:
        pytest.skip(f"{provider}: optional dependency not installed")
    for key in EXCLUDED[provider]:
        assert key in keywords, f"EXCLUDED lists {provider}.{key}, which the constructor no longer binds"
        assert key not in (spec.get("config_keys") or []), f"{provider}.{key} is both declared and excluded"
