"""What a policy preflight ACTUALLY verified.

``POST /api/policies/validate`` answered ``{"ok": true, "stage": "preflight"}``
for ``lerobot_local`` with an EMPTY config, and the run form rendered that as a
a green "lerobot_local resolves". Nothing about a model had been resolved: the
provider declares no required fields (``requires: []``,
``pretrained_name_or_path`` required *false*), ``policy_provider_error`` has
nothing to complain about, and ``preflight_policy``'s useful check - comparing a
model's declared image inputs against the observation keys it will receive -
needs a model to ask. With no checkpoint there is no model, so the pass is a pass
of an empty question.

An "ok" that means "no objection could be raised" must not be worded like "I
checked the thing you are about to run on a real arm". This module decides which
of the two a given (provider, config) earned, from the registry's own field
metadata - so a provider added later is covered without touching this file.
"""

from __future__ import annotations

from typing import Any

# Keys that name WHICH model runs. A preflight that never saw one of these did
# not look at a policy, whatever its verdict says.
_IDENTITY_HINTS = ("pretrained", "checkpoint", "model_path", "ckpt", "weights")
# Keys that name a REMOTE that holds the model - a preflight can only confirm
# these are set, never that the far end has anything.
_REMOTE_HINTS = ("host", "port", "server_address", "url", "endpoint")


def _is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _keys_of(spec: Any) -> list[str]:
    """Config keys a provider understands, from either shape the registry uses."""
    if not isinstance(spec, dict):
        return []
    keys: list[str] = []
    for field in spec.get("wire_fields") or []:
        if isinstance(field, dict) and field.get("key"):
            keys.append(str(field["key"]))
    for key in spec.get("config_keys") or []:
        if str(key) not in keys:
            keys.append(str(key))
    return keys


def _matches(key: str, hints: tuple[str, ...]) -> bool:
    low = key.lower()
    return any(h in low for h in hints)


def validation_scope(spec: Any, config: dict[str, Any] | None) -> dict[str, Any]:
    """Describe the preflight's reach for one provider + config.

    Returns ``resolved`` (was a model identity actually named and therefore
    inspectable), ``identity_keys`` (the keys that would have named one) and a
    ``scope_note`` in plain words, or ``None`` for the note when the verdict
    genuinely covers what the operator is about to run.
    """
    cfg = config if isinstance(config, dict) else {}
    keys = _keys_of(spec)
    identity_keys = [k for k in keys if _matches(k, _IDENTITY_HINTS)]
    remote_keys = [k for k in keys if _matches(k, _REMOTE_HINTS)]

    named = [k for k in identity_keys if _is_set(cfg.get(k))]
    if named:
        return {"resolved": True, "identity_keys": identity_keys, "scope_note": None}

    if not identity_keys:
        # Nothing in this provider names a model locally (mock, a whole-body
        # controller, a remote server). Saying "resolves" is fine, but a remote
        # provider's verdict still stops at this machine's edge.
        if remote_keys and any(_is_set(cfg.get(k)) for k in remote_keys):
            return {
                "resolved": True,
                "identity_keys": [],
                "scope_note": (
                    "the address is set, but nothing here can confirm the server at "
                    "the other end has a policy loaded"
                ),
            }
        return {"resolved": True, "identity_keys": [], "scope_note": None}

    hint = " or ".join(identity_keys[:2])
    return {
        "resolved": False,
        "identity_keys": identity_keys,
        "scope_note": (
            f"no model was named ({hint} is empty), so nothing about a policy was "
            "checked - this only confirms the provider exists and that no setting "
            "here is contradictory"
        ),
    }
