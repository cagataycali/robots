"""Policy checkpoint discovery for the run form.

Three sources, merged and ranked:

1. **Local HF cache** (``~/.cache/huggingface/hub``) - instant, offline,
   already-downloaded checkpoints run without a fetch. Marked ``local=True``.
2. **HuggingFace Hub search** (``filter="lerobot"``) - type-ahead search of
   every public LeRobot-format checkpoint (smolvla/act/pi0/diffusion/...).
   Cached per query for 5 minutes; hub outages degrade to local-only.
3. **Policy families** (``list_policy_types``) - the ``policy_type`` values
   lerobot_local/lerobot_async accept, used by the form's type dropdown.

The run form calls ``/api/checkpoints/search?q=...`` as the user types and
gets back ``{repo_id, downloads, local, policy_type?, tags}`` rows ready for
``pretrained_name_or_path``.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_CACHE_TTL_S = 300.0
_CACHE_LOCK = threading.Lock()

# lerobot policy family names that can appear in checkpoint tags/names.
_FAMILY_RE = re.compile(
    r"\b(smolvla|act|diffusion|pi0[-_]?fast|pi05|pi0|tdmpc|vqbet|groot|molmoact2?|sac|xvla|wall_x|eo1|evo1)\b",
    re.IGNORECASE,
)


def _guess_policy_type(repo_id: str, tags: list[str]) -> str | None:
    """Best-effort policy family from repo name + tags (form prefill only)."""
    for source in (*tags, repo_id):
        # underscores are word chars, so \bsmolvla\b misses 'smolvla_base' -
        # normalize separators before matching.
        m = _FAMILY_RE.search((source or "").replace("_", "-"))
        if m:
            return m.group(1).lower().replace("-", "_").replace("pi0fast", "pi0_fast")
    return None


def local_checkpoints(query: str = "") -> list[dict[str, Any]]:
    """Already-downloaded LeRobot-shaped checkpoints in the HF cache.

    A checkpoint is 'lerobot-shaped' when its snapshot carries a
    ``config.json`` with a ``type`` field or a ``train_config.json``
    (LeRobot v0.5+ layout). Cheap heuristic - we only stat a couple of
    files per cached model, never load weights.
    """
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.is_dir():
        return []
    q = query.lower()
    out: list[dict[str, Any]] = []
    for entry in hub.iterdir():
        if not entry.name.startswith("models--"):
            continue
        repo_id = entry.name[len("models--"):].replace("--", "/", 1)
        if q and q not in repo_id.lower():
            continue
        snapshots = entry / "snapshots"
        if not snapshots.is_dir():
            continue
        try:
            snap = next(iter(sorted(snapshots.iterdir())), None)
        except OSError:
            continue
        if snap is None:
            continue
        cfg = snap / "config.json"
        train_cfg = snap / "train_config.json"
        if not (cfg.exists() or train_cfg.exists()):
            continue
        policy_type = None
        try:
            if cfg.exists():
                import json

                data = json.loads(cfg.read_text())
                # lerobot policy configs carry "type"; transformers configs
                # carry "model_type" - only the former is runnable here.
                policy_type = data.get("type")
                if policy_type is None and "model_type" in data:
                    continue  # transformers model, not a policy checkpoint
        except Exception:  # noqa: BLE001 - unreadable config -> still list it
            pass
        out.append({
            "repo_id": repo_id,
            "local": True,
            "downloads": None,
            "policy_type": policy_type or _guess_policy_type(repo_id, []),
            "tags": [],
        })
    return out


def hub_search(query: str, limit: int = 12) -> tuple[list[dict[str, Any]], str | None]:
    """Type-ahead search of public LeRobot checkpoints on the Hub.

    Returns ``(rows, problem)`` - ``problem`` is None when the Hub answered,
    otherwise a short human sentence. The old behavior (log at DEBUG, return
    []) made three very different worlds render identically as "no results":
    hub outage, no network, and a fine query with no matches. The operator
    typing into the search box is exactly the person who needs to know which
    world they are in.
    """
    key = f"{query}:{limit}"
    now = time.time()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < _CACHE_TTL_S:
            return hit[1], None
    try:
        from huggingface_hub import HfApi

        # newer huggingface_hub removed direction=; sort="downloads" alone
        # returns descending. Fall back to unsorted on any TypeError so a
        # hub-client version bump can't kill search.
        api = HfApi()
        try:
            models = api.list_models(filter="lerobot", search=query or None, sort="downloads", limit=limit)
        except TypeError:
            models = api.list_models(filter="lerobot", search=query or None, limit=limit)
        rows = [
            {
                "repo_id": m.id,
                "local": False,
                "downloads": getattr(m, "downloads", None),
                "policy_type": _guess_policy_type(m.id, list(m.tags or [])),
                "tags": [t for t in (m.tags or []) if not t.startswith(("region:", "license:", "arxiv:"))][:6],
            }
            for m in models
        ]
    except Exception as exc:  # noqa: BLE001 - hub outage degrades to local-only
        logger.warning("hub checkpoint search failed: %r", exc)
        # do NOT cache a failure - the next keystroke should retry
        kind = type(exc).__name__
        return [], f"Hub search unavailable ({kind}) - showing local cache only"
    with _CACHE_LOCK:
        _CACHE[key] = (now, rows)
    return rows, None


_WHOAMI: dict[str, Any] = {"at": 0.0, "value": None}
_WHOAMI_TTL_S = 600.0


def hf_auth_state() -> dict[str, Any]:
    """Whether this machine can reach gated/private HF repos, and as whom.

    Token discovery is local + instant (env or ~/.cache/huggingface/token).
    whoami() is a network call, so its answer is cached 10 minutes; a token
    that fails whoami is reported as invalid rather than silently treated
    like anonymity - a revoked token behaves differently from no token
    (401 vs public-only), and the UI should say which one the user has.
    """
    try:
        from huggingface_hub import get_token
        token = get_token()
    except Exception:  # noqa: BLE001
        token = None
    if not token:
        return {"authenticated": False, "user": None, "detail": "no HF token on this machine"}
    now = time.time()
    if now - _WHOAMI["at"] < _WHOAMI_TTL_S and _WHOAMI["value"] is not None:
        return _WHOAMI["value"]
    try:
        from huggingface_hub import HfApi
        user = HfApi().whoami(token=token).get("name")
        value = {"authenticated": True, "user": user, "detail": None}
    except Exception as exc:  # noqa: BLE001
        value = {
            "authenticated": False,
            "user": None,
            "detail": f"HF token present but rejected ({type(exc).__name__})",
        }
    _WHOAMI.update(at=now, value=value)
    return value


#: Widest page the hub search is asked for, and the ceiling for any caller.
MAX_LIMIT = 40


def clamp_limit(limit: Any, default: int = 15, ceiling: int = MAX_LIMIT) -> int:
    """A limit is a promise to the caller: 1 means one row.

    ``rows[: max(limit, len(local))]`` used to keep every local cache row no
    matter what was asked for, so a type-ahead requesting 1 got 16 (and 0 or -5
    got 16 too). The no-hidden-local-rows intent survives without overriding the
    caller, because local rows are ORDERED FIRST -- truncating drops hub rows
    before local ones.
    """
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, ceiling))


def search(query: str = "", limit: int = 15) -> dict[str, Any]:
    """Merged checkpoint search: local cache first, then hub by downloads."""
    limit = clamp_limit(limit)
    local = local_checkpoints(query)
    local_ids = {r["repo_id"] for r in local}
    remote_rows, hub_problem = hub_search(query, limit=limit)
    remote = [r for r in remote_rows if r["repo_id"] not in local_ids]
    rows = local + remote
    return {
        "query": query,
        "results": rows[:limit],
        "total_matched": len(rows),
        "hub_problem": hub_problem,
        "hf_auth": hf_auth_state(),
    }


def policy_families() -> list[str]:
    """``policy_type`` values lerobot_local accepts (for the type dropdown)."""
    try:
        from strands_robots.policies.lerobot_local.resolution import list_policy_types

        return list(list_policy_types())
    except Exception:  # noqa: BLE001 - torch-less install
        return ["act", "diffusion", "pi0", "pi0_fast", "smolvla", "tdmpc", "vqbet"]
