"""Training job management for the dashboard - thin REST shim over train_policy."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from strands_robots.dashboard.dataset_check import dataset_verdict
from strands_robots.dashboard.ttl_cache import TTLCache

logger = logging.getLogger(__name__)

JOBS_FILE = Path(os.getenv(
    "DASHBOARD_JOBS_FILE",
    os.path.join(tempfile.gettempdir(), "strands_dashboard", "train_jobs.json"),
))
# : Dataset roots the dashboard itself created (collect wizard).
ROOTS_FILE = JOBS_FILE.parent / "dataset_roots.json"
_LOCK = threading.Lock()

# : Set when a ledger file existed but could not be read.
_JOBS_PROBLEM: str | None = None

def _write_json_durably(path: Path, payload: Any) -> None:
    """Write JSON so a crash mid-write cannot destroy what was already there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        # a failed serialization must not leave litter next to the real ledger
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()

def _quarantine(path: Path) -> Path:
    """Move an unreadable file aside, preserving it, and return where it went."""
    kept = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
    os.replace(path, kept)
    return kept

def remember_dataset_root(root: str) -> None:
    """Persist a dataset root so local_datasets() discovers it forever."""
    try:
        with _LOCK:
            roots: list[str] = []
            if ROOTS_FILE.exists():
                data = json.loads(ROOTS_FILE.read_text())
                if isinstance(data, list):
                    roots = data
            if root not in roots:
                roots.append(root)
                _write_json_durably(ROOTS_FILE, roots[-50:])
    except Exception as e:  # noqa: BLE001
        logger.debug("could not remember dataset root: %s", e)

def _remembered_roots() -> list[Path]:
    try:
        if ROOTS_FILE.exists():
            data = json.loads(ROOTS_FILE.read_text())
            if isinstance(data, list):
                return [Path(r) for r in data]
    except Exception:  # noqa: BLE001
        pass
    return []

def _load_jobs() -> list[dict[str, Any]]:
    """The job ledger, or an empty list - with ``_JOBS_PROBLEM`` set if it was lost."""
    global _JOBS_PROBLEM
    if not JOBS_FILE.exists():
        _JOBS_PROBLEM = None
        return []
    try:
        data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - any unreadable ledger, not just bad JSON
        logger.warning("could not load train jobs: %s", e)
        detail = f"{type(e).__name__}: {e}"
        try:
            kept = _quarantine(JOBS_FILE)
            _JOBS_PROBLEM = (
                f"the training job history could not be read ({detail}) and was moved to "
                f"{kept} - runs started before now have no card here, but any that are still "
                f"running are unaffected"
            )
        except OSError as move_err:
            _JOBS_PROBLEM = (
                f"the training job history could not be read ({detail}) and could not be moved "
                f"aside ({move_err}); refusing to overwrite it - runs started before now have "
                f"no card here"
            )
        return []
    if not isinstance(data, list):
        _JOBS_PROBLEM = f"the training job history is a {type(data).__name__}, not a list of jobs"
        return []
    _JOBS_PROBLEM = None
    return data

def _save_jobs(jobs: list[dict[str, Any]]) -> None:
    if _JOBS_PROBLEM and JOBS_FILE.exists():
        # the ledger is there but unreadable AND could not be quarantined; writing
        # now would replace runs we cannot see with the one we just started
        logger.warning("not overwriting an unreadable job ledger: %s", _JOBS_PROBLEM)
        return
    try:
        _write_json_durably(JOBS_FILE, jobs[-100:])
    except Exception as e:  # noqa: BLE001
        logger.warning("could not save train jobs: %s", e)

def jobs_problem() -> str | None:
    """Why the job list may be missing entries, or None when it is trustworthy."""
    with _LOCK:
        return _JOBS_PROBLEM

def _tool_result(res: dict[str, Any]) -> dict[str, Any]:
    """Flatten a train_policy tool result: status + text + json block."""
    out: dict[str, Any] = {"status": res.get("status", "unknown"), "text": "", "data": {}}
    for block in res.get("content") or []:
        if isinstance(block, dict):
            if "text" in block and not out["text"]:
                out["text"] = block["text"]
            if isinstance(block.get("json"), dict):
                out["data"].update(block["json"])
    return out

def list_trainers() -> list[str]:
    from strands_robots.training import list_trainers as _lt

    return list(_lt())

# : Trainer modules whose spec this form cannot build.
_RL_TRAINER_MODULE_PREFIX = "strands_robots.training.rl"

# : Phrased WITHOUT a leading article and in the plural on purpose: the form joins these :
# reasons after a list of provider names, and "fast_sac and ppo are a reinforcement-learning :
# trainer" is the sentence that comes out of the obvious singular wording.
_RL_REASON = (
    "reinforcement-learning trainers learn from a live environment, not from a recorded "
    "dataset, so they are driven from a script (RLTrainSpec) rather than this form"
)

def _declared_trainer_module(provider: str) -> str:
    """Where a provider's trainer class comes from, WITHOUT importing it."""
    from strands_robots.registry.policies import get_policy_provider

    cfg = get_policy_provider(provider) or {}
    module = str((cfg.get("trainer") or {}).get("module") or "")
    if module:
        return module
    from strands_robots.training import factory

    loader = factory._runtime_registry.get(provider)
    code = getattr(loader, "__code__", None)
    names = tuple(getattr(code, "co_names", ()) or ()) + tuple(getattr(code, "co_consts", ()) or ())
    for name in names:
        if isinstance(name, str) and name.startswith("strands_robots."):
            return name
    return ""

_FORM_CANNOT_EXPRESS: dict[str, str] = {
    "sagemaker": (
        "submits the job to AWS, so it needs an ECR image_uri, an execution role_arn, and s3:// "
        "paths for the dataset and the output - this form has no field for any of them, and a "
        "managed job cannot mount this machine's disk. Run the sagemaker trainer from a script"
    ),
    "cosmos3": (
        "needs a training recipe TOML (extra['sft_toml']) that selects the registered "
        "experiment, and this form has no field for it - run cosmos3 from a script"
    ),
}

def form_unsupported() -> dict[str, str]:
    """Providers that cannot be trained FROM THIS FORM, mapped to why."""
    out: dict[str, str] = {}
    for provider in list_trainers():
        try:
            module = _declared_trainer_module(provider)
        except Exception:  # a malformed registry entry must not blank the whole form
            continue
        if module.startswith(_RL_TRAINER_MODULE_PREFIX):
            out[provider] = _RL_REASON
        elif provider in _FORM_CANNOT_EXPRESS:
            out[provider] = _FORM_CANNOT_EXPRESS[provider]
    return out

# : The full set of fields a training spec accepts. submit() and validate() : share it so the
# two can never drift: a field the form sends either reaches : train_policy or is refused BY
# NAME - silently dropping a typo'd "step" : would train 10k default steps and call it
# success.
SPEC_KEYS = (
    "provider", "dataset_root", "dataset_repo_id", "base_model",
    "output_dir", "embodiment", "steps", "batch_size", "learning_rate",
    "save_freq", "method", "lora_r", "lora_alpha", "seed", "val_episodes",
)

# : train_policy parameters this form deliberately does NOT send, each with the reason.
_NOT_IN_FORM: dict[str, str] = {
    "action": "the verb, chosen by the endpoint rather than the operator",
    "job_id": "assigned by the job store; a client-chosen id could collide with a running job",
    "streaming": "mutually exclusive with val_episodes on the lerobot backend, and it is the "
                 "wrong default for a first dataset -- a Hub stream cannot be resumed offline",
    "num_gpus": "this dashboard drives ONE host; a multi-GPU/multi-node launch is a cluster "
                "decision, and torch elastic rendezvous on this Mac needed a fix (Q37) before "
                "it would even start",
    "num_nodes": "same: multi-node belongs to a script that knows the cluster's addresses",
    "resume": "resuming needs the previous run's output_dir to still hold its checkpoints; "
              "offering a tick box that silently starts fresh would be worse than not offering it",
    "lora_target_modules": "a list of model-internal module names -- unanswerable without the "
                           "architecture in front of you",
    "tune": "a per-backend dict of component toggles (GR00T llm/visual/projector/diffusion)",
    "augmentation": "a per-backend dict of augmentation parameters",
    "fps": "read from the dataset's own metadata; a mismatching value silently retimes the data",
    "extra": "the escape hatch for backend-specific kwargs, which is what a script is for",
}

def _spec_kwargs(body: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(kwargs, None) for a clean body, (None, error-result) for a bad one."""
    unknown = sorted(k for k in body if k not in SPEC_KEYS and k != "action")
    if unknown:
        return None, {
            "status": "error",
            "data": {},
            "text": (
                "unknown field(s): " + ", ".join(unknown)
                + ". Valid fields: " + ", ".join(SPEC_KEYS)
            ),
        }
    return {k: body[k] for k in SPEC_KEYS if body.get(k) is not None}, None

def _has_data_files(dataset_dir: Path) -> bool | None:
    """Does ``data/`` hold at least one file? ``None`` when it cannot be looked at."""
    data = dataset_dir / "data"
    if not data.is_dir():
        return False
    try:
        for _, _, files in os.walk(data):
            if files:
                return True
        return False
    except OSError:
        return None

def local_datasets(query: str = "") -> list[dict[str, Any]]:
    """LeRobotDataset roots on disk (meta/info.json present)."""
    roots: list[Path] = []
    home = os.getenv("HF_LEROBOT_HOME") or str(Path.home() / ".cache" / "huggingface" / "lerobot")
    roots.append(Path(home))
    for extra in (os.getenv("STRANDS_ROBOTS_DATA_DIRS") or "").split(":"):
        if extra.strip():
            roots.append(Path(extra.strip()).expanduser())
    # Collect-wizard roots: each remembered path is a dataset dir itself, so its PARENT enters the
    # scan (the walker finds meta/info.json one level down, and siblings collected next to it come
    # along for free).
    for r in _remembered_roots():
        if r.parent.is_dir() and r.parent not in roots:
            roots.append(r.parent)

    q = query.lower()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        # depth-2 walk: root/<repo> and root/<org>/<repo>
        try:
            candidates = list(root.iterdir())
        except OSError:
            continue
        stack = [(c, 1) for c in candidates if c.is_dir()]
        while stack:
            d, depth = stack.pop()
            info = d / "meta" / "info.json"
            if info.exists():
                rel = str(d.relative_to(root)) if d.is_relative_to(root) else str(d)
                if str(d) in seen or (q and q not in rel.lower()):
                    continue
                seen.add(str(d))
                meta: dict[str, Any] = {}
                try:
                    raw = json.loads(info.read_text())
                    meta = {
                        "total_episodes": raw.get("total_episodes"),
                        "total_frames": raw.get("total_frames"),
                        "fps": raw.get("fps"),
                        "robot_type": raw.get("robot_type"),
                    }
                except Exception:  # noqa: BLE001
                    pass
                verdict = dataset_verdict(meta, has_data_files=_has_data_files(d))
                out.append({"root": str(d), "repo_id": rel, **meta, **verdict})
            elif depth < 2:
                try:
                    stack.extend((c, depth + 1) for c in d.iterdir() if c.is_dir())
                except OSError:
                    pass
    return sorted(out, key=lambda r: r["repo_id"])[:50]

_HUB_DS_TTL_S = 300.0
#: Bounded and self-pruning - see ttl_cache: the old dict kept every prefix ever typed.
_HUB_DS_CACHE: TTLCache[list[dict[str, Any]]] = TTLCache(_HUB_DS_TTL_S)

def hub_datasets(query: str = "", limit: int = 12) -> tuple[list[dict[str, Any]], str | None]:
    """Type-ahead search of public LeRobot datasets on the Hub."""
    key = f"{query}:{limit}"
    cached = _HUB_DS_CACHE.get(key)
    if cached is not None:
        return cached, None
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            found = api.list_datasets(filter="LeRobot", search=query or None, sort="downloads", limit=limit)
        except TypeError:
            found = api.list_datasets(filter="LeRobot", search=query or None, limit=limit)
        rows = [
            {
                "repo_id": d.id,
                "local": False,
                "downloads": getattr(d, "downloads", None),
                "tags": [t for t in (getattr(d, "tags", None) or []) if not t.startswith(("region:", "license:", "arxiv:"))][:6],
            }
            for d in found
        ]
    except Exception as exc:  # noqa: BLE001 - hub outage degrades to local-only
        logger.warning("hub dataset search failed: %r", exc)
        kind = type(exc).__name__
        return [], f"Hub search unavailable ({kind}) - showing local datasets only"
    _HUB_DS_CACHE.put(key, rows)
    return rows, None

def search_datasets(query: str = "", limit: int = 12) -> dict[str, Any]:
    """Local dataset roots merged with a Hub search, local first."""
    limit = max(1, int(limit))
    local = [dict(r, local=True) for r in local_datasets(query)]
    hub, problem = hub_datasets(query, limit)
    have = {r.get("repo_id") for r in local}
    merged = local + [r for r in hub if r.get("repo_id") not in have]
    rows = merged[:limit]
    return {
        "query": query,
        "datasets": rows,
        "total_matched": len(merged),
        "problem": problem,
        "hub_count": len([r for r in rows if not r.get("local")]),
        "local_count": len([r for r in rows if r.get("local")]),
        "hf_auth": _hf_auth_state(),
    }

def _hf_auth_state() -> dict[str, Any]:
    """Who the Hub thinks we are - reused from checkpoints, not re-implemented."""
    try:
        from strands_robots.dashboard import checkpoints

        return checkpoints.hf_auth_state()
    except Exception as exc:  # noqa: BLE001 - never let an auth probe break search
        return {"authenticated": False, "user": None, "detail": f"auth state unavailable ({type(exc).__name__})"}

def output_dir_verdict(path: str) -> dict[str, Any]:
    from strands_robots.dashboard.output_dir_check import (
        default_checkpoint_probe,
        inspect_output_dir,
    )

    return inspect_output_dir(path, has_checkpoint=default_checkpoint_probe)

def submit(body: dict[str, Any]) -> dict[str, Any]:
    """Validate + launch a training job; persist it for the tab."""
    confirm_clear = bool(body.pop("confirm_clear", False))
    kwargs, err = _spec_kwargs(body)
    if err is not None:
        return err
    if not confirm_clear and kwargs and kwargs.get("output_dir"):
        verdict = output_dir_verdict(str(kwargs["output_dir"]))
        if verdict.get("needs_confirm"):
            return {
                "status": "error",
                "data": {"output_dir_verdict": verdict, "needs_confirm": True},
                "text": (
                    f"{verdict['path']} {verdict['detail']}. Send confirm_clear to train here "
                    "anyway, or point output_dir somewhere new"
                ),
            }
    from strands_robots.tools.train_policy import train_policy

    res = _tool_result(train_policy(action="train", **kwargs))
    if res["status"] == "success":
        job = {
            "job_id": res["data"].get("job_id"),
            "provider": kwargs.get("provider", "lerobot_local"),
            "dataset": kwargs.get("dataset_root") or kwargs.get("dataset_repo_id"),
            "base_model": kwargs.get("base_model"),
            "output_dir": kwargs.get("output_dir"),
            "steps": kwargs.get("steps"),
            "submitted_at": time.time(),
        }
        with _LOCK:
            jobs = _load_jobs()
            jobs.append(job)
            _save_jobs(jobs)
        res["job"] = job
    return res

def validate(body: dict[str, Any]) -> dict[str, Any]:
    kwargs, err = _spec_kwargs(body)
    if err is not None:
        return err
    from strands_robots.tools.train_policy import train_policy

    return _tool_result(train_policy(action="validate", **kwargs))

def status(provider: str, job_id: str) -> dict[str, Any]:
    from strands_robots.tools.train_policy import train_policy

    return _tool_result(train_policy(action="status", provider=provider, job_id=job_id))

def export(
    provider: str,
    output_dir: str,
    dataset_root: str = "",
    dataset_repo_id: str | None = None,
    base_model: str = "",
) -> dict[str, Any]:
    """Export the run's last checkpoint as a loadable artifact."""
    from strands_robots.dashboard.artifact_check import artifact_verdict
    from strands_robots.tools.train_policy import train_policy

    res = _tool_result(train_policy(
        action="export", provider=provider, output_dir=output_dir,
        dataset_root=dataset_root or None, dataset_repo_id=dataset_repo_id,
        base_model=base_model or "",
    ))
    exported = res.get("data", {}).get("exported_model")
    if res["status"] == "success":
        res["artifact"] = artifact_verdict(exported)
        # One boolean for the caller that only decides "may this be handed to an arm".
        res["deployable"] = bool(res["artifact"].get("ok"))
    return res

def jobs() -> list[dict[str, Any]]:
    with _LOCK:
        return _load_jobs()
