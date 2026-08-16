"""Training job management for the dashboard - thin REST shim over train_policy.

``train_policy`` is the one workflow tool in strands_robots that returns
structured JSON blocks (job_id, status, metrics), so the Training tab needs
no prose parsing. This module adds the two things a UI needs on top:

* **job persistence** - submitted jobs (provider, spec summary, job_id) are
  remembered across dashboard restarts so the tab repopulates.
* **dataset discovery** - local LeRobotDataset roots (``meta/info.json``)
  scanned from the default cache + STRANDS_ROBOTS_DATA_DIRS, so the submit
  form's dataset picker offers real paths instead of a bare text input.

Training runs in-process on this host (tools are in-process by design -
mesh is the control plane, workflows co-locate; see tiny-notes SOURCE-MAP
§6). GPU-heavy providers should run the dashboard on the GPU box.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JOBS_FILE = Path(os.getenv(
    "DASHBOARD_JOBS_FILE",
    os.path.join(tempfile.gettempdir(), "strands_dashboard", "train_jobs.json"),
))
#: Dataset roots the dashboard itself created (collect wizard). The default
#: scanner only walks HF_LEROBOT_HOME + STRANDS_ROBOTS_DATA_DIRS, so a
#: collect into /tmp/... would produce a dataset the Training tab cannot see.
ROOTS_FILE = JOBS_FILE.parent / "dataset_roots.json"
_LOCK = threading.Lock()


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
                ROOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
                ROOTS_FILE.write_text(json.dumps(roots[-50:]))
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
    try:
        if JOBS_FILE.exists():
            data = json.loads(JOBS_FILE.read_text())
            if isinstance(data, list):
                return data
    except Exception as e:  # noqa: BLE001
        logger.warning("could not load train jobs: %s", e)
    return []


def _save_jobs(jobs: list[dict[str, Any]]) -> None:
    try:
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(json.dumps(jobs[-100:], default=str))
    except Exception as e:  # noqa: BLE001
        logger.debug("could not save train jobs: %s", e)


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


def local_datasets(query: str = "") -> list[dict[str, Any]]:
    """LeRobotDataset roots on disk (meta/info.json present).

    Scans ``$HF_LEROBOT_HOME`` (default ``~/.cache/huggingface/lerobot``)
    plus any colon-separated ``STRANDS_ROBOTS_DATA_DIRS`` entries, two
    levels deep (org/repo layout).
    """
    roots: list[Path] = []
    home = os.getenv("HF_LEROBOT_HOME") or str(Path.home() / ".cache" / "huggingface" / "lerobot")
    roots.append(Path(home))
    for extra in (os.getenv("STRANDS_ROBOTS_DATA_DIRS") or "").split(":"):
        if extra.strip():
            roots.append(Path(extra.strip()).expanduser())
    # Collect-wizard roots: each remembered path is a dataset dir itself, so
    # its PARENT enters the scan (the walker finds meta/info.json one level
    # down, and siblings collected next to it come along for free).
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
                out.append({"root": str(d), "repo_id": rel, **meta})
            elif depth < 2:
                try:
                    stack.extend((c, depth + 1) for c in d.iterdir() if c.is_dir())
                except OSError:
                    pass
    return sorted(out, key=lambda r: r["repo_id"])[:50]


def submit(body: dict[str, Any]) -> dict[str, Any]:
    """Validate + launch a training job; persist it for the tab."""
    from strands_robots.tools.train_policy import train_policy

    kwargs = {
        k: body[k]
        for k in (
            "provider", "dataset_root", "dataset_repo_id", "base_model",
            "output_dir", "embodiment", "steps", "batch_size", "learning_rate",
            "save_freq", "method", "lora_r", "lora_alpha", "seed",
        )
        if body.get(k) is not None
    }
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
    from strands_robots.tools.train_policy import train_policy

    kwargs = {k: v for k, v in body.items() if v is not None and k != "action"}
    return _tool_result(train_policy(action="validate", **kwargs))


def status(provider: str, job_id: str) -> dict[str, Any]:
    from strands_robots.tools.train_policy import train_policy

    return _tool_result(train_policy(action="status", provider=provider, job_id=job_id))


def export(provider: str, output_dir: str, dataset_root: str = "", dataset_repo_id: str | None = None) -> dict[str, Any]:
    from strands_robots.tools.train_policy import train_policy

    return _tool_result(train_policy(
        action="export", provider=provider, output_dir=output_dir,
        dataset_root=dataset_root or None, dataset_repo_id=dataset_repo_id,
    ))


def jobs() -> list[dict[str, Any]]:
    with _LOCK:
        return _load_jobs()
