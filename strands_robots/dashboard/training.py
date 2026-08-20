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
#: Dataset roots the dashboard itself created (collect wizard). The default
#: scanner only walks HF_LEROBOT_HOME + STRANDS_ROBOTS_DATA_DIRS, so a
#: collect into /tmp/... would produce a dataset the Training tab cannot see.
ROOTS_FILE = JOBS_FILE.parent / "dataset_roots.json"
_LOCK = threading.Lock()


#: Set when a ledger file existed but could not be read. A record of every training
#: run this dashboard started is not something to lose silently: an empty list and an
#: unreadable file render identically as "No training jobs yet", while the runs
#: themselves keep going for hours with no card, no status and no export button.
_JOBS_PROBLEM: str | None = None


def _write_json_durably(path: Path, payload: Any) -> None:
    """Write JSON so a crash mid-write cannot destroy what was already there.

    ``Path.write_text`` TRUNCATES the file and then writes into it, so a kill or a
    power loss in that window leaves a half file. The next read fails, the loader
    falls back to "no jobs", and the next save makes that loss permanent. Writing a
    sibling temp file and ``os.replace``-ing it is atomic on POSIX: a reader sees
    either the old file or the new one, never a partial one. The fsync is what makes
    that promise survive the machine losing power, not just the process dying.
    """
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
    """Move an unreadable file aside, preserving it, and return where it went.

    Deleting it would be easier and is exactly wrong: this file is the only record
    that some runs happened, it may be recoverable by hand, and whatever corrupted it
    is worth being able to look at.
    """
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
    """The job ledger, or an empty list - with ``_JOBS_PROBLEM`` set if it was lost.

    A ledger that cannot be parsed is quarantined rather than overwritten, because
    the alternative is silent and permanent: the loader returns [], the UI says "No
    training jobs yet", and the next submit saves a one-entry list over the file that
    still held every earlier run.
    """
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


#: Trainer modules whose spec this form cannot build. Q48: the provider dropdown
#: offered `ppo` and `fast_sac`, and picking either one made validate answer
#: "ppo requires an RLTrainSpec, got TrainSpec" - a sentence about internal
#: classes, on a path that can NEVER succeed no matter what the operator types,
#: because the form builds a supervised TrainSpec by construction. An option that
#: cannot work is worse than an absent one: it costs a dataset choice, a click and
#: a bewildering error before it refuses.
_RL_TRAINER_MODULE_PREFIX = "strands_robots.training.rl"

#: Phrased WITHOUT a leading article and in the plural on purpose: the form joins these
#: reasons after a list of provider names, and "fast_sac and ppo are a reinforcement-learning
#: trainer" is the sentence that comes out of the obvious singular wording.
_RL_REASON = (
    "reinforcement-learning trainers learn from a live environment, not from a recorded "
    "dataset, so they are driven from a script (RLTrainSpec) rather than this form"
)


def _declared_trainer_module(provider: str) -> str:
    """Where a provider's trainer class comes from, WITHOUT importing it.

    Two declaration sites, because the SDK has two: registry/policies.json names the
    module as data, while ppo/fast_sac are registered at runtime with a loader lambda.
    A lambda's module is only where register_trainer was called (strands_robots.training
    for both), so that tells us nothing - but the module it IMPORTS is a constant in the
    loader's code object, which is readable without executing it. That matters: importing
    a trainer class pulls torch (and for GR00T/Cosmos3, much more) into a route that runs
    on every page load.
    """
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


#: Q49: providers whose validate() demands something this form has NO FIELD FOR, and cannot
#: grow one, because the key is not in SPEC_KEYS at all. cosmos3 needs a recipe TOML passed as
#: extra['sft_toml'] - the form can express provider, dataset, base model, output dir, steps,
#: method, embodiment and the LoRA knobs, and none of those is a recipe. Left selectable, it
#: promised a run that every attempt refuses.
#:
#: This list MIRRORS a requirement that the SDK declares only inside a trainer's validate(),
#: so tests/test_dashboard_trainer_form_support.py ASKS validate whether the mirror still
#: holds rather than trusting this constant.
_FORM_CANNOT_EXPRESS: dict[str, str] = {
    "cosmos3": (
        "needs a training recipe TOML (extra['sft_toml']) that selects the registered "
        "experiment, and this form has no field for it - run cosmos3 from a script"
    ),
}


def form_unsupported() -> dict[str, str]:
    """Providers that cannot be trained FROM THIS FORM, mapped to why.

    Unknown providers are deliberately absent: one we cannot classify is offered as
    usual, since guessing "unsupported" would hide a backend that works.
    """
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


#: The full set of fields a training spec accepts. submit() and validate()
#: share it so the two can never drift: a field the form sends either reaches
#: train_policy or is refused BY NAME - silently dropping a typo'd "step"
#: would train 10k default steps and call it success.
SPEC_KEYS = (
    "provider", "dataset_root", "dataset_repo_id", "base_model",
    "output_dir", "embodiment", "steps", "batch_size", "learning_rate",
    "save_freq", "method", "lora_r", "lora_alpha", "seed",
)


def _spec_kwargs(body: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(kwargs, None) for a clean body, (None, error-result) for a bad one.

    Splatting the raw request body into train_policy as **kwargs was Q6:
    any unexpected key became a TypeError, which FastAPI turned into a
    bare-HTML 500 the UI's res.json() choked on. Unknown keys now come
    back as a structured error naming them and the valid vocabulary.
    """
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
    """Does ``data/`` hold at least one file? ``None`` when it cannot be looked at.

    Stops at the FIRST file found: a chunked v3 dataset can hold thousands, and this runs once
    per listed dataset on a request thread. ``None`` (not False) when the directory is
    unreadable, because "I could not look" and "there is nothing there" lead to different
    advice - and the verdict reports the difference instead of guessing.
    """
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
                # Q37: meta/info.json exists from the moment a recording session OPENS, so
                # its presence is not evidence that anything was recorded. One cheap probe
                # (does data/ hold any file) turns "a directory with a config" into a row
                # that says whether it can be trained on. No shard is opened: listing 50
                # datasets must not become 50 dataset loads.
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
    """Type-ahead search of public LeRobot datasets on the Hub.

    Mirrors ``checkpoints.hub_search`` deliberately, including its hard-won
    behaviours: returns ``(rows, problem)`` so a hub outage, no network and a
    query with no matches cannot all render as the same empty list; a FAILURE
    IS NEVER CACHED, so the next keystroke retries; and a ``TypeError`` from a
    hub-client version bump falls back to an unsorted call instead of killing
    search.

    The filter is the ``LeRobot`` tag, which is what ``lerobot`` itself writes
    when it pushes a dataset - searching all of HF datasets would offer text
    corpora that ``train_policy`` cannot open.

    A Hub row carries NO ``root``: that absence is load-bearing. A local
    dataset trains from ``dataset_root`` (a path) and a Hub one from
    ``dataset_repo_id``, so the presence of ``root`` is how every caller tells
    which of the two fields to fill.
    """
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
    """Local dataset roots merged with a Hub search, local first.

    Local wins a repo_id collision: an already-downloaded dataset trains
    offline and instantly, so offering the Hub copy of something on disk would
    trade a working path for a download.

    Local rows come FIRST rather than being ranked against downloads, because
    they are the ones the operator just recorded - the collect wizard's whole
    promise is that what you record is what the train screen offers next
    (pinned by the U20 golden-path test).

    ``problem`` is a sentence about the HUB half only: local discovery cannot
    fail this way, and an empty screen with a working Hub is a real answer
    ("no matches") that must not be dressed up as an outage.

    The slice is a plain ``[:limit]``, NOT ``[: max(limit, len(local))]``. I
    wrote the max() version first and ``checkpoints.clamp_limit``'s docstring
    caught it: that idiom means a type-ahead asking for 1 row gets every local
    row instead. Ordering local first already guarantees truncation drops hub
    rows before local ones, so the caller's limit can simply be honoured
    (``total_matched`` reports what was found before the cut).
    """
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
    """Who the Hub thinks we are - reused from checkpoints, not re-implemented.

    A private or gated dataset is exactly the case where "no matches" and "you
    are anonymous" look identical on screen, so the picker needs the same auth
    line the checkpoint picker already shows.
    """
    try:
        from strands_robots.dashboard import checkpoints

        return checkpoints.hf_auth_state()
    except Exception as exc:  # noqa: BLE001 - never let an auth probe break search
        return {"authenticated": False, "user": None, "detail": f"auth state unavailable ({type(exc).__name__})"}


def output_dir_verdict(path: str) -> dict[str, Any]:
    """What a run would do to this directory (Q58) - see dashboard.output_dir_check."""
    from strands_robots.dashboard.output_dir_check import (
        default_checkpoint_probe,
        inspect_output_dir,
    )

    return inspect_output_dir(path, has_checkpoint=default_checkpoint_probe)


def submit(body: dict[str, Any]) -> dict[str, Any]:
    """Validate + launch a training job; persist it for the tab.

    Q58: the trainer's fresh-start hygiene ``shutil.rmtree``s a pre-existing ``output_dir`` that
    holds no resumable checkpoint. That is a silent, unrecoverable delete of whatever the operator
    typed, so it is gated here: an occupied directory refuses the launch and says what would be
    lost, and only ``confirm_clear: true`` in the body lets it through. Same continuable-refusal
    posture as this dashboard's other safety gates - the run is still possible, it just has to be
    asked for deliberately.
    """
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
    """Export the run's last checkpoint as a loadable artifact.

    ``base_model`` must ride along: the export action revalidates the spec,
    and a provider whose training required a base model (smolvla post-tune,
    GR00T) refuses the export of ITS OWN finished run without it. The job
    record carries the value - the caller just has to forward it.
    """
    from strands_robots.tools.train_policy import train_policy

    from strands_robots.dashboard.artifact_check import artifact_verdict

    res = _tool_result(train_policy(
        action="export", provider=provider, output_dir=output_dir,
        dataset_root=dataset_root or None, dataset_repo_id=dataset_repo_id,
        base_model=base_model or "",
    ))
    # The trainer's "success" is a report about its own run, not about the artifact:
    # latest_checkpoint() discovers a checkpoint dir BY ITS CONFIG FILE and the default
    # export() returns that path unchanged, so a run killed between the config write and
    # the weights write exports successfully and only fails when a policy is loaded on a
    # robot (BUGS.md Q36). Answer with BOTH verdicts and never overwrite theirs: an export
    # that ran fine and produced half a checkpoint is a different event from one that
    # refused, and the operator needs to be able to tell them apart.
    exported = res.get("data", {}).get("exported_model")
    if res["status"] == "success":
        res["artifact"] = artifact_verdict(exported)
        # One boolean for the caller that only decides "may this be handed to an arm".
        res["deployable"] = bool(res["artifact"].get("ok"))
    return res


def jobs() -> list[dict[str, Any]]:
    with _LOCK:
        return _load_jobs()
