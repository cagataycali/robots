"""Uploading a recorded LeRobotDataset into an HF Storage Bucket.

A transfer reads a finalized dataset directory and the ``hf`` CLI; it needs no
live :class:`~strands_robots.dataset_recorder.DatasetRecorder`, no sim world and
no lerobot import. Both callers are a layer above: the recorder's
:meth:`~strands_robots.dataset_recorder.DatasetRecorder.sync_to_bucket` delegate
and the idle-path bucket sync in ``stop_recording``, so the input validation and
CLI orchestration live here once, under both of them.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any

from strands_robots.utils import boolean_flag_error

logger = logging.getLogger(__name__)


# Allowlist patterns for HF Storage Bucket sync targets. Both `bucket` and
# `run_id` reach the `hf` CLI argv and the `hf://buckets/...` URI; they are
# agent-reachable via stop_recording(bucket=, run_id=) dispatched through the
# simulation action layer, so they MUST be validated before any subprocess /
# URI interpolation (AGENTS.md > LLM Input Safety). `bucket` is "name" or
# "org/name"; `run_id` is a single path segment. Neither may contain shell
# metacharacters, path-traversal (".."), or separators beyond the one allowed
# bucket "org/name" slash.
_BUCKET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?\Z")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def sync_dataset_to_bucket(
    root: str | Path,
    bucket: str,
    run_id: str | None = None,
    *,
    create: bool = True,
    private: bool = True,
    delete: bool = False,
) -> dict[str, Any]:
    """Sync an on-disk LeRobotDataset into an HF Storage Bucket (Phase 1/2).

    Lifecycle-independent: needs only a finalized dataset directory on disk
    (``meta/`` present) and the ``hf`` CLI - no live
    :class:`~strands_robots.dataset_recorder.DatasetRecorder`, no sim world. Covers syncing a dataset
    recorded earlier in the process, one recorded on hardware via
    ``lerobot-record``, or a daily re-sync of a directory that grew. Both
    :meth:`~strands_robots.dataset_recorder.DatasetRecorder.sync_to_bucket` and the idle-path bucket sync in
    ``stop_recording`` delegate here so input validation and CLI
    orchestration exist exactly once.

    Mutable, Xet-deduplicated dump target for COLLECTION - avoids git-LFS
    history bloat of push_to_hub during recording. Daily re-sync uploads
    only changed chunks (content-defined chunking). Requires the ``hf`` CLI
    with the ``buckets``/``sync`` subcommands (``huggingface_hub>=1.5``)
    and ``hf auth login``.

    ``bucket`` and ``run_id`` are validated against an allowlist before any
    subprocess or URI interpolation: ``bucket`` must be ``"name"`` or
    ``"org/name"`` and ``run_id`` a single path segment, both restricted to
    ``[A-Za-z0-9._-]`` (no path traversal or shell metacharacters). This
    path is agent-reachable via ``stop_recording(bucket=, run_id=)``. A
    rejected value returns ``{"status": "error", ...}`` without running ``hf``.

    ``create``, ``private`` and ``delete`` select *postures* rather than
    scaling a quantity, so each is checked against
    :func:`~strands_robots.utils.boolean_flag_error` before the ``hf`` CLI is
    even located - the same domain the mesh provisioning entry points apply to
    their own capability flags. Read by truthiness they fail toward the
    permissive posture in *both* directions, because every non-empty string is
    truthy and every falsy non-boolean takes the other branch:
    ``delete="false"`` - the spelling an operator reaches for when opting out -
    appends ``--delete`` and mirror-deletes remote files absent locally, while
    ``private=0`` drops ``--private`` and creates the bucket *public*.

    The shard layout is already Xet/bucket-friendly at lerobot's defaults
    (100 MB data parquet / 200 MB video MP4 shards), and ``meta/`` MUST
    ship or downstream loses normalization stats.

    Args:
        root: Local dataset directory, ``str`` or ``Path`` (must contain
            ``meta/``).
        bucket: Bucket target, ``"name"`` or ``"org/name"``.
        run_id: Subpath inside the bucket; defaults to the dataset directory
            name (``Path(root).name``).
        create: Create the bucket first (pre-existing bucket is not an error).
            Must be a boolean.
        private: Create the bucket as private (only used with ``create=True``).
            Must be a boolean.
        delete: Forward ``--delete`` to ``hf sync`` (mirror semantics -
            remove remote files absent locally). Must be a boolean.

    Returns:
        ``{"status": "success", "bucket_uri": ...}`` or
        ``{"status": "error", "message": ...}``. Never raises on ``hf``
        failure; errors are surfaced in the result dict. A flag outside its
        domain is reported the same way, without locating or running the CLI.
    """
    # Before the CLI probe so the same caller mistake reports identically
    # whether or not `hf` is installed, and so a refused posture flag can
    # never reach `hf buckets create` or `hf sync`.
    for flag_name, flag_value in (("create", create), ("private", private), ("delete", delete)):
        if flag_error := boolean_flag_error(flag_value, flag_name, "sync_dataset_to_bucket"):
            return {"status": "error", "message": flag_error}

    import subprocess

    hf = _hf_executable()
    if hf is None:
        return {
            "status": "error",
            "message": f'`hf` CLI not found. pip install -U "{_HF_BUCKET_CLI_MIN_SPEC}" and run `hf auth login`.',
        }

    # `hf buckets` / `hf sync` need huggingface_hub>=1.5; on every older
    # release (0.36.x, but also 1.0-1.4.x) the CLI exists and rejects those
    # subcommands with usage noise. Gate on the installed package version so
    # users get an upgrade instruction instead.
    version_error = _huggingface_hub_version_error()
    if version_error is not None:
        return {"status": "error", "message": version_error}

    if not _BUCKET_RE.match(bucket):
        return {
            "status": "error",
            "message": f"invalid bucket {bucket!r}: must match "
            "'name' or 'org/name' using [A-Za-z0-9._-] (no path traversal "
            "or shell metacharacters).",
        }

    local_root = str(root)
    # meta/ must ship or downstream loses normalization stats.
    if not (Path(local_root) / "meta").exists():
        return {
            "status": "error",
            "message": f"No meta/ under {local_root}; the dataset was never finalized. "
            "Call finalize() (stop_recording does this) before syncing to a bucket "
            "(stats/info required for streaming/training).",
        }

    run_id = run_id or Path(local_root).name
    if not _RUN_ID_RE.match(run_id):
        return {
            "status": "error",
            "message": f"invalid run_id {run_id!r}: must be a single path "
            "segment using [A-Za-z0-9._-] (no '/', path traversal, or shell "
            "metacharacters).",
        }
    dest = f"hf://buckets/{bucket}/{run_id}"

    if create:
        cp = subprocess.run(
            [hf, "buckets", "create", bucket] + (["--private"] if private else []),
            capture_output=True,
            text=True,
            errors="replace",
        )
        blob = (cp.stderr + cp.stdout).lower()
        # An already-created bucket is the normal case for a daily re-sync, so it
        # must not fail the sync. The hub reports it as "You already created this
        # bucket repo" with a 409, which does not contain "exists" - match the
        # status code and both phrasings rather than one substring.
        already_exists = "exist" in blob or "409" in blob or "already created" in blob
        if cp.returncode != 0 and not already_exists:
            return {
                "status": "error",
                "message": f"bucket create failed: {cp.stderr.strip()}",
            }

    cmd = [hf, "sync", local_root, dest]
    if delete:
        cmd.append("--delete")
    logger.info("Syncing %s -> %s", local_root, dest)
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        return {
            "status": "error",
            "message": proc.stderr.strip() or proc.stdout.strip(),
        }

    return {"status": "success", "bucket_uri": dest}


def _hf_executable() -> str | None:
    """Resolve the ``hf`` CLI, preferring the one in the running interpreter's
    environment before falling back to PATH.

    ``huggingface_hub`` installs the ``hf`` entry point next to the active
    Python (e.g. inside a virtualenv's ``bin``/``Scripts``). A bare ``hf`` on
    PATH is only found when that environment is also on PATH, which is often not
    the case for a subprocess launched from a venv whose ``bin`` was never
    activated. Checking ``sys.executable``'s directory first makes
    ``sync_to_bucket`` work from any environment where ``huggingface_hub`` is
    installed, not just an activated one. Returns ``None`` if no ``hf`` is found.
    """
    import shutil

    exe_dir = Path(sys.executable).parent
    for name in ("hf", "hf.exe"):
        candidate = exe_dir / name
        if candidate.exists():
            return str(candidate)
    return shutil.which("hf")


#: Oldest ``huggingface_hub`` release whose ``hf`` CLI carries the
#: ``hf buckets`` / ``hf sync`` subcommands :func:`sync_dataset_to_bucket`
#: invokes.
#:
#: They first ship in 1.5.0, as ``huggingface_hub/cli/buckets.py`` registered by
#: ``cli/hf.py`` (``app.add_group(buckets_cli, name="buckets")`` and
#: ``app.command()(sync)``). 1.0-1.4.x install the ``hf`` entry point without
#: that module, so they answer both invocations with
#: ``Error: No such command 'buckets'`` / ``'sync'``.
#:
#: This is the single source of the floor: the version gate below, the upgrade
#: instructions it and :func:`sync_dataset_to_bucket` emit, and the pin the
#: ``[wbc]`` extra declares are all checked against it, so the accepted domain
#: cannot drift from the release that can actually honor a bucket sync.
_HF_BUCKET_CLI_MIN_VERSION = (1, 5)

#: The requirement string every bucket-sync upgrade instruction quotes, derived
#: from :data:`_HF_BUCKET_CLI_MIN_VERSION` so the advice cannot name a release
#: that does not ship the subcommands.
_HF_BUCKET_CLI_MIN_SPEC = "huggingface_hub>=" + ".".join(str(part) for part in _HF_BUCKET_CLI_MIN_VERSION)


def _huggingface_hub_version_error() -> str | None:
    """Return an actionable error message if ``huggingface_hub`` is too old for bucket sync.

    The ``hf buckets`` / ``hf sync`` subcommands ship in
    :data:`_HF_BUCKET_CLI_MIN_VERSION` and later. On any older release - the
    0.36.x stable line, but equally 1.0-1.4.x - the ``hf`` binary exists, so
    :func:`_hf_executable` succeeds, but the subcommands fail with usage noise
    (``Error: No such command 'buckets'``) that gives no hint the fix is an
    upgrade. Version-checking the installed package up front turns that noise
    into a clear upgrade instruction without spawning a subprocess.

    Returns ``None`` (no error) when:

    - the installed version is >= :data:`_HF_BUCKET_CLI_MIN_VERSION`, or
    - ``huggingface_hub`` is not importable in this interpreter (the ``hf``
      binary may come from a different environment on PATH whose version we
      cannot see; the normal subprocess error path still applies), or
    - the version string is unparseable (fail open rather than block a
      possibly-capable CLI on a cosmetic version format).
    """
    try:
        import huggingface_hub
    except ImportError:
        return None

    version = getattr(huggingface_hub, "__version__", "")
    match = re.match(r"(\d+)\.(\d+)", version)
    if match is None:
        return None
    if (int(match.group(1)), int(match.group(2))) >= _HF_BUCKET_CLI_MIN_VERSION:
        return None
    return (
        f"bucket sync requires {_HF_BUCKET_CLI_MIN_SPEC} (`hf buckets`/`hf sync`); "
        f"installed: {version}. pip install -U '{_HF_BUCKET_CLI_MIN_SPEC}'."
    )
