"""Offline wandb-run-id -> local-checkpoint resolver for the VERA container.

Why this exists
---------------
VERA loads its Jacobian IDM by **wandb run id**: ``load_checkpoint`` builds a
``run_path = "{entity}/{project}/{run_id}"`` and calls
``vera.utils.ckpt_utils.download_checkpoint``, which does
``wandb.Api().run(run_path)`` **unconditionally** - before it checks whether the
checkpoint already exists on disk. In an offline container (no wandb network /
no API key) that call raises, so the server never comes up.

But the released checkpoints DO carry their wandb run path in ``provenance.json``
(e.g. ``idm-mimicgen-37oa162u/provenance.json`` ->
``your-wandb-entity/jacobian-learning/37oa162u``). So we can resolve the run id
to the locally-mounted ckpt dir WITHOUT wandb.

What this does
--------------
Monkeypatches ``vera.utils.ckpt_utils.download_checkpoint`` (and the symbol
already imported into ``vera.policy.motion_policy_loading``) with a version that:

1. Scans ``$VERA_CKPT_ROOT`` for ``*/provenance.json`` whose ``wandb_run`` matches
   the requested ``run_path`` (full ``entity/project/run_id`` OR just the trailing
   ``run_id`` - tolerant of the entity/project drift between code defaults and the
   released artifacts).
2. If found, returns that dir's ``model.ckpt`` (+ the ``config.yaml`` sidecar as
   the run config when ``return_config=True``), exactly like the real function.
3. If NOT found, falls back to the original wandb-backed implementation (so online
   use / other run ids still work).

Activate by importing this module before building the policy. The entrypoint sets
``PYTHONSTARTUP``-style activation via ``-c "import wandb_offline_resolve"`` … but
simplest is: the entrypoint runs ``python -c "import wandb_offline_resolve; import runpy; runpy.run_module('vera.server.start_vera_server', run_name='__main__')"``.
See entrypoint.sh.

No edits to the VERA source tree are required.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("vera.offline_resolve")


def _index_local_ckpts(ckpt_root: str) -> dict[str, Path]:
    """Map every known wandb_run string (full + trailing run_id) -> ckpt dir."""
    index: dict[str, Path] = {}
    for prov in glob.glob(os.path.join(ckpt_root, "**", "provenance.json"), recursive=True):
        try:
            data = json.loads(Path(prov).read_text())
        except Exception:
            continue
        run = data.get("wandb_run")
        if not run:
            continue
        ckpt_dir = Path(prov).parent
        if not (ckpt_dir / "model.ckpt").exists():
            continue
        index[run] = ckpt_dir  # full "entity/project/run_id"
        index[run.split("/")[-1]] = ckpt_dir  # trailing run_id only
    return index


def install(ckpt_root: str | None = None) -> None:
    """Monkeypatch download_checkpoint to resolve locally first (offline-safe)."""
    ckpt_root = ckpt_root or os.environ.get("VERA_CKPT_ROOT", "/ckpts")
    try:
        import vera.utils.ckpt_utils as cu
    except Exception as e:  # pragma: no cover - only meaningful inside the container
        log.warning("[offline-resolve] could not import vera.utils.ckpt_utils: %s", e)
        return

    index = _index_local_ckpts(ckpt_root)
    if index:
        log.info(
            "[offline-resolve] indexed %d local ckpt run ids under %s: %s",
            len(index),
            ckpt_root,
            sorted({v.name for v in index.values()}),
        )
    else:
        log.warning(
            "[offline-resolve] no local provenance.json ckpts found under %s; wandb will be used (online required)",
            ckpt_root,
        )

    _orig_download = cu.download_checkpoint

    def _patched(run_path, download_dir, option="latest", return_config=False, force_redownload=False):  # noqa: ANN001 - match original signature
        key = run_path if run_path in index else run_path.split("/")[-1]
        local_dir = index.get(key)
        if local_dir is not None and not force_redownload:
            ckpt = local_dir / "model.ckpt"
            log.info("[offline-resolve] %s -> %s (local, no wandb)", run_path, ckpt)
            if not return_config:
                return ckpt
            # Return the config.yaml sidecar as the run config dict.
            cfg_dict: dict[str, Any] = {}
            sidecar = local_dir / "config.yaml"
            if sidecar.exists():
                try:
                    from omegaconf import OmegaConf

                    cfg_dict = OmegaConf.to_container(OmegaConf.load(sidecar), resolve=True)  # type: ignore[assignment]
                except Exception as e:
                    log.warning("[offline-resolve] failed to read %s: %s", sidecar, e)
            return ckpt, cfg_dict
        # Unknown run id -> fall back to the real (wandb) implementation.
        log.info("[offline-resolve] %s not local; falling back to wandb", run_path)
        return _orig_download(
            run_path, download_dir, option=option, return_config=return_config, force_redownload=force_redownload
        )

    cu.download_checkpoint = _patched
    # The symbol is also imported by-name into motion_policy_loading at import time.
    try:
        import vera.policy.motion_policy_loading as mpl

        if hasattr(mpl, "download_checkpoint"):
            mpl.download_checkpoint = _patched
    except Exception as exc:
        # motion_policy_loading may not be importable in every server build; the
        # primary checkpoint_utils patch above already covers the common path, so a
        # missing by-name re-import is non-fatal. Log at debug for diagnosability.
        log.debug("[offline-resolve] motion_policy_loading patch skipped: %s", exc)
    log.info("[offline-resolve] download_checkpoint patched (local-first, wandb-fallback)")


# Auto-install on import so `python -c "import wandb_offline_resolve"` is enough.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
install()
