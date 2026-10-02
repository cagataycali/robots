"""``/api/env`` - the dashboard's own Python environment, and installing an extra into it.

``GET /api/env`` lists the package's declared extras with installed/missing per
extra; ``POST /api/env/install {extra}`` starts ONE install of a declared extra
(409 while one runs, 422 for a name the package does not declare - never a
package name from the client); ``GET /api/env/install/{id}`` streams its
redacted log tail and exit code. Every route needs a session, and the start
lands in the activity trail like a calibration run does.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from strands_robots.dashboard import access, env_install

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["env"])


def _record_activity(request: Request, *args: Any, **kwargs: Any) -> None:
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None:
        return
    try:
        bridge.record_activity(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the ledger is a courtesy, the action already happened
        logger.debug("activity not recorded: %s", exc)


@router.get("/env")
async def get_env(_: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Which extras this interpreter has, and the install running right now if any."""
    return await asyncio.to_thread(env_install.snapshot)


@router.post("/env/install")
async def post_env_install(
    payload: dict[str, Any], request: Request, who: dict = Depends(access.require_session)
) -> dict[str, Any]:
    """Install one declared extra into the dashboard's interpreter; one at a time."""
    # Changing the environment from a loopback-trusted session is honoured from the
    # loopback itself only, the rule ``/api/settings`` and ``/api/config`` apply.
    if who.get("via") == "loopback" and not access.peer_is_loopback(request):
        raise HTTPException(401, "sign in required")
    extra = payload.get("extra")
    try:
        run = await asyncio.to_thread(env_install.start, extra)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    _record_activity(request, "env", "install", target=run.extra, detail=" ".join(run.command))
    return run.status()


@router.get("/env/install/{run_id}")
async def get_env_install(run_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Where the install is: running, done or failed, with its redacted log tail."""
    run = env_install.get(run_id)
    if run is None:
        raise HTTPException(404, f"no install session {run_id!r}")
    status = run.status()
    if status["status"] == "done":
        env_install.refresh_import_caches()
    return status
