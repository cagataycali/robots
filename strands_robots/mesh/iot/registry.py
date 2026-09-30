"""Read-only view of the AWS IoT Thing registry, for a fleet view that lists what is
provisioned next to what is speaking.

The dashboard's fleet grid is fed by presence heartbeats: a robot that has not
booted, or went quiet, is simply absent. The registry is the second source: a
Thing that exists in the account is a robot somebody provisioned, and a fleet
owner wants to see it as a grey card ("last seen", "never heard") rather than
wonder whether provisioning happened at all.

Everything here is a read. :func:`list_things` calls ``iot:ListThings`` (paged,
capped) and, when the account has fleet indexing with connectivity switched on,
``iot:SearchIndex`` for the broker's own connected/disconnected verdict. Neither
call changes account state. boto3 is imported lazily so the module is importable
without the ``mesh-iot`` extra, and every failure mode is a ``status`` word on
the returned view, never an exception: a dashboard must render the rest of the
fleet when the registry cannot be read.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from strands_robots.mesh.security import as_wire_timestamp
from strands_robots.utils import refusal_str

logger = logging.getLogger(__name__)

#: Upper bound on Things one view returns. A fleet larger than this is a paging
#: question for the caller, not a wall of cards.
MAX_THINGS = 250

#: Environment variable that switches the registry read off outright (a dashboard
#: on a box with AWS credentials for an unrelated account should not list them).
DISABLE_ENV = "STRANDS_DASHBOARD_IOT_REGISTRY"

#: The ``status`` words a :class:`RegistryView` can carry.
STATUSES = ("ok", "off", "no-boto3", "no-credentials", "denied", "error")


@dataclass(frozen=True)
class RegistryThing:
    """One Thing as the fleet view needs it."""

    thing_name: str
    thing_type: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    #: The broker's connectivity verdict from the fleet index, ``None`` when the account
    #: does not index connectivity. Never inferred.
    connectivity: str | None = None
    #: Epoch seconds of the index's last connect or disconnect event, ``None`` when unknown.
    last_seen: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON shape for ``/api/mesh/iot/registry``."""
        return {
            "thing_name": self.thing_name,
            "thing_type": self.thing_type,
            "attributes": dict(self.attributes),
            "connectivity": self.connectivity,
            "last_seen": self.last_seen,
        }


@dataclass(frozen=True)
class RegistryView:
    """The registry read's outcome: a status word, a detail sentence and the Things."""

    status: str
    detail: str = ""
    region: str | None = None
    things: tuple[RegistryThing, ...] = ()
    #: Whether the connectivity verdicts came from the fleet index (``True``) or are absent.
    indexed: bool = False

    def as_dict(self) -> dict[str, Any]:
        """JSON shape for ``/api/mesh/iot/registry``."""
        return {
            "status": self.status,
            "detail": self.detail,
            "region": self.region,
            "indexed": self.indexed,
            "things": [t.as_dict() for t in self.things],
            "count": len(self.things),
        }


def registry_disabled_by_env() -> bool:
    """``True`` when :data:`DISABLE_ENV` is set to ``0``, ``false`` or ``off``."""
    return os.getenv(DISABLE_ENV, "").strip().lower() in ("0", "false", "off")


def _thing_row(raw: dict[str, Any], connectivity: dict[str, Any] | None) -> RegistryThing:
    name = str(raw.get("thingName") or "")
    attributes = {str(k): str(v) for k, v in (raw.get("attributes") or {}).items()}
    thing_type = raw.get("thingTypeName")
    verdict: str | None = None
    last_seen: float | None = None
    if isinstance(connectivity, dict):
        connected = connectivity.get("connected")
        if isinstance(connected, bool):
            verdict = "connected" if connected else "disconnected"
        stamp = as_wire_timestamp(connectivity.get("timestamp"))
        if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and stamp > 0:
            # The index reports milliseconds.
            last_seen = float(stamp) / 1000.0
    return RegistryThing(
        thing_name=name,
        thing_type=str(thing_type) if thing_type else None,
        attributes=attributes,
        connectivity=verdict,
        last_seen=last_seen,
    )


def _connectivity_index(iot: Any, names: list[str]) -> dict[str, dict[str, Any]] | None:
    """``thingName -> connectivity`` from the fleet index, ``None`` when indexing is off."""
    try:
        conf = iot.get_indexing_configuration().get("thingIndexingConfiguration") or {}
    except Exception as exc:  # noqa: BLE001 - a missing permission leaves the verdict unknown
        logger.debug("iot registry: get_indexing_configuration failed: %s", exc)
        return None
    if str(conf.get("thingConnectivityIndexingMode") or "OFF").upper() == "OFF":
        return None
    out: dict[str, dict[str, Any]] = {}
    try:
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"indexName": "AWS_Things", "queryString": "thingName:*", "maxResults": 250}
            if token:
                kwargs["nextToken"] = token
            page = iot.search_index(**kwargs)
            for row in page.get("things") or []:
                name = row.get("thingName")
                conn = row.get("connectivity")
                if isinstance(name, str) and isinstance(conn, dict):
                    out[name] = conn
            token = page.get("nextToken")
            if not token or len(out) >= MAX_THINGS:
                break
    except Exception as exc:  # noqa: BLE001 - the listing stands without verdicts
        logger.debug("iot registry: search_index failed: %s", exc)
        return None
    return out


def list_things(region: str | None = None, *, prefix: str | None = None, max_things: int = MAX_THINGS) -> RegistryView:
    """Every Thing the credentials in the environment can list, as a :class:`RegistryView`.

    Args:
        region: AWS region; the boto3 default session region when ``None``.
        prefix: Keep only Things whose name starts with this text (a fleet owner's naming
            scheme, ``dm-`` for example). ``None`` keeps every Thing.
        max_things: Cap on the rows returned, :data:`MAX_THINGS` by default.

    Returns:
        A view whose ``status`` is ``"ok"`` with the Things, or one of ``"off"``,
        ``"no-boto3"``, ``"no-credentials"``, ``"denied"``, ``"error"`` with an
        empty tuple and a ``detail`` sentence the fleet bar can show as is.
    """
    if registry_disabled_by_env():
        return RegistryView(status="off", detail=f"{DISABLE_ENV} switches the registry read off")
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, NoRegionError
    except ImportError:
        return RegistryView(status="no-boto3", detail="boto3 is not installed; install strands-robots[mesh-iot]")
    try:
        iot = boto3.client("iot", region_name=region)
    except NoRegionError:
        return RegistryView(status="no-credentials", detail="no AWS region configured (AWS_REGION or a profile)")
    except (BotoCoreError, ValueError) as exc:
        return RegistryView(status="error", detail=f"boto3 client: {refusal_str(exc)}")
    resolved_region = getattr(getattr(iot, "meta", None), "region_name", None) or region
    rows: list[dict[str, Any]] = []
    try:
        token: str | None = None
        while len(rows) < max_things:
            kwargs: dict[str, Any] = {"maxResults": min(250, max_things - len(rows))}
            if token:
                kwargs["nextToken"] = token
            page = iot.list_things(**kwargs)
            rows.extend(page.get("things") or [])
            token = page.get("nextToken")
            if not token:
                break
    except NoCredentialsError:
        return RegistryView(
            status="no-credentials", detail="no AWS credentials in the environment", region=resolved_region
        )
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code") or "")
        if code in (
            "AccessDeniedException",
            "UnauthorizedException",
            "UnrecognizedClientException",
            "ExpiredTokenException",
        ):
            return RegistryView(
                status="denied", detail=f"iot:ListThings refused: {refusal_str(code)}", region=resolved_region
            )
        return RegistryView(
            status="error", detail=f"iot:ListThings failed: {refusal_str(code or exc)}", region=resolved_region
        )
    except BotoCoreError as exc:
        return RegistryView(status="error", detail=f"iot:ListThings failed: {refusal_str(exc)}", region=resolved_region)
    if prefix:
        rows = [r for r in rows if str(r.get("thingName") or "").startswith(prefix)]
    rows = rows[:max_things]
    names = [str(r.get("thingName") or "") for r in rows]
    index = _connectivity_index(iot, names) if rows else None
    things = tuple(_thing_row(r, (index or {}).get(str(r.get("thingName") or "")) if index else None) for r in rows)
    return RegistryView(
        status="ok",
        detail=f"{len(things)} things"
        + ("" if index is not None else "; fleet indexing is off, so no connectivity verdict"),
        region=resolved_region,
        things=things,
        indexed=index is not None,
    )
