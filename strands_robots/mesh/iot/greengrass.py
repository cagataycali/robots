"""Greengrass v2 as the deployment and supervision layer (MAPPING rows 5.1, 5.2, 5.5): the process that mounts
robots becomes a component, the thing group becomes the fleet, a deployment becomes the fleet update.

Measured on ec2 (EDGE.md): the mesh inside the component is cloud-independent (invoke act p50 0.78 ms with the
cloud link cut for 45 s); a local deployment reaches RUNNING in 11.4 s, a cloud deployment of 5 components in 48 s.
Robot ↔ robot traffic stays on zenoh; the local MQTT broker (Moquette, MQTT 3.1.1) is for other LAN clients, not
the mesh. ``greengrassv2`` is a boto3 client the caller passes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

COMPONENT = "com.strands.robots"


def recipe(
    version: str,
    artifact_uri: str,
    *,
    tenant: str,
    robots: Sequence[Mapping[str, Any]],
    transports: Sequence[str] = ("zenoh://", "iot://us-west-2"),
    state_hz: float = 10.0,
    shadow_hz: float = 0.5,
    python: str = "python3",
) -> dict[str, Any]:
    """The component recipe: ``artifact_uri`` is the strands-robots wheel in S3; the configuration is one
    ``Mesh`` (tenant, transports) and the robots to mount (``{"name": "so101", "device_id": "arm-1"}`` …).
    ``Run`` is ``python -m strands_robots.mesh.iot.greengrass`` reading that configuration from the environment."""
    config = {"tenant": tenant, "transports": list(transports), "robots": [dict(r) for r in robots],
              "state_hz": state_hz, "shadow_hz": shadow_hz}  # fmt: skip
    wheel = artifact_uri.rsplit("/", 1)[-1]
    doc = {
        "RecipeFormatVersion": "2020-01-25",
        "ComponentName": COMPONENT,
        "ComponentVersion": version,
        "ComponentDescription": "strands-robots: the mesh with the robots of this host mounted",
        "ComponentPublisher": "strands",
        "ComponentConfiguration": {"DefaultConfiguration": config},
        "Manifests": [
            {
                "Platform": {"os": "linux"},
                "Lifecycle": {
                    "Install": {"Script": f"{python} -m pip install --user {{artifacts:path}}/{wheel}", "Timeout": 600},
                    "Run": {
                        "Script": f"{python} -m strands_robots.mesh.iot.greengrass",
                        "Setenv": {"STRANDS_GG_CONFIG": "{configuration:/}"},
                        "RequiresPrivilege": False,
                    },
                },
                "Artifacts": [{"URI": artifact_uri, "Unarchive": "NONE"}],
            }
        ],
    }
    if len(json.dumps(doc)) > 8 * 1024:
        raise ValueError("recipe over 8 KB")
    return doc


def deployment(target_arn: str, version: str, *, name: str = "strands-robots") -> dict[str, Any]:
    """The deployment document for ``greengrassv2.create_deployment``: the component to the thing group (= the
    tenant's fleet), components updated one host at a time, and a 60 s window for the robot to defer the update
    between ``run.started`` and ``run.stopped``."""
    return {
        "targetArn": target_arn,
        "deploymentName": name,
        "components": {COMPONENT: {"componentVersion": version}},
        "deploymentPolicies": {
            "failureHandlingPolicy": "ROLLBACK",
            "componentUpdatePolicy": {"timeoutInSeconds": 60, "action": "NOTIFY_COMPONENTS"},
        },
    }


def deploy(gg: Any, target_arn: str, version: str) -> str:
    """Create the deployment; returns its id."""
    return str(gg.create_deployment(**deployment(target_arn, version))["deploymentId"])


def main(config: Mapping[str, Any] | None = None, stop: Any = None) -> None:
    """The component's ``Run``: mount every configured robot on one Mesh and stay up until SIGTERM (or ``stop``)."""
    import signal
    import threading

    from strands_robots import Robot
    from strands_robots.mesh import Mesh

    from .options import IotOptions

    cfg = dict(config or json.loads(os.environ.get("STRANDS_GG_CONFIG") or "{}"))
    mesh = Mesh(tuple(cfg.get("transports") or ("zenoh://",)), tenant=cfg.get("tenant"))
    on_iot = any(t.url.startswith("iot://") for t in mesh.transports)
    iot = IotOptions(shadow_hz=float(cfg.get("shadow_hz", 0.5))) if on_iot else None
    for r in cfg.get("robots") or ():
        mesh.mount(Robot(r["name"]), device_id=r.get("device_id"), state_hz=float(cfg.get("state_hz", 10.0)), iot=iot)
    stop = stop if stop is not None else threading.Event()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.set())
    stop.wait()
    mesh.close()


if __name__ == "__main__":
    main()
