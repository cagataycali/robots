"""Measure what each consumer of the VERA server_port receives, per tree."""
from __future__ import annotations
import json, math, pathlib, sys

import strands_robots.policies.vera.config as _cfgmod
TREE = str(pathlib.Path(_cfgmod.__file__).parents[3])

from strands_robots.policies.vera.provider import VeraPolicy
from strands_robots.policies.vera.server_runner import VeraServerRunner

CASES = [8820, 2.7, True, 0, -1, 70000, math.nan, "8820"]

def label(v):
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return repr(v)

out = {"tree": TREE, "rows": []}
for X in CASES:
    row = {"value": label(X)}
    try:
        p = VeraPolicy(embodiment="pusht", server_port=X, auto_launch_server=False)
        cmd = VeraServerRunner(p.config)._build_command()
        row.update(
            verdict="accepted",
            client=p._client.uri,
            client_port=str(p._client.port),
            server_uri=p.config.server_uri,
            argv=cmd[cmd.index("--port") + 1],
        )
        seen = {row["client_port"], row["server_uri"].rsplit(":", 1)[1], row["argv"]}
        row["agree"] = len(seen) == 1
        row["distinct"] = len(seen)
    except BaseException as exc:  # noqa: BLE001 - the escape IS the finding
        row.update(verdict="refused", exc=type(exc).__name__, msg=str(exc)[:110],
                   agree=None, distinct=0)
    out["rows"].append(row)

# vis_port: the documented zero must keep disabling the viewer
vis = {}
for V in [8821, 0, -1, True]:
    try:
        p = VeraPolicy(embodiment="pusht", vis_port=V, auto_launch_server=False)
        cmd = VeraServerRunner(p.config)._build_command()
        vis[label(V)] = cmd[cmd.index("--vis-port") + 1] if "--vis-port" in cmd else "viewer disabled"
    except BaseException as exc:  # noqa: BLE001
        vis[label(V)] = f"refused: {type(exc).__name__}"
out["vis"] = vis

pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
print("TREE:", TREE)
