"""``Policy(spec, **kw)``: pick a provider from the FORM of ``spec``, never from a policy family name."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
FORMS = (
    "zenoh://host:port/<deployment>   a lerobot-policy-server",
    "<owner>/<repo>                   a LeRobot checkpoint on the Hugging Face Hub, run in-process",
    "<dir with config.json>           a local LeRobot checkpoint, run in-process",
    "<callable>                       f(obs) -> action",
)


def make(spec: Any, **kw: Any) -> Any:
    if not isinstance(spec, (str, Path)):
        if callable(spec) and not kw:
            from .callable import CallablePolicy

            return CallablePolicy(spec)
        raise TypeError(f"Policy({type(spec).__name__}) is not a policy spec; forms:\n  " + "\n  ".join(FORMS))
    s = str(spec)
    if s.startswith("zenoh://"):
        from .lerobot import ZenohPolicy

        u = urlparse(s)
        deployment = u.path.strip("/")
        if not u.hostname or not u.port or not deployment or "/" in deployment:
            raise ValueError(f"{s!r}: expected zenoh://host:port/<deployment>")
        return ZenohPolicy(endpoint=f"tcp/{u.hostname}:{u.port}", deployment=deployment, **kw)
    if (Path(s).expanduser() / "config.json").is_file() or _REPO.match(s):
        from .chunked import ChunkedPolicy
        from .lerobot import LeRobotPolicy

        replan = kw.pop("replan_every", None)
        return ChunkedPolicy(LeRobotPolicy(s, **kw), replan_every=replan)
    raise ValueError(f"Policy({s!r}): unknown form; forms:\n  " + "\n  ".join(FORMS))
