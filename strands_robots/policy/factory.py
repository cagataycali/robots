"""``Policy(spec, **kw)``: pick a provider from the FORM of ``spec``, never from a policy family name."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_PRESET = re.compile(r"^[a-z][a-z0-9_]*(\?[^\s/]*)?$")
FORMS = (
    "<preset>[?k=v&k2=v2]            a named ONNX preset, run in-process",
    "zenoh://host:port/<deployment>   a lerobot-policy-server",
    "<owner>/<repo>                   a LeRobot checkpoint on the Hugging Face Hub, run in-process",
    "<dir with config.json>           a local LeRobot checkpoint, run in-process",
    "<dir with meta/info.json>[?episode=n]   a recorded LeRobot dataset, one episode replayed",
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
    path, _, query = s.partition("?")
    if _PRESET.match(s):  # a known preset wins: the spec is a name, never a path in the CWD
        from .onnx import preset, preset_names

        if path in preset_names():
            return preset(s, **kw)
    local = Path(path).expanduser()
    if (local / "meta" / "info.json").is_file():
        from .dataset import ReplayPolicy

        try:  # the query is a grammar, not a hint: one key, ``episode``; anything else is refused by name
            params = dict(parse_qsl(query, keep_blank_values=True, strict_parsing=bool(query)))
        except ValueError as exc:
            raise ValueError(f"Policy({s!r}): malformed query {query!r}; the form is <dir>?episode=n ({exc})") from exc
        if unknown := sorted(set(params) - {"episode"}):
            raise ValueError(f"Policy({s!r}): unknown parameter {', '.join(map(repr, unknown))}; known: episode")
        episode = params.get("episode", "0")
        if not episode.isdigit():
            raise ValueError(f"Policy({s!r}): episode must be an integer >= 0, got {episode!r}")
        return ReplayPolicy(local, int(episode), **kw)
    if (local / "config.json").is_file() or _REPO.match(s):
        from .chunked import ChunkedPolicy
        from .lerobot import LeRobotPolicy

        replan = kw.pop("replan_every", None)
        return ChunkedPolicy(LeRobotPolicy(s, **kw), replan_every=replan)
    if local.is_dir():
        raise ValueError(
            f"Policy({s!r}): a directory, but neither a checkpoint nor a dataset; forms:\n  " + "\n  ".join(FORMS)
        )
    if _PRESET.match(s):
        from .onnx import preset_names

        known = f"; known presets: {', '.join(preset_names())}"
        raise ValueError(f"Policy({s!r}): unknown form{known}; forms:\n  " + "\n  ".join(FORMS))
    raise ValueError(f"Policy({s!r}): unknown form; forms:\n  " + "\n  ".join(FORMS))


def from_spec(spec: str) -> Any:
    """``from_spec("microduck?vx=0.3")`` is ``Policy("microduck?vx=0.3")``: a spec string alone, no keywords."""
    if not isinstance(spec, str):
        raise TypeError(f"from_spec takes a spec string, got {type(spec).__name__}")
    return make(spec)
