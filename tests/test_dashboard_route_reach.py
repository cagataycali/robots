"""Q126: every HTTP route must be REACHED by something, or say who it is for.

Three iterations of censuses (Q123 the deploy snippet with no button, Q124 the passkeys nobody
could revoke, Q125 the build-time path guard) each started by hand-counting routes and each had to
re-decide the same handful of "is this one dead?" questions. This test settles them once, from
EVIDENCE rather than an allowlist, so a route can only stay in the surface if something points at
it -- and a NEW unreached route fails here, at the moment it is added, instead of being discovered
by accident months later.

Four kinds of evidence count, because all four are real ways a route gets used:

* a frontend module fetches it (the dashboard's own screens);
* the server EMITS its URL into a payload the frontend renders (record thumbnails work this way --
  the path never appears in the frontend source at all);
* docs/ documents it for curl/scripting (a deliberate public rail, e.g. /api/frame/{peer}/{cam});
* a script in scripts/ sweeps it (the field audits).

NARROWING LAW: route discovery walks every module and is prefix-aware, because server.py holds 68
of the routes and record_api.py registers 9 more on an APIRouter(prefix=...) -- a census that greps
server.py alone is 12% short, which is how Q125's first version invented eleven missing routes.
Comments are stripped from the frontend before it is searched: a JSDoc line that MENTIONS a path is
documentation, not a caller.
"""

from __future__ import annotations

import re
from pathlib import Path

DASH = Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard"
REPO = DASH.parents[1]

ROUTE_RE = re.compile(r'@(\w+)\.(get|post|put|patch|delete|websocket)\(\s*["\']([^"\']+)["\']')
PREFIX_RE = re.compile(r'(\w+)\s*=\s*APIRouter\(\s*prefix\s*=\s*["\']([^"\']*)["\']')
COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/|^\s*//.*$", re.M)


def _py_sources() -> list[Path]:
    return [p for p in DASH.rglob("*.py") if "frontend" not in p.parts]


def _routes() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for f in _py_sources():
        text = f.read_text()
        prefixes = {m.group(1): m.group(2) for m in PREFIX_RE.finditer(text)}
        for m in ROUTE_RE.finditer(text):
            obj, verb, path = m.group(1), m.group(2).upper(), m.group(3)
            out.append((verb, prefixes.get(obj, "") + path, f.name))
    return out


def _read_all(paths) -> str:
    return "\n".join(p.read_text(errors="ignore") for p in paths)


def _reaches(path: str, haystack: str) -> bool:
    """The literal prefix before the first path param, plus every literal segment after the last."""
    head = path.split("{")[0]
    if head not in haystack:
        return False
    tail = [s for s in path.split("}")[-1].split("/") if s]
    return all(t in haystack for t in tail)


def test_every_route_is_reached_or_explained() -> None:
    routes = _routes()
    src = DASH / "frontend" / "src"
    frontend = COMMENT_RE.sub(
        "", _read_all([p for p in src.rglob("*.ts*") if ".test." not in p.name])
    )
    # URLs the server hands to the client. The route DECLARATIONS are removed first, so a route
    # cannot be its own evidence: with a prefixed router the decorator holds only "/thumb/{...}"
    # while the emitted URL is the full "/api/record/thumb/..." string, and counting occurrences
    # instead of removing the declarations mis-judged exactly that case.
    emitted = ROUTE_RE.sub("", _read_all(_py_sources()))
    docs = _read_all(list((REPO / "docs").rglob("*.md")))
    scripts = _read_all(
        [p for p in (REPO / "scripts").rglob("*.py")]
        + [p for p in (src.parent / "scripts").rglob("*.mjs")]
    )

    # The scan must not silently narrow to nothing: an empty haystack would mark everything
    # unreached, and an empty route list would pass this test while checking nothing at all.
    assert len(routes) >= 60, f"route discovery collapsed: {len(routes)} routes"
    for name, blob in (("frontend", frontend), ("docs", docs), ("scripts", scripts)):
        assert len(blob) > 1000, f"{name} corpus is empty ({len(blob)} chars) — the test is broken"

    unexplained = []
    for verb, path, where in routes:
        if _reaches(path, frontend) or _reaches(path, docs) or _reaches(path, scripts):
            continue
        # A URL the server builds for the client counts (an f-string template, e.g. thumbnails).
        if _reaches(path, emitted):
            continue
        unexplained.append(f"{verb} {path} ({where})")

    assert not unexplained, (
        "these routes are reached by nothing and documented nowhere — give each one a caller, a "
        "line in docs/, or delete it:\n  " + "\n  ".join(unexplained)
    )
