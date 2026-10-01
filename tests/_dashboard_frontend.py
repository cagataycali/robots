# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the dashboard SPA's ``src/lib`` modules under Node, as a browser would.

The frontend ships no test runner of its own, so the cells
about ``endpoints.ts`` (which host a request is built for, whether the bearer
rides along) are pytest cells that hand a script to Node when one is on the
machine and skip when none is, or when the one on PATH is older than the first
release that strips TypeScript types itself (22.18 on the LTS line, 23.6 after),
so nothing is compiled: the ``lib`` directory is copied, the extensionless
relative imports the bundler resolves are given their ``.ts`` back, and the
script imports the module it is about.

The browser the script sees is the minimum a lib module reads: ``localStorage``
(a Map), ``location`` (whatever the cell says the address bar holds),
``history.replaceState`` (recorded, so a cell can check what was scrubbed) and
``fetch`` (recorded, never dials, answers what the cell hands it). A cell prints
one JSON object on its last line; that object is what it asserts about.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import tempfile

import pytest

FRONTEND_SRC = pathlib.Path(__file__).resolve().parent.parent / "strands_robots" / "dashboard" / "frontend" / "src"
LIB = FRONTEND_SRC / "lib"

_RELATIVE_IMPORT = re.compile(r"""(from\s+['"])(\.{1,2}/[^'"]+?)(['"])""")

#: The first Node release of each line that strips TypeScript types without a flag:
#: 23.6 on the current line, backported to 22.18 on the LTS line. An older Node
#: (20 LTS, early 22) refuses ``.ts`` with ``ERR_UNKNOWN_FILE_EXTENSION``, so a
#: cell on such a machine must skip, not fail.
STRIPS_TYPES_FROM = ((22, 18), (23, 6))


def node_version() -> tuple[int, int] | None:
    """``(major, minor)`` of the ``node`` on PATH, or None when there is none or it does not answer."""
    node = shutil.which("node")
    if node is None:
        return None
    try:
        proc = subprocess.run(
            [node, "--version"], capture_output=True, text=True, encoding="utf-8", timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.match(r"v(\d+)\.(\d+)", proc.stdout.strip())
    if proc.returncode != 0 or match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def node_strips_types(version: tuple[int, int] | None) -> bool:
    """Whether that Node runs a ``.ts`` module natively (22.18+ on the 22 line, 23.6+ after)."""
    if version is None:
        return False
    major, minor = version
    if major == 22:
        return (major, minor) >= STRIPS_TYPES_FROM[0]
    return (major, minor) >= STRIPS_TYPES_FROM[1]


def _node_skip_reason() -> str:
    version = node_version()
    if version is None:
        return "node is not on this machine"
    return f"node {version[0]}.{version[1]} cannot strip TypeScript types; 22.18 or 23.6 and later run the cells"


requires_node = pytest.mark.skipif(not node_strips_types(node_version()), reason=_node_skip_reason())

#: The browser a lib module runs in, as far as any lib module reads it.
BROWSER = """
const _store = new Map()
globalThis.localStorage = {
  getItem: k => (_store.has(k) ? _store.get(k) : null),
  setItem: (k, v) => { _store.set(k, String(v)) },
  removeItem: k => { _store.delete(k) },
  key: i => [..._store.keys()][i] ?? null,
  get length() { return _store.size },
  clear: () => { _store.clear() },
}
globalThis.replaced = []
globalThis.history = { replaceState: (_s, _t, url) => { globalThis.replaced.push(url) } }
globalThis.sent = []
globalThis.answer = { status: 200, headers: {}, body: '{}' }
globalThis.fetch = async (url, init = {}) => {
  globalThis.sent.push({ url: String(url), headers: { ...(init.headers ?? {}) }, method: init.method ?? 'GET' })
  const a = globalThis.answer
  const lower = Object.fromEntries(Object.entries(a.headers ?? {}).map(([k, v]) => [k.toLowerCase(), v]))
  return {
    ok: a.status >= 200 && a.status < 300,
    status: a.status,
    statusText: String(a.status),
    headers: { get: name => lower[name.toLowerCase()] ?? null },
    text: async () => a.body ?? '',
    json: async () => JSON.parse(a.body ?? 'null'),
    blob: async () => new Blob([a.body ?? '']),
  }
}
globalThis.URL.createObjectURL = () => 'blob:stub'
function setLocation(href) {
  const u = new URL(href)
  globalThis.location = { href: u.href, origin: u.origin, host: u.host, hostname: u.hostname,
    pathname: u.pathname, search: u.search, hash: u.hash, protocol: u.protocol }
}
globalThis.setLocation = setLocation
globalThis.out = obj => { console.log(JSON.stringify(obj)) }
"""


def _lib_copy(into: pathlib.Path) -> None:
    for src in LIB.glob("*.ts"):
        text = src.read_text(encoding="utf-8")
        text = _RELATIVE_IMPORT.sub(lambda m: f"{m.group(1)}{m.group(2)}.ts{m.group(3)}", text)
        (into / src.name).write_text(text, encoding="utf-8")


def run_frontend(script: str, *, page: str = "http://robot.lan:8090/") -> dict:
    """Run ``script`` under Node beside a copy of ``src/lib`` and return its last JSON line.

    ``page`` is the address bar when the script starts; the script may call
    ``setLocation(href)`` before it imports a module to change it. Modules are
    imported by the script with ``await import('./endpoints.ts')`` so the cell
    decides what state exists before the module's first read.
    """
    node = shutil.which("node")
    assert node is not None and node_strips_types(node_version()), (
        "run_frontend needs a node that strips TypeScript types; guard the cell with requires_node"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _lib_copy(root)
        (root / "cell.mjs").write_text(f"{BROWSER}\nsetLocation({json.dumps(page)})\n{script}\n", encoding="utf-8")
        proc = subprocess.run(
            [node, "--no-warnings", "cell.mjs"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
    assert proc.returncode == 0, f"the cell failed under node:\n{proc.stderr}\n{proc.stdout}"
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"the cell printed nothing:\n{proc.stderr}"
    return json.loads(lines[-1])
