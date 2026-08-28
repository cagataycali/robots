"""Grade the mesh audit log environment reference against the code's env surface.

``docs/security.md`` names the credentials operators must configure to secure a
fleet: the transport TLS material, the IoT credentials the AWS backend reads,
and the mesh subscribe allow-list.  It does not, until this file's companion
change, name the environment variables that configure the *audit log itself* -
the JSONL write path, the size and file caps that bound disk use, and the PSK
that turns the log's per-record HMAC on.  A reader who followed the page
believed the transport was hardened and the actions surface was gated, and
never saw the four variables that decide where the audit trail lives and
whether it can be forged.

Two properties are graded here, both derived from the package rather than from
a list kept alongside it - so a variable added to ``mesh/audit.py`` later is
graded on arrival:

- **Every audit variable the module reads is documented.** A variable the
  audit surface honours but the page omits is a setting nobody configuring an
  audited fleet can find. That is the pattern harness#376 tracks across ~58
  ``STRANDS_*`` names; this file pins the four that touch the audit log.
- **The audit variables are documented under one heading.** The four names
  configure one channel (the audit log file) and its integrity check (the
  PSK), so an operator turning on tamper-evidence should find them together,
  not split between "Credentials and secrets" and "Telemetry exposure".

The behavioural fact the rule exists to make discoverable is asserted too:
with the PSK unset, the audit log accepts a record and re-reads it as-is; the
HMAC field is absent and the sequence-restore step trusts the file.  That is
the failure mode - an operator who never saw ``STRANDS_MESH_AUDIT_PSK`` cannot
know the tamper-evidence contract is off - and it holds on both sides of this
change, so it is what the four documentation bullets exist to name.
"""

import ast
import pathlib
import re

import pytest

import strands_robots

_AUDIT_MODULE = pathlib.Path(strands_robots.__file__).parent / "mesh" / "audit.py"
_PAGE = pathlib.Path(strands_robots.__file__).parent.parent / "docs" / "security.md"

#: The prefix every audit-log env var shares.  Reading the module for the
#: literal names lets a fifth variable added later fail this file on arrival.
_AUDIT_PREFIX = "STRANDS_MESH_AUDIT_"

#: The heading the four bullets sit under.  A single heading is the point of
#: rule two: a reader turning on audit-log tamper-evidence finds the file
#: location, the size caps, and the PSK together.
_AUDIT_HEADING = "Audit log"


def _audit_env_reads() -> set[str]:
    """The set of ``STRANDS_MESH_AUDIT_*`` env vars ``mesh/audit.py`` reads.

    Returns:
        Every literal ``os.getenv`` / ``os.environ.get`` / ``os.environ[...]``
        key in ``mesh/audit.py`` that starts with ``STRANDS_MESH_AUDIT_``.
    """
    found: set[str] = set()
    tree = ast.parse(_AUDIT_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        name: str | None = None
        if isinstance(node, ast.Call) and ast.unparse(node.func) in (
            "os.getenv",
            "os.environ.get",
        ):
            if node.args and isinstance(node.args[0], ast.Constant):
                name = node.args[0].value
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
            if isinstance(node.slice, ast.Constant):
                name = node.slice.value
        if isinstance(name, str) and name.startswith(_AUDIT_PREFIX):
            found.add(name)
    return found


def _documented_audit_vars() -> dict[str, str]:
    """Map each documented ``STRANDS_MESH_AUDIT_*`` var to its section heading.

    Returns:
        ``{"VAR": "heading text"}`` for every backticked ``STRANDS_MESH_AUDIT_*``
        token that appears in ``docs/security.md``, keyed to the nearest
        preceding ``## `` or ``### `` heading.  A variable named twice under
        different headings takes its first mention, because that is the entry
        the reader lands on when they follow the page top to bottom.
    """
    text = _PAGE.read_text(encoding="utf-8")
    mapping: dict[str, str] = {}
    current = ""
    var_pattern = re.compile(r"`(" + re.escape(_AUDIT_PREFIX) + r"[A-Z0-9_]+)`")
    for line in text.splitlines():
        heading = re.match(r"^#{2,3}\s+(.+?)\s*$", line)
        if heading:
            current = heading.group(1).strip()
            continue
        for match in var_pattern.finditer(line):
            name = match.group(1)
            mapping.setdefault(name, current)
    return mapping


def test_every_audit_env_var_is_documented():
    """Every ``STRANDS_MESH_AUDIT_*`` var the module reads has a docs entry.

    This is the rule that fires when a knob lands in ``mesh/audit.py`` without
    a corresponding mention in ``docs/security.md``.  It reads the module's AST
    for the literal keys rather than trusting an authored list, so no future
    variable is silently omitted.
    """
    read = _audit_env_reads()
    documented = _documented_audit_vars()
    missing = sorted(read - documented.keys())
    assert not missing, (
        f"``mesh/audit.py`` reads {sorted(read)} but ``docs/security.md`` "
        f"names {sorted(documented)}. Undocumented: {missing}. "
        f"Add a bullet under the ``{_AUDIT_HEADING}`` heading naming each."
    )


def test_audit_env_vars_are_documented_together():
    """All documented ``STRANDS_MESH_AUDIT_*`` vars sit under one heading.

    Rule two: the four names configure one channel.  Splitting them across
    sections is what let ``REACHY_DAEMON_TLS`` sit under a different heading
    from the token it carried, leaving a reader who configured half a posture.
    The audit log is one channel and its variables belong together.
    """
    documented = _documented_audit_vars()
    if not documented:
        pytest.fail(
            f"No ``STRANDS_MESH_AUDIT_*`` variables are documented in "
            f"``docs/security.md``. Expected an ``{_AUDIT_HEADING}`` section."
        )
    headings = {heading for heading in documented.values() if heading}
    assert len(headings) == 1, (
        f"``STRANDS_MESH_AUDIT_*`` variables are split across headings "
        f"{sorted(headings)}. An operator turning on the audit log expects "
        f"one section listing every knob; splitting them lets a reader who "
        f"followed the page configure a partial posture."
    )


def test_audit_env_vars_sit_under_the_named_heading():
    """The one heading is the ``{_AUDIT_HEADING}`` heading.

    Naming the heading pins the section anchor: a reader following a link,
    a search, or a table of contents lands on the same title the module's
    module-level docstring references.  Without this pin, rules one and two
    could be satisfied by a section titled "Notes" that the page does not
    otherwise treat as the audit-log reference.
    """
    documented = _documented_audit_vars()
    if not documented:
        pytest.skip("no audit vars documented yet; rule one names the omission")
    headings = {heading for heading in documented.values() if heading}
    assert _AUDIT_HEADING in headings, (
        f"``STRANDS_MESH_AUDIT_*`` variables sit under {sorted(headings)}, "
        f"not under a heading named ``{_AUDIT_HEADING}``. The module's "
        f"docstring references the audit log as a discrete channel; the "
        f"page's heading should match."
    )


def test_audit_variables_the_module_reads_include_the_four_known_names():
    """Pin the four names ``mesh/audit.py`` reads today.

    A premise cell: if this fails, the audit module stopped reading one of the
    four names the docstring narrates, and the documentation rule above may
    now name a variable the code no longer reads.  Splitting the premise from
    the rule keeps a passing rule from resting on a stale assumption.
    """
    read = _audit_env_reads()
    known = {
        "STRANDS_MESH_AUDIT_DIR",
        "STRANDS_MESH_AUDIT_PSK",
        "STRANDS_MESH_AUDIT_MAX_BYTES",
        "STRANDS_MESH_AUDIT_MAX_FILES",
    }
    missing = sorted(known - read)
    assert not missing, (
        f"``mesh/audit.py`` no longer reads {missing}. Update this test's "
        f"``known`` set and the docstrings in ``docs/security.md`` accordingly."
    )
