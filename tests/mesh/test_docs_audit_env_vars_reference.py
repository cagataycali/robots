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
import logging
import pathlib
import re

import pytest

import strands_robots
from strands_robots.mesh import audit

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
                value = node.args[0].value
                if isinstance(value, str):
                    name = value
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
            if isinstance(node.slice, ast.Constant):
                value = node.slice.value
                if isinstance(value, str):
                    name = value
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


# --- The claims the section makes, graded against the module ----------------
#
# The four rules above grade whether each variable is *named*.  They cannot see
# whether what the page says *about* a variable is true, and two of the first
# section's claims were not: it promised that a process which cannot open the
# log "refuses to start the auditor, rather than running with the audit trail
# silently off" (the inverse of ``log_safety_event``'s documented fail-soft
# contract), and it named the persisted HMAC field ``hmac`` (the module writes
# ``sig``).  Both survived a green suite because a presence rule is satisfied by
# a bullet that mentions the variable, whatever the bullet asserts.
#
# The rules below close that gap on the two axes an operator or an external
# verifier actually builds against:
#
# - **the persisted record schema**, derived from the module's own dataflow, so
#   a field name in the page that no record carries fails here rather than in
#   somebody's SIEM rule; and
# - **the failure posture**, so the page cannot promise fail-closed behaviour
#   for a writer whose contract is fail-soft.
#
# The posture rules are scoped to this one section deliberately: ``refuses to
# start`` appears twice elsewhere in ``docs/security.md`` and once in
# ``docs/rtps-integration.md`` about ``HardwareRtpsBridge``, where it is
# accurate - that bridge really does refuse to construct without DDS Security
# material.  A page-wide phrase rule would flag those true claims, so the
# scope is the audit section and the derivation is the audit writer.

#: Refusal-to-start phrasings.  The negative rule catches the specific false
#: assurance; the positive rule (the section must describe the swallow) is the
#: half that survives a rewording, since a bullet reinstating a fail-closed
#: claim has to delete the swallow sentence to be coherent.
_REFUSAL_TO_START_PATTERNS = (
    r"refus\w*\s+to\s+start",
    r"refus\w*\s+to\s+begin",
    r"(?:will|does|would)\s+not\s+start",
)


def _audit_section() -> str:
    """Return the text of the ``Audit log`` section of ``docs/security.md``.

    Returns:
        Every line from the ``## Audit log`` heading up to the next ``## ``
        heading.  Scoping the posture rules to this slice keeps them from
        flagging the accurate ``refuses to start`` claims the same page makes
        about the RTPS bridge.
    """
    lines = _PAGE.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if re.match(r"^##\s+", line):
            if inside:
                break
            inside = line.strip() == f"## {_AUDIT_HEADING}"
            continue
        if inside:
            out.append(line)
    return "\n".join(out)


def _log_safety_event_ast() -> ast.FunctionDef:
    """Return the ``log_safety_event`` definition from ``mesh/audit.py``.

    Returns:
        The :class:`ast.FunctionDef` for the audit writer, which is the single
        function that decides both the persisted record schema and what happens
        when the destination cannot be written.
    """
    tree = ast.parse(_AUDIT_MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "log_safety_event":
            return node
    raise AssertionError("mesh/audit.py no longer defines log_safety_event")


def _record_field_names() -> set[str]:
    """Every literal field name the audit writer puts into a record.

    Returns:
        The keys of the ``record`` dict literal plus every literal key assigned
        through ``record[...] = ...``, which together are the persisted JSONL
        schema an external verifier reads.
    """
    found: set[str] = set()
    for node in ast.walk(_log_safety_event_ast()):
        # ``record: dict[str, Any] = {...}`` is an AnnAssign, so a walk that
        # only reads Assign finds the ``record[...] = ...`` additions and none
        # of the envelope the writer starts from.
        if isinstance(node, ast.AnnAssign):
            targets: list[ast.expr] = [node.target]
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            continue
        if isinstance(node.value, ast.Dict):
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "record":
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            found.add(key.value)
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "record"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                found.add(target.slice.value)
    return found


def _signature_field_name() -> str:
    """Return the record field the per-record HMAC is written into.

    Derived by dataflow rather than by name: find the local the
    ``_sign_record`` call binds, then the ``record[...]`` key that local is
    assigned to.  Renaming the field in ``mesh/audit.py`` therefore moves this
    answer, and the documentation rule below follows it.

    Returns:
        The field name, e.g. ``"sig"``.
    """
    fn = _log_safety_event_ast()
    signed_locals: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "_sign_record"
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    signed_locals.add(target.id)
    assert signed_locals, "mesh/audit.py no longer binds the result of _sign_record"

    fields: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Name) and node.value.id in signed_locals):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "record"
                and isinstance(target.slice, ast.Constant)
            ):
                fields.add(str(target.slice.value))
    assert len(fields) == 1, f"expected exactly one record field assigned from _sign_record, found {sorted(fields)}"
    return fields.pop()


def _fields_the_section_names() -> set[str]:
    """Field names the audit section claims a record carries.

    Returns:
        Every backticked token the section describes as a ``field``, e.g. the
        ``sig`` in "each record's ``sig`` field".
    """
    return set(re.findall(r"`([a-z_][a-z0-9_]*)`\s+field", _audit_section()))


def _audit_log_lines_the_section_quotes() -> set[str]:
    """Log-line fragments the section tells an operator to monitor for.

    Returns:
        Every backticked token in the section that starts with the module's
        ``[audit]`` log prefix.
    """
    return set(re.findall(r"`(\[audit\][^`]*)`", _audit_section()))


@pytest.fixture
def isolated_audit(monkeypatch, tmp_path):
    """Point the audit writer at a scratch directory with no PSK.

    Mirrors the isolation fixture in ``tests/mesh/test_audit_integrity.py``:
    the sequence counters and the per-run PSK fingerprint snapshot are process
    globals, so a behavioural cell that does not reset them inherits whatever
    an earlier test left behind.  Deliberately opt-in rather than autouse - the
    documentation rules above must read the page, not a patched environment.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None
    yield tmp_path
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None


def test_the_section_names_the_field_the_signature_is_written_into():
    """The documented HMAC field is the one the writer assigns the signature to.

    The field name is part of the persisted JSONL schema, so it is what a SIEM
    rule or a forensic script greps for.  The page named ``hmac``; the writer
    assigns ``record["sig"]``.  A checker built from the old wording finds no
    match on a correctly-signed fleet and either alarms forever or concludes
    tamper-evidence is off.
    """
    field = _signature_field_name()
    named = _fields_the_section_names()
    assert field in named, (
        f"``mesh/audit.py`` writes the per-record HMAC into ``record[{field!r}]`` "
        f"but the ``{_AUDIT_HEADING}`` section names {sorted(named)} as record "
        f"fields. Name ``{field}`` where the PSK bullet describes the signature."
    )


def test_every_field_the_section_names_is_one_a_record_carries():
    """No documented field name is absent from the persisted schema.

    The complement of the rule above: naming the right field is not enough if
    the page also names one no record has.  Both halves are needed because the
    page describes the signed and unsigned postures in separate sentences, and
    only one of them has to be wrong to send a verifier looking for a key that
    never appears.
    """
    schema = _record_field_names()
    named = _fields_the_section_names()
    unknown = sorted(named - schema)
    assert not unknown, (
        f"The ``{_AUDIT_HEADING}`` section names {unknown} as record field(s), "
        f"but ``log_safety_event`` writes {sorted(schema)}. A verifier built "
        f"from this page would look for a key no record carries."
    )


def test_the_section_does_not_promise_a_refusal_to_start():
    """The page must not offer fail-closed assurance for a fail-soft writer.

    ``log_safety_event`` swallows write errors by contract, so there is no
    startup refusal to promise: an unusable destination yields a running peer
    with the audit trail off.  Scoped to this section because the same page
    accurately says ``refuses to start`` about the RTPS bridge.
    """
    section = _audit_section()
    hits = [pattern for pattern in _REFUSAL_TO_START_PATTERNS if re.search(pattern, section, re.IGNORECASE)]
    assert not hits, (
        f"The ``{_AUDIT_HEADING}`` section asserts a refusal to start "
        f"(matched {hits}), but ``log_safety_event`` logs write errors at "
        f"WARNING and swallows them - the peer keeps running with the audit "
        f"trail off. Describe that posture instead."
    )


def test_the_section_describes_the_swallowed_write_error():
    """The page states the posture an operator has to plan for.

    The positive half of the rule above, and the half that survives a
    rewording: a bullet that reinstates a fail-closed claim has to remove this
    description to read coherently, so this cell fires on the rewrite even if
    the phrasing dodges the negative patterns.
    """
    section = _audit_section().lower()
    assert "warning" in section, (
        f"The ``{_AUDIT_HEADING}`` section does not mention the WARNING level "
        f"that an audit write failure is reported at, so a reader cannot know "
        f"which signal to monitor for."
    )
    assert "swallow" in section, (
        f"The ``{_AUDIT_HEADING}`` section does not say that audit write "
        f"errors are swallowed, which is the posture ``log_safety_event`` "
        f"documents and the reason a running mesh does not imply a trail."
    )


def test_every_log_line_the_section_quotes_is_one_the_module_emits():
    """A monitoring instruction must name a string the module actually logs.

    The section tells an operator which line to alert on.  If that literal
    drifts from the module's format string, the alert silently never fires -
    the same class of failure as the wrong field name, one layer out.
    """
    quoted = _audit_log_lines_the_section_quotes()
    assert quoted, (
        f"The ``{_AUDIT_HEADING}`` section quotes no ``[audit]`` log line, so "
        f"it tells an operator to monitor for a signal it does not name."
    )
    source = _AUDIT_MODULE.read_text(encoding="utf-8")
    missing = sorted(fragment for fragment in quoted if fragment not in source)
    assert not missing, (
        f"The ``{_AUDIT_HEADING}`` section quotes {missing}, which do not "
        f"appear in ``mesh/audit.py``. An alert built on a line the module "
        f"never emits never fires."
    )


@pytest.mark.parametrize("destination", ["unwritable-parent", "symlink-at-log-path", "log-path-is-a-directory"])
def test_an_unusable_destination_leaves_the_peer_running_with_the_trail_off(
    isolated_audit, monkeypatch, tmp_path, destination, caplog
):
    """The behavioural fact the corrected wording describes.

    Three ways to make the destination unusable, all of which survive the
    writer's own hardening (``_ensure_paths`` re-chmods the parent to 0o700 on
    every call, so merely dropping the write bit is undone).  In each, three
    safety events are issued: nothing raises, nothing is recorded, and the only
    signal is a WARNING per attempt.
    """
    root = tmp_path / destination
    root.mkdir()
    if destination == "unwritable-parent":
        target = root / "auditdir"
    else:
        target = root / "d"
        target.mkdir()
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(target))
    log_path = audit.audit_log_path()
    if destination == "symlink-at-log-path":
        log_path.symlink_to("/dev/null")
    elif destination == "log-path-is-a-directory":
        log_path.mkdir()
    else:
        root.chmod(0o500)

    try:
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.audit"):
            for index in range(3):
                audit.log_safety_event("emergency_stop", "peer-a", {"index": index})
    finally:
        root.chmod(0o700)

    written = [line for line in log_path.read_text().splitlines() if line.strip()] if log_path.is_file() else []
    assert written == [], (
        f"an unusable audit destination ({destination}) wrote {len(written)} "
        f"record(s); the point of this cell is that it writes none"
    )
    assert caplog.records, (
        f"an unusable audit destination ({destination}) recorded nothing at "
        f"WARNING, so the peer would run with the audit trail off and no signal"
    )


def test_a_signed_record_carries_the_documented_field_and_not_the_old_name(isolated_audit, monkeypatch):
    """A written record confirms the schema the derivation reads off the AST.

    With the PSK set from the process's first signed write, the persisted
    record gains exactly the field the page now names - and does not gain the
    name the page used to claim.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "docs-reference-psk")
    audit.log_safety_event("emergency_stop", "peer-b", {"reason": "docs reference"})
    records = audit.read_audit_log()
    assert len(records) == 1, f"expected one audit record, got {len(records)}"
    field = _signature_field_name()
    assert field in records[0], f"signed record {sorted(records[0])} carries no {field!r} field"
    assert len(str(records[0][field])) == 64, "the signature is a SHA-256 hex digest"
    assert "hmac" not in records[0], (
        "a signed record carries no ``hmac`` field; that name was the "
        "documentation error this file's companion change corrects"
    )


def test_the_writer_documents_the_fail_soft_contract_the_section_mirrors():
    """Premise: the swallow the page describes is the module's stated contract.

    If ``log_safety_event`` ever raises on a write failure, the corrected
    wording becomes wrong in the other direction and this cell says so, rather
    than letting the posture rules pin prose against a contract that moved.
    """
    doc = " ".join((audit.log_safety_event.__doc__ or "").split())
    assert "WARNING" in doc and "swallowed" in doc, (
        "``log_safety_event`` no longer documents write errors as logged at "
        "WARNING and swallowed; the audit-log section's posture wording and "
        "the rules above need to be revisited against the new contract."
    )


def test_the_record_schema_derivation_finds_the_documented_envelope():
    """Premise: the AST derivation reads a plausible record schema.

    A derivation that silently returned an empty set would satisfy the
    "no unknown field" rule vacuously, so pin the envelope fields the module
    header narrates.
    """
    schema = _record_field_names()
    expected = {"ts", "event", "peer_id", "payload", "seq"}
    missing = sorted(expected - schema)
    assert not missing, (
        f"the record-schema derivation found {sorted(schema)} and is missing "
        f"{missing}; ``log_safety_event`` may no longer build the record as a "
        f"dict literal, which would make the field rules vacuous"
    )
