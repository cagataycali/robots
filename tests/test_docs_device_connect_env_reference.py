"""Grade the Device Connect environment reference against the code's env surface.

``docs/reference/configuration.md`` is where an operator configures Device Connect: the
``{{env_vars}}`` token on it renders, through ``docs/hooks/env_vars.py``, one table per
prefix of every variable the package reads. The page is graded as the reader sees it,
with the token expanded by the hook. Two properties are graded here,
both derived from the package rather than from a list kept alongside it, so a
variable added later is graded on arrival:

- **Every variable the surface reads is documented.** A variable the code
  honours but the page omits is a setting nobody can find. That is how
  ``REACHY_DAEMON_TLS`` - the only knob that encrypts the Reachy Mini daemon
  link - stayed undocumented while ``REACHY_DAEMON_TOKEN``, the credential that
  link carries, was listed. An operator who read the page set the token,
  believed the link was protected, and got ``Bearer <token>`` over ``http://``
  and ``ws://``: authenticated, and readable by anyone on the segment.
- **The variables of one link are documented in one table.** ``ReachyMiniDriver``
  reaches its robot over the daemon's own REST/WebSocket interface, and three
  variables secure it: one authenticates, one encrypts, one weakens
  verification. Splitting them across sections - the token under "Other", the
  encryption knob nowhere - is what let the reader configure half a posture.
  A reader configuring that link should find its variables together.

The behaviour those rules exist to make discoverable is asserted directly too:
with only the documented token set, both daemon URLs are plaintext and carry the
bearer credential. That is a fact about the code, not the page, so it holds on
both sides of this change - it is the reason the omission mattered.
"""

import ast
import importlib.util
import pathlib
import re
import sys

import pytest

import strands_robots
from tests._package_ast import parse_file

_REPO = pathlib.Path(strands_robots.__file__).parent.parent
_PACKAGE = _REPO / "strands_robots" / "device_connect"
#: The daemon link module, which moved beside the native Reachy driver.
_DAEMON_LINK = _REPO / "strands_robots" / "drivers" / "reachy_transport.py"
_PAGE = _REPO / "docs" / "reference" / "configuration.md"
_HOOK = _REPO / "docs" / "hooks" / "env_vars.py"

#: The module that owns the Reachy Mini daemon link. Its variables configure one
#: channel, so the reference documents them together.
_DAEMON_LINK_MODULE = "reachy_transport"

#: A documented row's first cell: exactly one env-var-shaped token, in the
#: backticks a hand-written row uses or the ``<code>`` the hook emits.
_ROW_VAR = re.compile(r"^\|\s*(?:`|<code>)([A-Z][A-Z0-9_]*)(?:`|</code>)\s*\|")

#: The page states the accepted boolean spellings once, for every boolean row:
#: "A boolean variable accepts `1`, `true`, `yes` (case-insensitive) ...".
_BOOLEAN_RULE = re.compile(r"boolean variable accepts ((?:`[A-Za-z0-9]+`,\s*)*`[A-Za-z0-9]+`)")


def _rendered_page() -> str:
    """The configuration page with ``{{env_vars}}`` expanded by the shipped hook."""
    spec = importlib.util.spec_from_file_location("docs_hooks_env_vars", _HOOK)
    assert spec is not None and spec.loader is not None
    module = sys.modules.get(spec.name) or importlib.util.module_from_spec(spec)
    if spec.name not in sys.modules:
        sys.modules[spec.name] = module  # dataclasses in the hook resolve their module here
        spec.loader.exec_module(module)
    source = _PAGE.read_text(encoding="utf-8")
    rendered = module.on_page_markdown(source, page=None, config=None, files=None)
    assert rendered != source, "configuration.md carries no {{env_vars}} token for the hook to expand"
    return rendered


def _env_reads() -> dict[str, set[str]]:
    """Map each env var the Device Connect surface reads to the modules reading it.

    Returns:
        ``{"VAR": {"module_stem", ...}}`` for every literal ``os.getenv`` /
        ``os.environ.get`` / ``os.environ[...]`` key under the package.
    """
    found: dict[str, set[str]] = {}
    for path in [*sorted(_PACKAGE.rglob("*.py")), _DAEMON_LINK]:
        tree = parse_file(path)
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Call) and ast.unparse(node.func) in (
                "os.getenv",
                "os.environ.get",
            ):
                if node.args and isinstance(node.args[0], ast.Constant):
                    name = node.args[0].value
            elif isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
                if isinstance(node.slice, ast.Constant):
                    name = node.slice.value
            if isinstance(name, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                found.setdefault(name, set()).add(path.stem)
    return found


def _documented_rows(page: str) -> dict[str, str]:
    """Map each variable the page documents to the heading it sits under.

    Args:
        page: The markdown source to read.

    Returns:
        ``{"VAR": "heading text"}`` for every table row naming a variable.
    """
    rows: dict[str, str] = {}
    heading = ""
    for line in page.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            continue
        match = _ROW_VAR.match(line)
        if match:
            rows.setdefault(match.group(1), heading)
    return rows


def _undocumented(page: str) -> list[str]:
    """Return the variables the surface reads that *page* does not document."""
    return sorted(set(_env_reads()) - set(_documented_rows(page)))


def _daemon_link_sections(page: str) -> dict[str, str]:
    """Return the heading each daemon-link variable is documented under.

    A variable the page omits entirely maps to ``""`` so the caller sees one
    verdict for "documented elsewhere" and "not documented at all".
    """
    rows = _documented_rows(page)
    return {var: rows.get(var, "") for var, modules in _env_reads().items() if _DAEMON_LINK_MODULE in modules}


def _row_for(page: str, var: str) -> str:
    """Return the table row documenting *var*, or ``""`` when absent."""
    for line in page.splitlines():
        match = _ROW_VAR.match(line)
        if match and match.group(1) == var:
            return line
    return ""


def _documented_spellings(page: str, var: str) -> list[str]:
    """Return the accepted spellings the page advertises for the boolean *var*.

    The generated table does not repeat the vocabulary per row; the page states
    it once in its boolean rule, and the row has to describe the variable as a
    truthy switch for that rule to apply to it.
    """
    row = _row_for(page, var)
    if "truthy" not in row:
        return []
    rule = _BOOLEAN_RULE.search(page)
    return re.findall(r"`([A-Za-z0-9]+)`", rule.group(1)) if rule else []


@pytest.fixture(scope="module")
def page() -> str:
    """The shipped configuration reference, as rendered."""
    return _rendered_page()


class TestTheReferenceCoversTheSurface:
    """Every variable the code honours is documented, and one link reads as one."""

    def test_every_device_connect_env_var_is_documented(self, page: str) -> None:
        """A variable the surface reads but the page omits is unreachable config."""
        missing = _undocumented(page)
        assert not missing, (
            "docs/reference/configuration.md documents no row for environment variables the "
            f"Device Connect surface reads: {missing}. A variable the code honours "
            "and the reference omits cannot be found by the operator who needs it."
        )

    def test_the_daemon_link_variables_are_documented_in_one_table(self, page: str) -> None:
        """The three variables securing one link belong under one heading."""
        sections = _daemon_link_sections(page)
        assert len(set(sections.values())) == 1, (
            "the variables that secure the Reachy Mini daemon link are documented "
            f"under different headings (empty = not documented at all): {sections}. "
            "One link's variables belong in one table: the token authenticates and "
            "REACHY_DAEMON_TLS encrypts, so a reader who finds only one of them "
            "configures half a posture."
        )

    def test_the_documented_tls_spellings_really_upgrade_the_link(self, page: str) -> None:
        """Each spelling the row advertises turns the daemon link into TLS."""
        from strands_robots.drivers import reachy_transport

        spellings = _documented_spellings(page, "REACHY_DAEMON_TLS")
        assert spellings, (
            "REACHY_DAEMON_TLS' row advertises no accepted spelling, so a reader cannot tell what value enables TLS."
        )
        for spelling in spellings:
            for value in (spelling, spelling.upper()):
                with pytest.MonkeyPatch.context() as patch:
                    patch.setenv("REACHY_DAEMON_TLS", value)
                    assert reachy_transport._daemon_use_tls() is True, (
                        f"the row advertises {spelling!r} but REACHY_DAEMON_TLS={value!r} "
                        "leaves the daemon link on plaintext http:// / ws://"
                    )


class TestThePlaintextDefaultIsWhyTheOmissionMatters:
    """Facts about the code, true on both sides of the documentation change."""

    def test_the_token_alone_leaves_the_credential_in_cleartext(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting only the documented variable ships the token over plaintext."""
        from strands_robots.drivers import reachy_transport

        monkeypatch.setenv("REACHY_DAEMON_TOKEN", "s3cr3t")
        monkeypatch.delenv("REACHY_DAEMON_TLS", raising=False)

        assert reachy_transport._daemon_auth_token() == "s3cr3t"
        assert reachy_transport._http_scheme() == "http"
        assert reachy_transport._ws_scheme() == "ws"

    def test_a_spelling_the_row_does_not_document_leaves_the_link_plaintext(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The accepted vocabulary is closed, so the row can enumerate it."""
        from strands_robots.drivers import reachy_transport

        monkeypatch.setenv("REACHY_DAEMON_TLS", "enabled")
        assert reachy_transport._daemon_use_tls() is False

    def test_certificates_are_verified_until_the_operator_opts_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verification is the default; the documented opt-out is what drops it."""
        from strands_robots.drivers import reachy_transport

        monkeypatch.delenv("REACHY_DAEMON_TLS_INSECURE", raising=False)
        assert reachy_transport._daemon_verify_tls() is True
        monkeypatch.setenv("REACHY_DAEMON_TLS_INSECURE", "1")
        assert reachy_transport._daemon_verify_tls() is False


class TestTheRulesAreScopedAndLoadBearing:
    """Controls: the scan reaches the package, and each rule reports a plant."""

    def test_the_scan_reaches_every_device_connect_module(self) -> None:
        """A scan that stopped finding variables would report a clean page."""
        reads = _env_reads()
        assert len(reads) >= 5, f"expected the package to read several variables, found {reads}"
        daemon = _daemon_link_sections(_rendered_page())
        assert len(daemon) >= 3, f"expected the daemon link to carry several variables, found {daemon}"

    def test_the_messaging_variables_are_not_required_in_the_daemon_link_table(self, page: str) -> None:
        """The one-table rule is scoped to the module that owns the daemon link.

        ``MESSAGING_BACKEND`` and ``DEVICE_CONNECT_ALLOW_INSECURE`` configure the
        Device Connect messaging transport, a different channel, and are read
        outside that module - so they are graded for presence, not placement.
        """
        daemon = _daemon_link_sections(page)
        assert "MESSAGING_BACKEND" not in daemon
        assert "DEVICE_CONNECT_ALLOW_INSECURE" not in daemon
        rows = _documented_rows(page)
        assert "MESSAGING_BACKEND" in rows
        assert "DEVICE_CONNECT_ALLOW_INSECURE" in rows

    def test_a_planted_page_missing_a_variable_is_reported(self, page: str) -> None:
        """Dropping a documented row must be reported, not tolerated."""
        var = sorted(_env_reads())[0]
        stripped = "\n".join(
            line for line in page.splitlines() if (m := _ROW_VAR.match(line)) is None or m.group(1) != var
        )
        assert var in _undocumented(stripped)

    def test_a_page_splitting_the_daemon_link_is_reported(self) -> None:
        """The one-table rule reports a split page and accepts a joined one.

        Built here rather than by mutating the shipped page, so the plant states
        the rule on its own and does not depend on what the page happens to say.
        """
        daemon = sorted(_daemon_link_sections(_rendered_page()))
        assert len(daemon) >= 2, f"expected several daemon-link variables, found {daemon}"
        header = "| Variable | Default | What it does |\n|---|---|---|\n"
        rest = "".join(f"| `{var}` | unset | what it does |\n" for var in daemon[1:])
        first = f"| `{daemon[0]}` | unset | what it does |\n"

        split = f"#### Transport security\n\n{header}{rest}\n#### Other\n\n{header}{first}"
        assert len(set(_daemon_link_sections(split).values())) > 1

        joined = f"#### Transport security\n\n{header}{rest}{first}"
        assert len(set(_daemon_link_sections(joined).values())) == 1
