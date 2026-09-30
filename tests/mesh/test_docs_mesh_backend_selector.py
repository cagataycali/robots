"""The documented mesh configuration must name the transport selector.

``STRANDS_MESH_BACKEND`` is the variable that decides which transport a fleet
runs on. :mod:`strands_robots.mesh._backend_select` owns its vocabulary, and
two readers consult it: ``session._backend_choice`` on every publish path, and
``transport.factory.get_transport`` once that verdict is ``iot`` or ``bridge``.

The documented half was the dependency, not the selector. ``[mesh-iot]``
appears in four pages and ``STRANDS_MESH_BACKEND`` appeared in none, while
``docs/reference/security/mesh.md`` stated the mechanism as "Adding the ``[mesh-iot]`` extra
routes traffic through AWS IoT Core". Installing the extra routes nothing: the
default is ``zenoh`` and stays ``zenoh`` until this variable is set. So a
reader who followed the page installed the dependency, believed the fleet was
on IoT, and got Zenoh - and the one variable that would have moved it was not
in the matrix README.md's own IoT section points them at (docs/reference/configuration.md).

That is the same shape ``tests/test_docs_device_connect_env_reference.py``
exists for, where ``REACHY_DAEMON_TLS`` - the knob that encrypts a link - was
undocumented while the credential that link carries was listed, and a reader
configured half a posture. A selector documented only by its dependency is half
a configuration.

The new docs tree documents the selector in three shapes: the generated
matrix on ``docs/reference/configuration.md`` (the ``{{env_vars}}`` token that
``docs/hooks/env_vars.py`` expands, read here through that hook so the page is
graded as built, its first cell a ``<code>`` span), the "Three switches" row on
``docs/learn/mesh/index.md``, and the "Three backends" table on
``docs/learn/mesh/bridges.md`` whose header cell is the variable and whose body
rows are its values. The extra is named in prose as ```mesh-iot``` (README) as
well as ``[mesh-iot]``, so both spellings count as naming it.

Three properties are graded, all derived from
:mod:`strands_robots.mesh._backend_select` rather than from a list kept beside
it, so a fourth transport is graded the hour it lands:

- **The selector is documented, in the matrix the IoT section points at.**
- **The documented spellings are exactly the vocabulary the resolver accepts.**
  A row advertising a value the resolver refuses sends a reader to a silent
  fallback; a value the resolver accepts and the row omits is a transport
  nobody can find.
- **No page states the extra as the thing that routes traffic.** A paragraph
  that names the extra and makes a routing claim has to name the selector too,
  or it reproduces the reading this guard exists to close.

The behaviour those rules exist to make discoverable is asserted directly too:
with the extra's dependency importable and the variable unset, the resolver
still answers ``zenoh``. That is a fact about the code, not about any page, so
it holds on both sides of this change - it is the reason the omission mattered.

Scope. ``tests/test_docs_mesh_join_is_opt_in.py`` and
``tests/mesh/test_mesh_env_opt_in_documented_default.py`` grade ``STRANDS_MESH``,
which decides whether a robot joins a mesh at all - a different variable and a
different question from which transport it joins over. Neither reads this one.

Deliberately not graded: the other nineteen ``STRANDS_MESH_*`` variables the
package reads and no page documents. They are rate, size and path knobs *within*
a transport, and which of them are public API is a decision this guard should
not make. The selector is separable because it is the only one of them whose
accepted values the package enumerates in a module constant, which is what lets
a rule about its documented spellings be derived rather than restated.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from strands_robots.mesh import _backend_select
from tests._docs_hooks import docs_hook

_ENV = _backend_select.BACKEND_ENV_VAR
_BACKENDS = _backend_select.BACKENDS
_DEFAULT = _backend_select.DEFAULT_BACKEND

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page whose IoT section tells the reader to consult the configuration
#: matrix for the ``STRANDS_MESH_*`` knobs, so the selector has to be in it.
_MATRIX_PAGE = "docs/reference/configuration.md"

#: The extra that installs the dependency ``iot`` and ``bridge`` need. Naming it
#: is not a routing claim; naming it *as* the routing mechanism is.
_EXTRA = re.compile(r"\[mesh-iot\]|`mesh-iot`")

#: Verbs a paragraph uses to claim traffic moves somewhere.
_ROUTING_CLAIM = re.compile(r"\brout(?:e|es|ing)\b|\bswitch(?:es)?\b|\bsends? traffic\b", re.I)

#: A single-variable table row: first cell is exactly one variable, in the
#: backticks a hand-written row uses or the ``<code>`` the hook emits.
_ROW_VAR = re.compile(r"^\|\s*(?:`|<code>)([A-Z][A-Z0-9_]*)(?:`|</code>)\s*\|")
#: A table separator row, which tells a header row from a body row.
_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")

#: The guard is only meaningful while it still reaches the shipped pages.
_MINIMUM_PAGES = 20
_MINIMUM_PARAGRAPHS = 200


def _pages() -> list[Path]:
    """Every shipped markdown page a reader configures the mesh from."""
    return [_REPO_ROOT / "README.md", *sorted((_REPO_ROOT / "docs").rglob("*.md"))]


def _text(page: Path) -> str:
    """The page as built: the configuration matrix with ``{{env_vars}}`` expanded by its hook."""
    text = page.read_text(encoding="utf-8")
    if str(page.relative_to(_REPO_ROOT)) != _MATRIX_PAGE:
        return text
    module = docs_hook("env_vars")
    assert "{{env_vars}}" in text, f"{_MATRIX_PAGE} must carry the {{{{env_vars}}}} token"
    rendered = module.on_page_markdown(text, page=None, config=None, files=None)
    assert "{{env_vars}}" not in rendered, "the env_vars hook left the token unexpanded"
    return rendered


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _paragraphs(text: str) -> list[str]:
    """Return *text*'s prose paragraphs, with fenced blocks blanked.

    A claim inside a fence is a transcript rather than prose, and blanking
    rather than dropping the lines keeps paragraph boundaries where they are.

    Args:
        text: The markdown source to split.

    Returns:
        One whitespace-collapsed string per paragraph.
    """
    kept: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            kept.append("")
            continue
        kept.append("" if in_fence else line)
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", "\n".join(kept)) if block.strip()]


def _selector_rows() -> list[tuple[str, str, str]]:
    """Return every single-variable table row documenting the selector.

    Returns:
        ``(relative_path, description_cell, default_cell)`` per row.
    """
    rows: list[tuple[str, str, str]] = []
    for page in _pages():
        if not page.exists():
            continue
        lines = _text(page).splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            match = _ROW_VAR.match(stripped)
            if not (match and match.group(1) == _ENV and stripped.endswith("|")):
                continue
            cells = _cells(stripped)
            if len(cells) < 3:
                continue
            relative = str(page.relative_to(_REPO_ROOT))
            if index + 1 < len(lines) and _SEPARATOR.match(lines[index + 1].strip()):
                # A table headed by the variable: its body rows are the values,
                # one per row, and the row marked default is the default.
                body: list[str] = []
                for row in lines[index + 2 :]:
                    if not row.strip().startswith("|"):
                        break
                    body.append(_cells(row)[0])
                default = next((cell for cell in body if "default" in cell.lower()), "")
                rows.append((relative, " ".join(body), default))
            else:
                # A hand-written row is ``| var | description | default |``; the generated
                # matrix is ``| var | read in | default | meaning |`` with ``<code>`` spans.
                folded = [cell.replace("<code>", "`").replace("</code>", "`") for cell in cells]
                default = folded[2] if len(folded) > 3 else folded[-1]
                rows.append((relative, " ".join(folded[1:]), default))
    return rows


def _advertised_spellings(description: str) -> set[str]:
    """Return the lowercase backticked spellings a row's description offers.

    Only bare words are read. A spelling written as an assignment
    (``STRANDS_MESH_BACKEND=iot``) is prose about the variable rather than an
    entry in the value vocabulary, so it is not graded as one.

    Args:
        description: The row's description cell.

    Returns:
        The set of candidate backend values the row names.
    """
    return {token for token in re.findall(r"`([a-z][a-z0-9_]*)`", description)}


def _unnamed_routing_claims(text: str) -> list[str]:
    """Return paragraphs that credit the extra with routing and omit the selector."""
    return [
        para for para in _paragraphs(text) if _EXTRA.search(para) and _ROUTING_CLAIM.search(para) and _ENV not in para
    ]


class TestTheScanReachesThePages:
    """Non-vacuity: a rename or reformat must fail loudly, not report clean."""

    def test_the_pages_are_found(self) -> None:
        found = [page for page in _pages() if page.exists()]
        assert len(found) >= _MINIMUM_PAGES, (
            f"only {len(found)} markdown pages found under docs/ plus {_MATRIX_PAGE}; "
            "the scan no longer reaches the shipped documentation"
        )

    def test_enough_paragraphs_are_scanned(self) -> None:
        total = sum(len(_paragraphs(_text(page))) for page in _pages() if page.exists())
        assert total >= _MINIMUM_PARAGRAPHS, f"only {total} paragraphs parsed; the prose scan reads almost nothing"

    def test_the_extra_is_still_mentioned_somewhere(self) -> None:
        """The routing rule is only meaningful while some page names the extra."""
        mentions = sum(
            1 for page in _pages() if page.exists() for para in _paragraphs(_text(page)) if _EXTRA.search(para)
        )
        assert mentions, "no paragraph mentions the [mesh-iot] extra, so the routing rule grades nothing"


class TestTheSelectorIsDocumented:
    """The variable that chooses the transport is in the matrix."""

    def test_the_selector_has_a_documented_row(self) -> None:
        rows = _selector_rows()
        assert rows, (
            f"no configuration table documents {_ENV}, the variable that selects the mesh "
            f"transport. The [mesh-iot] extra installs the dependency; this variable chooses "
            f"the transport, and without it the fleet stays on {_DEFAULT!r}."
        )

    def test_the_matrix_the_iot_section_points_at_documents_it(self) -> None:
        pages = {page for page, _, _ in _selector_rows()}
        assert _MATRIX_PAGE in pages, (
            f"{_MATRIX_PAGE} documents no {_ENV} row, but its AWS IoT Core section sends the "
            f"reader to that configuration matrix for the STRANDS_MESH_* knobs. Documented in "
            f"{sorted(pages)} instead."
        )


class TestTheDocumentedVocabularyIsTheResolversOwn:
    """Derived from BACKENDS, so a fourth transport is graded on arrival."""

    def test_every_accepted_backend_is_documented(self) -> None:
        for page, description, _ in _selector_rows():
            advertised = _advertised_spellings(description)
            missing = sorted(set(_BACKENDS) - advertised)
            assert not missing, (
                f"{page}'s {_ENV} row omits accepted value(s) {missing}; the resolver accepts "
                f"{list(_BACKENDS)}, so a transport the row does not name is one nobody can find"
            )

    def test_no_documented_spelling_is_refused_by_the_resolver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for page, description, _ in _selector_rows():
            for spelling in sorted(_advertised_spellings(description)):
                monkeypatch.setenv(_ENV, spelling)
                assert _backend_select.select_backend() == spelling, (
                    f"{page}'s {_ENV} row advertises {spelling!r}, but the resolver does not "
                    f"select it - it falls back to {_backend_select.select_backend()!r}"
                )

    def test_the_documented_default_is_the_resolvers_default(self) -> None:
        for page, _, default in _selector_rows():
            assert _DEFAULT in default, (
                f"{page}'s {_ENV} row prints Default {default!r}, but an unset variable resolves to {_DEFAULT!r}"
            )


class TestNoPageCreditsTheExtraWithRouting:
    """The extra installs a dependency; the selector moves the traffic."""

    def test_no_paragraph_claims_the_extra_routes_traffic(self) -> None:
        offenders = [
            (str(page.relative_to(_REPO_ROOT)), para)
            for page in _pages()
            if page.exists()
            for para in _unnamed_routing_claims(_text(page))
        ]
        assert not offenders, (
            f"paragraph(s) credit the [mesh-iot] extra with routing traffic without naming "
            f"{_ENV}: {offenders}. Installing the extra changes no transport - it installs the "
            f"dependency, and the variable selects."
        )


class TestTheDefaultIsWhyTheOmissionMatters:
    """Facts about the resolver, true on both sides of the documentation change."""

    def test_an_unset_variable_selects_the_lan_transport(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The extra's dependency being importable does not move the transport."""
        monkeypatch.delenv(_ENV, raising=False)
        assert _backend_select.select_backend() == _DEFAULT

    @pytest.mark.parametrize("raw", ["iot", "IOT", " iot ", "Bridge", "ZENOH"])
    def test_case_and_whitespace_are_normalised(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(_ENV, raw)
        assert _backend_select.select_backend() == raw.strip().lower()

    @pytest.mark.parametrize("raw", ["iott", "mqtt", "aws", ""])
    def test_an_unrecognized_value_falls_back_and_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
    ) -> None:
        """A typo does not crash the host, and it does not pass silently either."""
        monkeypatch.setenv(_ENV, raw)
        with caplog.at_level("WARNING", logger=_backend_select.__name__):
            assert _backend_select.select_backend() == _DEFAULT
        messages = [record.getMessage() for record in caplog.records]
        assert any(_ENV in message and repr(raw) in message for message in messages), (
            f"{_ENV}={raw!r} fell back to {_DEFAULT!r} without a report naming it: {messages}"
        )


class TestTheGuardWouldCatchARegression:
    """Constructed exemplars: the shipped corpus is clean, so grade both outcomes."""

    def test_a_paragraph_crediting_the_extra_is_reported(self) -> None:
        planted = "Adding the `[mesh-iot]` extra routes traffic through AWS IoT Core.\n"
        assert _unnamed_routing_claims(planted), "the routing rule does not report the claim it exists for"

    def test_the_same_paragraph_naming_the_selector_is_not_reported(self) -> None:
        planted = (
            f"The `[mesh-iot]` extra installs the dependency, and `{_ENV}=iot` routes traffic through AWS IoT Core.\n"
        )
        assert not _unnamed_routing_claims(planted), (
            "a paragraph that names the selector is still reported, so the rule grades the extra "
            "rather than the missing selector"
        )

    def test_a_claim_inside_a_fence_is_not_reported(self) -> None:
        planted = "```\nAdding the `[mesh-iot]` extra routes traffic through AWS IoT Core.\n```\n"
        assert not _unnamed_routing_claims(planted), "a transcript inside a fence is graded as prose"

    def test_a_paragraph_mentioning_the_extra_without_a_routing_claim_is_not_reported(self) -> None:
        planted = "Install the `[mesh-iot]` extra to pull in `awsiotsdk`, `awscrt` and `boto3`.\n"
        assert not _unnamed_routing_claims(planted), (
            "naming the extra is reported even with no routing claim, so the rule is not scoped to the mechanism claim"
        )

    def test_a_row_omitting_a_backend_is_reported(self) -> None:
        advertised = _advertised_spellings("Mesh transport: `zenoh` or `iot`")
        assert set(_BACKENDS) - advertised, (
            "a row naming only two of the accepted backends reads as complete, so the "
            "vocabulary rule cannot see a missing transport"
        )

    def test_a_table_headed_by_the_variable_reads_its_body_rows_as_the_vocabulary(self) -> None:
        """The bridges page shape: header cell is the variable, one value per body row."""
        rows = [row for row in _selector_rows() if row[0] == "docs/learn/mesh/bridges.md"]
        assert rows, "premise: docs/learn/mesh/bridges.md no longer heads a table with the selector"
        _, description, default = rows[0]
        assert _advertised_spellings(description) == set(_BACKENDS)
        assert "default" in default.lower() and _DEFAULT in default

    def test_an_assignment_spelling_is_not_read_as_a_value(self) -> None:
        """`VAR=iot` is prose about the variable, not an entry in the vocabulary."""
        assert _advertised_spellings(f"set `{_ENV}=iot` to switch") == set(), (
            "an assignment is read as a documented value, so a row could satisfy the "
            "vocabulary rule without naming any bare spelling"
        )
