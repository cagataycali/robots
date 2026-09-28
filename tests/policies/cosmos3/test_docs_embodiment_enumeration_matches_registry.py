"""``docs/learn/policies/cosmos3.md`` enumerates the embodiments the provider registers.

The cosmos3 provider is the one policy whose behaviour is selected by a second
name: ``create_policy("cosmos3", embodiment=...)``. That name picks the
conditioning domain, the action width and the column layout, so the set of
accepted embodiments is a public API surface in its own right - and it is
enumerated by hand on four surfaces. Two sit on the pages a reader consults:
the enumeration sentence that opens the provider page's ``## Embodiments``
section, and the cosmos3 row of the provider matrix that ``docs/hooks/providers.py``
renders into ``docs/learn/policies/index.md`` from ``registry/policies.json``
(the row's "what it drives" cell carries the list in a parenthetical). Two more
were found by sweeping for the class rather than by reading the provider page,
and are graded here for the same reason: the ``Available embodiments:`` sentence
in the package docstring, one import from the registry and what ``help()``
prints; and the parenthetical in
:class:`~strands_robots.policies.cosmos3.policy.Cosmos3Policy`'s ``embodiment:``
``Args:`` entry, which is the accepted-value list for the parameter a caller
passes.

The two docstring surfaces are graded through ``__doc__`` rather than by
reading the source, so a reflow or a moved definition cannot disarm them and
the graded text is exactly what a reader is shown. The matrix row is graded
through the hook that renders it, so the page's token is never read as prose.

The old provider page carried five more enumerations (front-matter description,
an ``## Embodiments`` table, an inline ``# droid | umi | ...`` comment, an
``Embodiments:`` quickstart paragraph with per-embodiment numbers, and the
domain/raw-dim/bundled-stats table of the in-process backend page). The new
tree keeps one list per page, so those surfaces have no successor and their
graders were retired; the sentence grader below grades both directions in
their place.

Nothing tied any of those to
:data:`~strands_robots.policies.cosmos3.embodiments.EMBODIMENTS`. The
provider-level catalogue is guarded - ``tests/test_docs_policy_coverage.py``
ties the overview table to ``policies.json`` - but that guard grades
*providers*, so an embodiment added to an existing provider is graded by
nothing, and a reader is told the accepted set is smaller than it is. The
failure is silent in the direction that matters: the code accepts the new name,
so no call breaks, and only the documentation disagrees.

The companion guard in this directory,
``tests/policies/cosmos3/test_documented_backend_knob_routes.py``, already
applies the same rule to this page's *keywords*: it grades them against
``inspect.signature`` "rather than against a copied list, so ... a newly
documented one is graded without touching this file". This file applies that
rule to the page's embodiment names.

Every grader below is a pure function of ``(page text, registry)`` so
:class:`TestTheGradersAreNotVacuous` can hand them a registry carrying an
embodiment the page cannot mention and assert each one reports it. A grader
that cannot see a missing embodiment would report a clean page forever.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from strands_robots.policies import cosmos3 as cosmos3_package
from strands_robots.policies.cosmos3.embodiments import EMBODIMENTS, Cosmos3Embodiment
from strands_robots.policies.cosmos3.policy import Cosmos3Policy

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PAGE = _REPO_ROOT / "docs" / "learn" / "policies" / "cosmos3.md"
_PROVIDERS = _REPO_ROOT / "docs" / "learn" / "policies" / "index.md"  # carries the {{providers:table}} token
_PROVIDERS_HOOK = _REPO_ROOT / "docs" / "hooks" / "providers.py"

#: The header of the provider matrix ``docs/hooks/providers.py`` renders.
_MATRIX_HEADER = ["provider", "class", "install extra", "also spelled", "what it drives", "trainer"]


def _page_text() -> str:
    """Return the cosmos3 provider page."""
    return _PAGE.read_text(encoding="utf-8")


def _providers_hook() -> ModuleType:
    """Load ``docs/hooks/providers.py`` by path, the way mkdocs does."""
    spec = importlib.util.spec_from_file_location("docs_hooks_providers", _PROVIDERS_HOOK)
    assert spec is not None and spec.loader is not None, _PROVIDERS_HOOK
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("docs_hooks_providers", module)
    spec.loader.exec_module(module)
    return module


def _readme_text() -> str:
    """Return the provider matrix as the reader sees it.

    ``docs/learn/policies/index.md`` carries a ``{{providers:table}}`` token that
    ``docs/hooks/providers.py`` expands at build time from
    ``registry/policies.json``; grading the page source would read the token,
    so the rendered table is graded instead.
    """
    page = _PROVIDERS.read_text(encoding="utf-8")
    assert "{{providers:table}}" in page, f"{_PROVIDERS} no longer carries the providers:table token"
    return _providers_hook().table()


def _cells(line: str) -> list[str]:
    """Split one markdown table row into its cells.

    Args:
        line: A single ``| a | b |`` table line.

    Returns:
        The inner cells, stripped. Splits on unescaped pipes so a cell
        containing ``\\|`` stays one cell.
    """
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def _table(md: str, header: list[str]) -> list[list[str]] | None:
    """Return the rows of the table whose header matches ``header``.

    Located by header cells rather than by position, so the page can grow
    sections above or below without moving the graded table.

    Args:
        md: Markdown document text.
        header: Expected header cells, lower-cased.

    Returns:
        The data rows, or ``None`` when no table carries that header.
    """
    lines = md.split("\n")
    for i, line in enumerate(lines[:-1]):
        if not line.strip().startswith("|"):
            continue
        if not re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            continue
        if [h.lower() for h in _cells(line)] != header:
            continue
        rows = []
        for row in lines[i + 2 :]:
            if not row.strip().startswith("|"):
                break
            rows.append(_cells(row))
        return rows
    return None


def _names(cell: str) -> set[str]:
    """Return the backtick-quoted identifiers in a cell or sentence."""
    return set(re.findall(r"`([A-Za-z0-9_]+)`", cell))


# --------------------------------------------------------------------------- #
# Graders. Each takes the registry so the vacuity meta-test can plant an entry.
# Each returns the embodiments the surface fails to account for, both ways.
# --------------------------------------------------------------------------- #


def _embodiments_sentence(md: str) -> str:
    """Return the enumeration sentence that opens the ``## Embodiments`` section.

    The section's first paragraph opens with a clause ending in a colon and then
    lists the accepted names in backticks, ending at the first full stop.
    Located by heading rather than by line number, so the page can grow above it.
    """
    match = re.search(r"^## Embodiments\s*\n+(.+?)(?:\n\n|\Z)", md, re.M | re.S)
    assert match is not None, f"{_PAGE} has no '## Embodiments' section with a paragraph under it"
    paragraph = " ".join(match.group(1).split())
    head, colon, tail = paragraph.partition(":")
    assert colon, f"{_PAGE} '## Embodiments' paragraph has no 'a, b, c:' enumeration clause: {paragraph!r}"
    return tail.split(".")[0]


def _embodiments_sentence_gap(md: str, registered: dict[str, Cosmos3Embodiment]) -> tuple[set[str], set[str]]:
    """Return (registered but unlisted, listed but unregistered) for the section sentence."""
    listed = _names(_embodiments_sentence(md))
    return set(registered) - listed, listed - set(registered)


def _readme_gap(readme: str, registered: dict[str, Cosmos3Embodiment]) -> set[str]:
    """Return registered embodiments the provider matrix's cosmos3 row omits.

    The "what it drives" cell of that row lists the embodiments in a
    parenthetical (``(DROID/UMI/AV/bridge/OpenArm)``), spelled the way their
    projects spell them; the registry's keys are lower case, so the comparison
    is case-insensitive.
    """
    rows = _table(readme, _MATRIX_HEADER)
    assert rows is not None, f"the provider matrix has no '| {' | '.join(_MATRIX_HEADER)} |' table"
    row = [r for r in rows if r and re.sub(r"[\[\]`]|\(.*\)", "", r[0]).strip() == "cosmos3"]  # cell is a link
    assert row, "the provider matrix has no 'cosmos3' row"
    drives = row[0][_MATRIX_HEADER.index("what it drives")]
    listed = {tok.strip().lower() for paren in re.findall(r"\(([^)]*)\)", drives) for tok in paren.split("/")}
    return set(registered) - listed


def _package_docstring_gap(doc: str, registered: dict[str, Cosmos3Embodiment]) -> set[str]:
    """Return registered embodiments the package docstring's list omits.

    Args:
        doc: ``strands_robots.policies.cosmos3.__doc__``.
        registered: Embodiment registry to grade against.

    Returns:
        The registered names absent from the ``Available embodiments:`` sentence.
    """
    match = re.search(r"Available embodiments:\s*([^(]*)", " ".join(doc.split()))
    assert match is not None, (
        "strands_robots.policies.cosmos3.__doc__ has no 'Available embodiments: ...' sentence. It is what "
        "help() on the package prints, so the sentence is the graded surface - reword it and this guard "
        "must be re-pointed rather than silently reporting a clean set."
    )
    return set(registered) - set(re.findall(r"[a-z0-9_]+", match.group(1)))


def _policy_args_gap(doc: str, registered: dict[str, Cosmos3Embodiment]) -> set[str]:
    """Return registered embodiments the ``embodiment:`` Args entry omits.

    Args:
        doc: :class:`Cosmos3Policy`'s docstring.
        registered: Embodiment registry to grade against.

    Returns:
        The registered names absent from the entry's first parenthetical.
    """
    match = re.search(r"embodiment:[^(]*\(([^)]*)\)", " ".join(doc.split()))
    assert match is not None, (
        "Cosmos3Policy.__doc__ has no 'embodiment: ... (...)' Args entry. That parenthetical is the "
        "accepted-value list for the parameter a caller passes, so it is the graded surface."
    )
    return set(registered) - set(re.findall(r"[a-z0-9_]+", match.group(1)))


class TestEveryRegisteredEmbodimentIsDocumented:
    """The page and the provider matrix account for exactly the registered set."""

    def test_embodiments_sentence_lists_exactly_the_registered_set(self) -> None:
        missing, extra = _embodiments_sentence_gap(_page_text(), EMBODIMENTS)
        assert not missing, (
            f"the '## Embodiments' sentence in docs/learn/policies/cosmos3.md omits {sorted(missing)}, which "
            "create_policy('cosmos3', embodiment=...) accepts. It is the one enumeration on the page, so an "
            "embodiment absent there is one a reader browsing the docs never learns the provider accepts."
        )
        assert not extra, (
            f"the '## Embodiments' sentence lists {sorted(extra)}, which the registry does "
            "not accept - a reader following the page gets a loud unknown-embodiment "
            "failure. Remove the name or register the embodiment."
        )

    def test_readme_provider_row_names_every_embodiment(self) -> None:
        missing = _readme_gap(_readme_text(), EMBODIMENTS)
        assert not missing, (
            f"the provider matrix's cosmos3 row omits {sorted(missing)}. The row's 'what it drives' cell "
            "enumerates the accepted embodiments (from registry/policies.json), so it drifts the same way "
            "the page does."
        )

    def test_package_docstring_names_every_embodiment(self) -> None:
        missing = _package_docstring_gap(cosmos3_package.__doc__ or "", EMBODIMENTS)
        assert not missing, (
            f"the 'Available embodiments:' sentence in strands_robots.policies.cosmos3's docstring omits "
            f"{sorted(missing)}. It is one import from the registry and is what help() on the package prints."
        )

    def test_policy_embodiment_arg_names_every_embodiment(self) -> None:
        missing = _policy_args_gap(Cosmos3Policy.__doc__ or "", EMBODIMENTS)
        assert not missing, (
            f"Cosmos3Policy's 'embodiment:' Args entry omits {sorted(missing)}. That parenthetical is the "
            "accepted-value list for the parameter, so a caller reading it is told the registry accepts less "
            "than it does."
        )


class TestThePremisesHold:
    """A reformat must fail loudly rather than make the graders report clean."""

    def test_every_graded_surface_is_found(self) -> None:
        md, readme = _page_text(), _readme_text()
        assert _table(readme, _MATRIX_HEADER)
        assert _names(_embodiments_sentence(md))
        assert re.search(r"Available embodiments:", " ".join((cosmos3_package.__doc__ or "").split()))
        assert re.search(r"embodiment:[^(]*\(", " ".join((Cosmos3Policy.__doc__ or "").split()))

    def test_the_registry_is_non_trivial(self) -> None:
        assert len(EMBODIMENTS) >= 4, (
            "fewer embodiments than the four this guard was written against - if the "
            "registry shrank deliberately, re-read the graded surfaces."
        )


class TestTheGradersAreNotVacuous:
    """Each grader must report an embodiment the page cannot mention."""

    @staticmethod
    def _planted() -> dict[str, Cosmos3Embodiment]:
        """Return the live registry plus one embodiment no surface names."""
        planted = dict(EMBODIMENTS)
        planted["zzz_planted_embodiment"] = Cosmos3Embodiment(
            name="zzz_planted_embodiment",
            domain_name="zzz_planted_domain",
            raw_action_dim=10,
            action_chunk_size=16,
            fps=15,
        )
        return planted

    def test_embodiments_sentence_grader_reports_it(self) -> None:
        missing, _ = _embodiments_sentence_gap(_page_text(), self._planted())
        assert "zzz_planted_embodiment" in missing

    def test_readme_grader_reports_it(self) -> None:
        assert "zzz_planted_embodiment" in _readme_gap(_readme_text(), self._planted())

    def test_package_docstring_grader_reports_it(self) -> None:
        assert "zzz_planted_embodiment" in _package_docstring_gap(cosmos3_package.__doc__ or "", self._planted())

    def test_policy_args_grader_reports_it(self) -> None:
        assert "zzz_planted_embodiment" in _policy_args_gap(Cosmos3Policy.__doc__ or "", self._planted())

    def test_extra_direction_reports_an_unregistered_name(self) -> None:
        """A documented embodiment the registry drops must be reported too."""
        trimmed = {k: v for k, v in EMBODIMENTS.items() if k != "av"}
        _, extra = _embodiments_sentence_gap(_page_text(), trimmed)
        assert "av" in extra

    def test_the_sentence_grader_reads_only_the_enumeration_clause(self) -> None:
        """Backticked names after the first full stop (`franka`, `panda`) are not embodiments."""
        page = _page_text()
        section = re.search(r"^## Embodiments\s*\n+(.+?)(?:\n\n|\Z)", page, re.M | re.S)
        assert section is not None
        assert "`franka`" in section.group(1), "the sim-asset sentence this test relies on moved"
        assert "franka" not in _names(_embodiments_sentence(page))


def test_a_missing_surface_is_reported_rather_than_skipped() -> None:
    """A moved page or renamed table must raise, never grade an empty set."""
    assert _PAGE.is_file(), _PAGE
    assert _PROVIDERS.is_file(), _PROVIDERS
    assert _PROVIDERS_HOOK.is_file(), _PROVIDERS_HOOK
    assert _table("| a | b |\n|---|---|\n| 1 | 2 |\n", _MATRIX_HEADER) is None
    with pytest.raises(AssertionError):
        _readme_gap("no table here", EMBODIMENTS)
    with pytest.raises(AssertionError):
        _embodiments_sentence_gap("no section here", EMBODIMENTS)
    with pytest.raises(AssertionError):
        _package_docstring_gap("no 'Available embodiments' sentence here", EMBODIMENTS)
    with pytest.raises(AssertionError):
        _policy_args_gap("no embodiment Args entry here", EMBODIMENTS)
