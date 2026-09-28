"""The docs describe what ``[all]`` installs, derived from ``pyproject.toml``.

``[all]`` is a convenience bundle, not a union: it names most extras and leaves
the GPU-only backends, the separate-install toolchains and several service
clients opt-in. Three pages tell a reader what it covers, and the membership
they describe is a fact about ``[project.optional-dependencies]`` rather than
prose - so it is derivable, and it drifted.

The old ``docs/getting-started/installation.md`` enumerated ``[all]`` as five
extras while the bundle had grown to nineteen, and the old architecture page
called it a "union". The new tree generates the extras table on
``docs/start/install.md`` from ``pyproject.toml`` through
``docs/hooks/extras.py`` (the ``{{extras:table}}`` token), so the membership
column can no longer drift; what can still drift is the hand-written purpose
column of the ``all`` row and any prose that describes the bundle. The page is
graded as the reader sees it, with the token expanded by the hook, and every
rule derives its expectation from ``pyproject.toml``.

Deliberately out of scope: *why* a given extra is left out of ``[all]``. The
excluded set has no single rule - ``[sim-isaac]`` and ``[sim-gs]`` need a GPU
and say so in the README, while ``[cosmos3-service]`` is two pure-Python
packages and ``[microduck]`` is one - so documenting the reason per extra is a
maintainer's call, not a derivation. These rules grade *which* extras are
excluded, never why.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_DOCS = _ROOT / "docs"
_INSTALL_PAGE = _DOCS / "start" / "install.md"
_HOOK = _DOCS / "hooks" / "extras.py"

# The bundle is a developer convenience, so its tooling extra is not a
# capability a reader installs it for; the pages describe capability extras.
_TOOLING_EXTRAS = frozenset({"dev"})

#: The ``all`` row of the extras table: the hook writes ``| `all` |``, a
#: hand-written row ``| `[all]` |``.
_ALL_ROW = re.compile(r"^\|\s*`\[?all\]?`\s*\|")

# Words that describe the bundle as complete. ``[all]`` is not, so a page using
# one of these tells a reader they need no further extra.
_COMPLETENESS_CLAIMS = ("union", "everything", "every policy", "every extra", "every runtime extra")
_NEGATED_CLAIM = re.compile(
    r"not(?:\*\*)?\s+(?:a\s+)?(?:" + "|".join(re.escape(c) for c in _COMPLETENESS_CLAIMS) + r")"
)
_EXTRA_IN_TEXT = re.compile(r"`\[([a-z0-9][a-z0-9-]*)\]`")


def _rendered(page: Path) -> str:
    """``page`` with the ``{{extras:...}}`` tokens expanded by the shipped hook."""
    spec = importlib.util.spec_from_file_location("docs_hooks_extras", _HOOK)
    assert spec is not None and spec.loader is not None
    module = sys.modules.get(spec.name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module.substitute(page.read_text(encoding="utf-8"), str(page.relative_to(_DOCS)))


def _pages_describing_all() -> dict[Path, str]:
    """Every hand-written page that mentions the bundle, rendered."""
    out: dict[Path, str] = {}
    for page in sorted(_DOCS.rglob("*.md")):
        if _DOCS / "robots" in page.parents:
            continue
        text = _rendered(page) if page == _INSTALL_PAGE else page.read_text(encoding="utf-8")
        if "[all]" in text or any(_ALL_ROW.match(line) for line in text.splitlines()):
            out[page] = text
    return out


def _extras() -> dict[str, list[str]]:
    """Every entry of ``[project.optional-dependencies]``."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def _closure(extras: dict[str, list[str]], name: str) -> set[str]:
    """Extras reachable from ``name`` through ``strands-robots[...]`` specs.

    An extra can name siblings (``[mesh-iot]`` pulls ``[mesh]``), so the set a
    reader gets is the transitive closure rather than the literal list.
    """
    reached: set[str] = set()
    pending = [name]
    while pending:
        for spec in extras.get(pending.pop(), []):
            found = re.search(r"strands-robots\[([^]]+)\]", str(spec))
            if found is None:
                continue
            for part in found.group(1).split(","):
                extra = part.strip()
                if extra and extra not in reached:
                    reached.add(extra)
                    pending.append(extra)
    return reached


def _membership() -> tuple[set[str], set[str], int]:
    """``(installed, left_opt_in, declared_total)`` for ``[all]``."""
    extras = _extras()
    installed = _closure(extras, "all")
    declared = set(extras) - {"all"}
    left_out = declared - installed - _TOOLING_EXTRAS
    return installed, left_out, len(declared)


def _row(page: Path) -> str:
    """The ``all`` row of ``page``'s rendered extras table."""
    for line in _rendered(page).splitlines():
        if _ALL_ROW.match(line):
            return line
    raise AssertionError(f"{page.name} no longer carries an `all` row in its extras table")


def _direct_members() -> set[str]:
    """The extras ``[all]`` names itself, before the closure walk."""
    return {
        part.strip()
        for spec in _extras()["all"]
        for match in [re.search(r"strands-robots\[([^]]+)\]", str(spec))]
        if match
        for part in match.group(1).split(",")
    }


def _extras_named(text: str) -> set[str]:
    """Extra names written as ``` `[name]` ``` in ``text``."""
    return set(_EXTRA_IN_TEXT.findall(text))


class TestThePagesAgreeWithPyproject:
    """The membership the page shows and the words around it both come from ``pyproject.toml``."""

    def test_the_row_states_the_derived_membership(self) -> None:
        """The install table's ``all`` row names what the bundle pulls, no more and no less.

        The generated row lists the extras ``[all]`` names directly; a reader
        deciding whether the bundle covers them reads that cell. It must match
        pyproject exactly, so the reader is not told the bundle is narrower (or
        wider) than it is.
        """
        row = _row(_INSTALL_PAGE)
        named = _extras_named(row)
        direct = _direct_members()
        assert named == direct, (
            f"install.md: the `all` row names {sorted(named)} but pyproject's [all] names {sorted(direct)}. The row reads:\n  {row}"
        )

    def test_the_install_row_names_every_extra_it_installs_directly(self) -> None:
        installed, _, _ = _membership()
        row = _row(_INSTALL_PAGE)
        named = _extras_named(row)
        missing = sorted(_direct_members() - named)
        assert not missing, (
            f"install.md: the `all` row must name every extra the bundle pulls, because that is what a reader "
            f"gets from one install line. Unnamed: {missing}."
        )
        assert named <= installed, f"the row names extras the closure does not reach: {sorted(named - installed)}"

    def test_the_install_row_claims_nothing_it_leaves_opt_in(self) -> None:
        _, left_out, _ = _membership()
        row = _row(_INSTALL_PAGE)
        wrongly_named = sorted(_extras_named(row) & left_out)
        assert not wrongly_named, (
            f"install.md: the `all` row lists {wrongly_named} as installed, but `all` leaves them opt-in. "
            f"A reader would believe they have an extra they still need to add."
        )

    def test_no_page_calls_the_bundle_a_union_or_everything(self) -> None:
        _, left_out, _ = _membership()
        assert left_out, "nothing is left opt-in, so 'union' would be accurate and this rule is vacuous"
        pages = _pages_describing_all()
        assert _INSTALL_PAGE in pages, "install.md no longer describes the [all] bundle"
        for page, text in pages.items():
            for line in text.splitlines():
                if "strands-robots[all]" not in line and "[all]" not in line and not _ALL_ROW.match(line):
                    continue
                # A corrected page says "not a union", so a bare substring test would
                # report the very wording that fixes this. Drop negated forms first and
                # grade what is left, which is the affirmative claim.
                lowered = _NEGATED_CLAIM.sub("", line.lower())
                for claim in _COMPLETENESS_CLAIMS:
                    assert claim not in lowered, (
                        f"{page.relative_to(_ROOT)}: {claim!r} describes `[all]` as complete, and {len(left_out)} extras "
                        f"stay opt-in ({sorted(left_out)}). The line reads:\n  {line}"
                    )


class TestTheDerivationIsNotVacuous:
    """The membership split is real, so the rules above have something to grade."""

    def test_the_bundle_installs_most_but_not_all_extras(self) -> None:
        installed, left_out, declared_total = _membership()
        assert len(installed) > 1, f"only {len(installed)} extras reached from `all`; the closure walk is blind"
        assert left_out, "no extra is left opt-in, so `all` really is a union and these rules grade nothing"
        assert len(installed) + len(left_out) + len(_TOOLING_EXTRAS) == declared_total, (
            f"the split does not account for every extra: {len(installed)} installed + {len(left_out)} opt-in "
            f"+ {len(_TOOLING_EXTRAS)} tooling != {declared_total} declared"
        )

    def test_the_closure_follows_an_extra_named_two_levels_down(self) -> None:
        extras = _extras()
        direct = _direct_members()
        indirect = _closure(extras, "all") - direct
        assert indirect, (
            "every extra `all` installs is named directly by it, so a walk that did not recurse would "
            "still produce the right count and the counts these rules assert would not grade the walk"
        )
        assert indirect <= _closure(extras, "all"), "the closure must contain what it reached indirectly"


class TestTheRulesReportAConstructedDrift:
    """The rules are graded on built exemplars as well as on the shipped page."""

    def test_a_row_missing_a_member_is_reported(self) -> None:
        direct = sorted(_direct_members())
        stale = "| `all` | " + ", ".join(f"`[{name}]`" for name in direct[1:]) + " | x |"
        assert _ALL_ROW.match(stale)
        assert _extras_named(stale) != set(direct)

    def test_a_row_naming_an_opt_in_extra_is_reported(self) -> None:
        _, left_out, _ = _membership()
        extra = sorted(left_out)[0]
        stale = f"| `all` | `[{extra}]` | x |"
        assert _extras_named(stale) & left_out

    def test_a_union_claim_is_reported(self) -> None:
        for stale in ("| `[all]` | union | CI / exploration |", "| `all` | `[mesh]` | every runtime extra above |"):
            assert any(c in _NEGATED_CLAIM.sub("", stale.lower()) for c in _COMPLETENESS_CLAIMS), stale

    def test_a_negated_claim_is_not_reported(self) -> None:
        corrected = "| `[all]` | 19 of the 31 extras - **not** a union | x |".lower()
        assert not any(c in _NEGATED_CLAIM.sub("", corrected) for c in _COMPLETENESS_CLAIMS), (
            "the rule must accept a page that says the bundle is NOT complete, or the wording that "
            "fixes this defect would itself be reported"
        )
