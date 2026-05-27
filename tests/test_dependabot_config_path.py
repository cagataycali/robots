"""Repo hygiene: pin Dependabot config at the canonical path GitHub reads.

Background: Dependabot reads ``.github/dependabot.yml`` only. Files at any
other path (e.g. ``.github/workflows/dependabot.yml``) are valid YAML and
are quietly listed by other tooling, but Dependabot itself ignores them.
This produces the worst failure mode of a config file: it appears to work
(parses, lints, lives in version control) but does nothing.

History: ``.github/workflows/dependabot.yml`` shipped on main and was
silently inactive for the entire window between #92 (which introduced
the SHA-pinning convention that Dependabot is supposed to keep fresh)
and #234 (the first SHA-realignment that exposed the gap). Net result:
the ``pip`` ecosystem grouping for ``torch`` / ``lerobot`` / ``transformers``
never fired, and every workflow SHA pin became manual maintenance.

This test pins:
  1. The config exists at the canonical path GitHub reads.
  2. The historical wrong path is no longer present (a future copy-paste
     that re-creates it would silently disable the canonical file's
     intended supersede semantics? No -- but it would re-create the
     ambiguity that this test exists to prevent. Block both states.)
  3. The schema is the minimum-viable Dependabot v2 shape: a top-level
     ``version: 2`` and a non-empty ``updates`` list. A schema typo at
     either point silently disables the config without a parser error
     (Dependabot validates server-side; the YAML loader does not).

See: https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = REPO_ROOT / ".github" / "dependabot.yml"
HISTORICAL_WRONG = REPO_ROOT / ".github" / "workflows" / "dependabot.yml"


def test_dependabot_yml_at_canonical_path() -> None:
    """Dependabot reads ``.github/dependabot.yml`` only.

    A config at any other path is silently ignored by Dependabot, so
    presence at the canonical path is a hard contract.
    """
    assert CANONICAL.is_file(), (
        f"Dependabot config must live at {CANONICAL.relative_to(REPO_ROOT)}; "
        f"any other path is silently ignored by Dependabot. "
        f"See https://docs.github.com/en/code-security/dependabot/"
        f"dependabot-version-updates/configuration-options-for-the-dependabot.yml-file"
    )


def test_dependabot_yml_not_at_historical_wrong_path() -> None:
    """Block the historical placement bug from coming back.

    A future contributor who files Dependabot config alongside Actions
    workflows (because they look related) re-creates the silent-ignore
    state this test exists to prevent. Fail loud.
    """
    assert not HISTORICAL_WRONG.is_file(), (
        f"Dependabot config at {HISTORICAL_WRONG.relative_to(REPO_ROOT)} "
        f"is silently ignored by Dependabot; move it to "
        f"{CANONICAL.relative_to(REPO_ROOT)} (the canonical path)."
    )


def test_dependabot_yml_minimum_viable_schema() -> None:
    """Pin the v2 schema shape: ``version: 2`` and a non-empty ``updates`` list.

    Catches typos that pass YAML parsing but silently disable the config
    on Dependabot's side (e.g. ``version: 1`` is a parseable string but
    Dependabot v1 was deprecated; ``updates: {}`` is parseable but
    semantically a no-op; ``update:`` -- singular -- is parseable but
    ignored).
    """
    config = yaml.safe_load(CANONICAL.read_text(encoding="utf-8"))

    assert isinstance(config, dict), (
        f"{CANONICAL.relative_to(REPO_ROOT)}: top-level must be a mapping, got {type(config).__name__}"
    )

    version = config.get("version")
    assert version == 2, (
        f"{CANONICAL.relative_to(REPO_ROOT)}: 'version' must be 2 (the only "
        f"supported Dependabot config version), got {version!r}"
    )

    updates = config.get("updates")
    assert isinstance(updates, list) and len(updates) > 0, (
        f"{CANONICAL.relative_to(REPO_ROOT)}: 'updates' must be a non-empty "
        f"list (config-with-no-updates is a silent no-op), got {updates!r}"
    )

    # Every entry must declare a package-ecosystem; missing this key on
    # any entry silently disables that entry without a parser error.
    for idx, entry in enumerate(updates):
        assert isinstance(entry, dict), (
            f"{CANONICAL.relative_to(REPO_ROOT)}: updates[{idx}] must be a mapping, got {type(entry).__name__}"
        )
        assert "package-ecosystem" in entry, (
            f"{CANONICAL.relative_to(REPO_ROOT)}: updates[{idx}] missing "
            f"'package-ecosystem' (required by Dependabot v2 schema). "
            f"Got keys: {sorted(entry.keys())}"
        )


@pytest.mark.parametrize(
    "ecosystem",
    [
        "pip",
        "github-actions",
    ],
)
def test_dependabot_covers_expected_ecosystems(ecosystem: str) -> None:
    """Pin that the two ecosystems the repo depends on stay in scope.

    ``pip`` keeps the ML stack (torch / lerobot / transformers) updated.
    ``github-actions`` keeps SHA pins fresh (#92, #234). Either one
    going missing silently regresses dependency hygiene.
    """
    config = yaml.safe_load(CANONICAL.read_text(encoding="utf-8"))
    declared = {entry.get("package-ecosystem") for entry in config["updates"]}
    assert ecosystem in declared, (
        f"{CANONICAL.relative_to(REPO_ROOT)}: 'package-ecosystem: {ecosystem}' "
        f"entry missing. Declared ecosystems: {sorted(declared - {None})}"
    )
