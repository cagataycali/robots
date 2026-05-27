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
  2. The historical wrong path is no longer present. A future contributor
     who files Dependabot config alongside Actions workflows re-creates
     the silent-ignore state this test exists to prevent. Block both states.
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


def test_dependabot_yml_unique_in_tree() -> None:
    """The canonical path must be the only dependabot config in the tree.

    A copy-paste to ``.github/workflows/dependabot.yml``,
    ``.github/dependabot.yaml``, or any other path re-creates the
    silent-ignore state this test exists to prevent.
    """
    found = sorted(REPO_ROOT.glob("**/dependabot.y*ml"))
    assert found == [CANONICAL], (
        f"Dependabot reads only {CANONICAL.relative_to(REPO_ROOT)}; "
        f"any other dependabot.y*ml file is silently ignored. Found: "
        f"{[str(p.relative_to(REPO_ROOT)) for p in found]}"
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

    # Every entry must declare all three Dependabot v2 required fields;
    # missing any silently disables that entry without a parser error.
    for idx, entry in enumerate(updates):
        assert isinstance(entry, dict), (
            f"{CANONICAL.relative_to(REPO_ROOT)}: updates[{idx}] must be a mapping, got {type(entry).__name__}"
        )
        assert "package-ecosystem" in entry, (
            f"{CANONICAL.relative_to(REPO_ROOT)}: updates[{idx}] missing "
            f"'package-ecosystem' (required by Dependabot v2 schema). "
            f"Got keys: {sorted(entry.keys())}"
        )
        assert "directory" in entry, (
            f"{CANONICAL.relative_to(REPO_ROOT)}: updates[{idx}] missing "
            f"'directory' (required by Dependabot v2; entry is silently "
            f"disabled without it)"
        )
        schedule = entry.get("schedule")
        assert isinstance(schedule, dict) and "interval" in schedule, (
            f"{CANONICAL.relative_to(REPO_ROOT)}: updates[{idx}] missing "
            f"'schedule.interval' (required by Dependabot v2; entry is "
            f"silently disabled without it)"
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
    declared = {entry["package-ecosystem"] for entry in config["updates"]}
    assert ecosystem in declared, (
        f"{CANONICAL.relative_to(REPO_ROOT)}: 'package-ecosystem: {ecosystem}' "
        f"entry missing. Declared ecosystems: {sorted(declared)}"
    )


def test_dependabot_pip_groups_pin_ml_stack() -> None:
    """Pin that the pip entry's ``groups`` block covers ML-stack packages.

    The PR description's lead bullet on real-world harm is that the
    ``pip`` grouping for ``torch`` / ``lerobot`` / ``transformers`` never
    fired. A future edit that deletes the ``groups:`` block silently
    regresses exactly the harm this PR cites. Pin the key patterns.
    """
    config = yaml.safe_load(CANONICAL.read_text(encoding="utf-8"))

    pip_entry = next(
        (e for e in config["updates"] if e["package-ecosystem"] == "pip"),
        None,
    )
    assert pip_entry is not None, "pip ecosystem entry missing"

    groups = pip_entry.get("groups")
    assert isinstance(groups, dict) and len(groups) > 0, (
        f"{CANONICAL.relative_to(REPO_ROOT)}: pip entry must declare "
        f"'groups' to batch related dependency updates (currently missing "
        f"or empty)"
    )

    # The ml-stack group must exist and cover torch + lerobot
    assert "ml-stack" in groups, (
        f"{CANONICAL.relative_to(REPO_ROOT)}: pip groups must include "
        f"'ml-stack' (covers torch/lerobot/transformers). "
        f"Found groups: {sorted(groups.keys())}"
    )
    ml_patterns = groups["ml-stack"].get("patterns", [])
    assert any("torch" in p for p in ml_patterns), f"ml-stack group must include a torch* pattern. Got: {ml_patterns}"
    assert any("lerobot" in p for p in ml_patterns), (
        f"ml-stack group must include a lerobot* pattern. Got: {ml_patterns}"
    )
