"""Regression test: all workflow action references must pin to full 40-char SHAs.

Enforces AGENTS.md > Review Learnings (PR #92) > Action Pinning:
  "All uses: references in workflows pin to a full 40-character commit SHA,
   with the version tag preserved as a trailing comment."

This prevents floating tags (e.g. @v4) from silently re-entering the repo,
which is the supply-chain attack vector exploited in the tj-actions/changed-files
incident.
"""

import re
from pathlib import Path

# Pattern for a valid pinned action reference:
# owner/action@<40-hex-chars>  # <version-tag>
# Version tag must match one of:
#   v1, v1.2, v1.2.3  (semver-style)
#   release/v1         (pypa convention)
# This rejects stale/typo'd tags while still accepting known valid formats.
_SHA_PIN_RE = re.compile(
    r"^[A-Za-z0-9_./-]+@[a-f0-9]{40}\s+#\s+(release/)?v[0-9]+(\.[0-9]+)*$",
)
_LOCAL_WORKFLOW_RE = re.compile(r"^\./")


def _get_workflow_dir() -> Path:
    """Find .github/workflows/ relative to repo root."""
    repo_root = Path(__file__).resolve().parent.parent
    workflows_dir = repo_root / ".github" / "workflows"
    assert workflows_dir.is_dir(), f"Workflows dir not found: {workflows_dir}"
    return workflows_dir


def test_all_uses_pinned_to_full_sha():
    """Every 'uses:' in .github/workflows/*.{yml,yaml} must pin to a full SHA with version comment."""
    workflows_dir = _get_workflow_dir()
    violations = []

    workflow_files = sorted([*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")])

    for yml_file in workflow_files:
        with open(yml_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()

                # Skip commented-out lines
                if stripped.startswith("#"):
                    continue

                # Find lines with 'uses:'
                match = re.search(r"\buses:\s+(.+)", stripped)
                if not match:
                    continue

                action_ref = match.group(1).strip()

                # Local workflow references (./.github/workflows/X.yml) are fine
                if _LOCAL_WORKFLOW_RE.match(action_ref):
                    continue

                # Must match: owner/action@<40-hex>  # [release/]v<version>
                if not _SHA_PIN_RE.match(action_ref):
                    violations.append(f"  {yml_file.name}:{line_num}: {action_ref}")

    assert not violations, (
        "Floating tags or missing version-tag comments found in workflow actions.\n"
        "All uses: must match: owner/action@<40-char-sha>  # [release/]v<version>\n"
        "See AGENTS.md > Review Learnings (PR #92) > Action Pinning.\n\n"
        "Violations:\n" + "\n".join(violations)
    )
