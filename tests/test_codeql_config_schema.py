"""Pin the schema of .github/codeql/config.yml and the codeql workflow layout.

These are config-correctness contracts, not runtime behaviour. A schema typo
in config.yml (excludes vs exclude, misspelled rule id, id-as-list) silently
no-ops the py/unsafe-cyclic-import suppression and only manifests weeks later as
alerts re-appearing in the Security tab. These pins fail loud at PR time.

Companion to tests/simulation/test_no_import_cycle.py (which pins runtime
safety) and the codeql-suppression-narrowness CI job (which pins scope).

Issues: #215, #229, #236, #237.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / ".github/codeql/config.yml"
README = REPO_ROOT / ".github/codeql/README.md"
WORKFLOWS = REPO_ROOT / ".github/workflows"
CHECKER = REPO_ROOT / ".github/codeql/check_suppression_narrowness.py"

RULE_ID = "py/unsafe-cyclic-import"
EXPECTED_TRIPLE = {
    "strands_robots/simulation/base.py",
    "strands_robots/simulation/policy_runner.py",
    "strands_robots/simulation/benchmark.py",
}
TOP_LEVEL_ALLOWLIST = {
    "name",
    "disable-default-queries",
    "queries",
    "paths-ignore",
    "paths",
    "query-filters",
}


# --------------------------------------------------------------------------- #
# #236 — schema-validation pin for config.yml                                  #
# --------------------------------------------------------------------------- #
def _load_config() -> dict:
    assert CONFIG.exists(), f"missing CodeQL config: {CONFIG}"
    return yaml.safe_load(CONFIG.read_text())


def test_config_query_filter_shape():
    config = _load_config()
    filters = config.get("query-filters", [])
    assert len(filters) >= 1, "config.yml: query-filters block missing or empty"
    exclude = filters[0].get("exclude")
    assert exclude is not None, (
        "config.yml: first query-filter must be an 'exclude' key "
        f"(not 'excludes' or another key); got keys {list(filters[0])}"
    )
    rule = exclude.get("id")
    assert isinstance(rule, str), f"config.yml: exclude.id must be a string, got {type(rule).__name__}: {rule!r}"
    assert rule == RULE_ID, f"config.yml: exclude.id must be {RULE_ID!r}, got {rule!r}"


def test_config_top_level_keys_in_allowlist():
    config = _load_config()
    unknown = set(config) - TOP_LEVEL_ALLOWLIST
    assert not unknown, (
        f"config.yml: unknown top-level key(s) {sorted(unknown)} — a typo such as "
        f"'querie-filters' would silently no-op. Allowed: {sorted(TOP_LEVEL_ALLOWLIST)}"
    )


def test_config_suppression_is_single():
    """The suppression must stay narrow: exactly one query-filter exclude."""
    config = _load_config()
    filters = config.get("query-filters", [])
    excludes = [f for f in filters if "exclude" in f]
    assert len(excludes) == 1, (
        f"config.yml: expected exactly 1 exclude filter (narrow suppression), got {len(excludes)}. "
        "Widening the suppression needs a documented rationale in .github/codeql/README.md."
    )


# --------------------------------------------------------------------------- #
# #237 — single CodeQL workflow; no duplicate Python scan                      #
# --------------------------------------------------------------------------- #
def test_only_one_codeql_workflow():
    codeql_workflows = sorted(WORKFLOWS.glob("codeql*.yml"))
    names = [p.name for p in codeql_workflows]
    assert names == ["codeql.yml"], (
        f"expected exactly one CodeQL workflow (codeql.yml); found {names}. "
        "codeql-advanced.yml was consolidated away in #237 to drop the duplicate Python scan."
    )


def test_codeql_workflow_scans_python_once():
    wf = yaml.safe_load((WORKFLOWS / "codeql.yml").read_text())
    matrix = wf["jobs"]["analyze"]["strategy"]["matrix"]["include"]
    langs = [entry["language"] for entry in matrix]
    assert langs.count("python") == 1, f"python must be scanned exactly once, got {langs}"
    assert "actions" in langs, f"actions language should be folded into the single workflow, got {langs}"


def test_codeql_workflow_uses_shared_config():
    wf = yaml.safe_load((WORKFLOWS / "codeql.yml").read_text())
    steps = wf["jobs"]["analyze"]["steps"]
    init = next(s for s in steps if "init" in s.get("uses", ""))
    assert init["with"].get("config-file") == "./.github/codeql/config.yml", (
        "analyze job must consume the shared config so the suppression applies"
    )


# --------------------------------------------------------------------------- #
# #229 — narrowness checker behaviour (unit test of the SARIF parser)          #
# --------------------------------------------------------------------------- #
def _load_checker():
    spec = importlib.util.spec_from_file_location("_narrowness", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _sarif_for(files: set[str]) -> dict:
    return {
        "runs": [
            {
                "results": [
                    {
                        "ruleId": RULE_ID,
                        "locations": [{"physicalLocation": {"artifactLocation": {"uri": f}}}],
                    }
                    for f in files
                ]
            }
        ]
    }


def test_checker_passes_on_exact_triple():
    mod = _load_checker()
    assert mod._extract_violating_files(_sarif_for(EXPECTED_TRIPLE)) == EXPECTED_TRIPLE


def test_checker_detects_expansion(tmp_path):
    mod = _load_checker()
    expanded = EXPECTED_TRIPLE | {"strands_robots/mesh/core.py"}
    sarif_file = tmp_path / "x.sarif"
    import json

    sarif_file.write_text(json.dumps(_sarif_for(expanded)))
    rc = mod.main(["check", str(sarif_file)])
    assert rc == 1, "checker must FAIL (rc=1) when a non-triple file fires the rule"


def test_checker_strips_leading_dotslash():
    mod = _load_checker()
    got = mod._extract_violating_files(_sarif_for({"./strands_robots/simulation/base.py"}))
    assert got == {"strands_robots/simulation/base.py"}


# --------------------------------------------------------------------------- #
# #215 — config + README + breadcrumb exist and cross-reference               #
# --------------------------------------------------------------------------- #
def test_readme_documents_suppression():
    assert README.exists(), "missing .github/codeql/README.md"
    text = README.read_text()
    assert RULE_ID in text
    assert "simulation/base.py" in text
    assert "test_no_import_cycle.py" in text


def test_simulation_base_breadcrumb_points_at_readme():
    base = (REPO_ROOT / "strands_robots/simulation/base.py").read_text()
    assert ".github/codeql/README.md" in base, (
        "base.py must carry a breadcrumb pointing readers at the CodeQL suppression rationale"
    )
