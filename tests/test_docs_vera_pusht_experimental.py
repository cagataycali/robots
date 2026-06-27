"""Repo hygiene: VERA docs/examples must not claim PushT solves the task.

VERA's PushT IDM ``du`` action path is **not wired end-to-end upstream** (VERA's
own ``configurations/dataset/pusht.yaml`` documents the gap), so the PushT
embodiment validates the provider -> server -> action plumbing rather than
producing a solving rollout. The faithful, working VERA demo is
``mimicgen`` (WAN planner + Jacobian IDM -> eef-delta -> IK -> Panda).

This guard fails fast if the misleading "PushT works" example, its rollout
assets, or any cross-link to the removed example creep back, while asserting the
``pusht`` embodiment config stays usable (the server still runs) and the working
``mimicgen`` example stays present and referenced.
"""

from __future__ import annotations

from pathlib import Path

from strands_robots.policies import create_policy
from strands_robots.policies.vera.config import VeraConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
EXAMPLES = REPO_ROOT / "examples"
VERA_DOC = DOCS / "policies" / "vera.md"


def test_pusht_example_dir_removed() -> None:
    """The plumbing-only PushT example no longer ships."""
    assert not (EXAMPLES / "vera_pusht_mujoco").exists(), (
        "examples/vera_pusht_mujoco/ validates plumbing only (PushT IDM not wired upstream) and must not ship"
    )


def test_pusht_rollout_assets_removed() -> None:
    """The PushT rollout gif/mp4 (implying a solving rollout) are gone."""
    for name in ("pusht_rollout.gif", "pusht_rollout.mp4"):
        assert not (DOCS / "assets" / "vera" / name).exists(), f"stale PushT asset: {name}"


def test_no_dead_links_to_removed_pusht_example() -> None:
    """No doc or example cross-links to the removed PushT example path."""
    for base in (DOCS, EXAMPLES):
        for md in base.rglob("*.md"):
            assert "vera_pusht_mujoco" not in md.read_text(encoding="utf-8"), (
                f"{md.relative_to(REPO_ROOT)} links the removed PushT example"
            )


def test_examples_make_no_pusht_claim() -> None:
    """Shipped examples carry no PushT references at all."""
    for path in EXAMPLES.rglob("*"):
        if path.suffix in {".py", ".md"} and path.is_file():
            assert "pusht" not in path.read_text(encoding="utf-8").lower(), (
                f"{path.relative_to(REPO_ROOT)} still references PushT"
            )


def test_vera_doc_marks_pusht_experimental_not_solving() -> None:
    """The VERA doc flags PushT as experimental and shows no solving rollout."""
    text = VERA_DOC.read_text(encoding="utf-8")
    assert "pusht_rollout" not in text, "VERA doc still embeds a PushT rollout artifact"
    assert "Wave-1 (local)" not in text, "VERA doc still marks pusht as a shipped Wave-1 embodiment"
    lowered = text.lower()
    assert "pusht" in lowered, "VERA doc should still document pusht as experimental"
    assert "experimental" in lowered and "not wired end-to-end" in lowered, (
        "VERA doc must flag the pusht IDM gap as experimental / not wired end-to-end"
    )


def test_mimicgen_example_kept_and_referenced() -> None:
    """The working MimicGen -> Panda demo stays present and linked from the doc."""
    assert (EXAMPLES / "vera_mimicgen_panda" / "rollout.py").is_file()
    assert (DOCS / "assets" / "vera" / "mimicgen_panda.gif").is_file()
    assert "vera_mimicgen_panda" in VERA_DOC.read_text(encoding="utf-8")


def test_pusht_embodiment_config_still_valid() -> None:
    """Keeping the config: the pusht embodiment still constructs (server runs)."""
    cfg = VeraConfig(embodiment="pusht")
    assert cfg.embodiment == "pusht"
    # default ports preserved for the running server
    assert (cfg.server_port, cfg.vis_port) == (8820, 8821)
    # provider still builds for pusht without launching a server
    policy = create_policy("vera", embodiment="pusht", auto_launch_server=False)
    assert policy is not None
