"""Every HuggingFace id the docs name is a repository that exists.

Issue #4158: the ``create_policy`` sketch on the lerobot_local page named
``lerobot/act_so101``, a repository that does not exist on the Hub, so the first
copied line failed with ``RepositoryNotFoundError``. The page was rewritten
around ``robotfuel/act_so101_t16b`` (#4234); the same phantom id survived on the
remote-inference page as the ``PolicyServer`` provider string. An id in a fence
reads as a promise that the line runs, so every one of them is graded here.

The Hub is not reachable from CI, so the grade has two layers:

* :data:`VERIFIED` is the table of ids the docs may name, each with the commit
  sha ``HfApi().model_info`` returned when it was checked and the day it was
  checked. Every quoted ``owner/name`` in the docs must be in it, be the reader
  placeholder ``you/<name>`` (a dataset the reader records themselves), or be
  one of the few quoted slash strings that are not Hub ids at all
  (:data:`NOT_HUB_IDS`), listed so a new one is a reviewed decision.
* With ``STRANDS_DOCS_HUB_LIVE=1`` the table itself is re-checked against the Hub,
  which is how a row is added: run it, paste the sha.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"

#: Hub ids the docs name, with the sha the Hub returned and the day it was read.
VERIFIED: dict[str, tuple[str, str]] = {
    "allenai/MolmoAct2-SO100_101": ("152569fe", "2026-09-30"),
    "lerobot/smolvla_base": ("d9f33c94", "2026-09-30"),
    "nvidia/Cosmos3-Nano-Policy-DROID": ("805c0d6d", "2026-09-30"),
    "nvidia/GEAR-SONIC": ("6733128a", "2026-09-30"),
    "nvidia/GR00T-N1.7-3B": ("2fc962b9", "2026-09-30"),
    "nvidia/Kimodo-G1-RP-v1": ("3020ad8c", "2026-09-30"),
    "robotfuel/act_so101_t16b": ("95bcae97", "2026-09-30"),
}

#: The owner the docs use for a repository the reader creates (a recorded dataset).
PLACEHOLDER_OWNER = "you"

#: Quoted ``a/b`` strings on checkpoint-naming lines that are not repository
#: ids: a directory under a training ``output_dir``. A relative path (``./x``)
#: and a file name (``meta/info.json``) are recognised by shape and need no entry.
NOT_HUB_IDS: frozenset[str] = frozenset({"checkpoints/last"})

_QUOTED = re.compile(r"[\"'`]([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)[\"'`]")
_FILE_SUFFIX = re.compile(r"\.(json|jsonl|py|md|xml|yaml|yml|toml|txt|png|svg|mp4|onnx|safetensors|csv)\Z")

#: A line names a checkpoint when one of these sits on it: the constructor
#: keyword, the smart-string factory, a provider string, a model id keyword, or
#: the Hub itself. A quoted slash string on any other line is a topic, a body
#: path or a ROS type, not a promise that a download works.
_CHECKPOINT_CONTEXT = re.compile(
    r"pretrained_name_or_path|create_policy\(|policy_provider\s*[=:]|model_id\s*=|checkpoint|huggingface\.co/|hf_hub_download|snapshot_download|from_pretrained"
)


def hub_ids_in(text: str) -> set[str]:
    """Every quoted ``owner/name`` on a checkpoint-naming line of ``text``."""
    found: set[str] = set()
    for line in text.splitlines():
        if not _CHECKPOINT_CONTEXT.search(line):
            continue
        for match in _QUOTED.finditer(line):
            token = match.group(1)
            _, _, name = token.partition("/")
            if _FILE_SUFFIX.search(name):
                continue
            found.add(token)
    return found


def _pages() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md") if "node_modules" not in p.parts)


def test_the_scanner_reads_the_shapes_it_claims() -> None:
    """Non-vacuity: a Hub id is found, a file name and a nested path are not."""
    text = (
        'create_policy("lerobot/smolvla_base"); open("meta/info.json")\n'
        'sim.add_camera(parent_body="so101/gripper")\n'
        'policy_config={"pretrained_name_or_path": "robotfuel/act_so101_t16b"}'
    )
    assert hub_ids_in(text) == {"lerobot/smolvla_base", "robotfuel/act_so101_t16b"}


def test_every_verified_row_is_still_named_somewhere() -> None:
    """A row nobody names is a stale table; drop it or name it."""
    named: set[str] = set()
    for page in _pages():
        named |= hub_ids_in(page.read_text(encoding="utf-8"))
    unused = sorted(set(VERIFIED) - named)
    assert not unused, f"VERIFIED names ids no docs page uses: {unused}"


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(DOCS)))
def test_every_hub_id_on_the_page_is_verified(page: Path) -> None:
    """An id in a fence is a promise that the line runs; every one is in the table."""
    unknown = sorted(
        token
        for token in hub_ids_in(page.read_text(encoding="utf-8"))
        if token not in VERIFIED and token not in NOT_HUB_IDS and token.partition("/")[0] != PLACEHOLDER_OWNER
    )
    assert not unknown, (
        f"{page.relative_to(DOCS)} names Hub ids that are not in VERIFIED: {unknown}. "
        "Check each with HfApi().model_info (STRANDS_DOCS_HUB_LIVE=1 runs the check) and add the row, "
        "or add a string that is not a repository id to NOT_HUB_IDS."
    )


@pytest.mark.skipif(not os.environ.get("STRANDS_DOCS_HUB_LIVE"), reason="set STRANDS_DOCS_HUB_LIVE=1 to ask the Hub")
@pytest.mark.parametrize("repo_id", sorted(VERIFIED))
def test_the_verified_table_still_resolves(repo_id: str) -> None:
    """The live half: each row is a repository the Hub knows today."""
    huggingface_hub = pytest.importorskip("huggingface_hub")

    info = huggingface_hub.HfApi().model_info(repo_id)
    assert info.sha, f"{repo_id} resolved without a sha"
