"""The LeRobot training page must open the gate its load-it-back step hits.

``create_policy("lerobot_local", pretrained_name_or_path=...)`` resolves a
lerobot checkpoint directory to ``lerobot_local``, which
``_check_trust_remote_code`` refuses unless ``STRANDS_TRUST_REMOTE_CODE`` is
set - a local, freshly trained directory included.
``examples/07_post_tune_any_policy.py`` sets the opt-in at that step; the
training page is what a reader follows, so wherever it tells them to load the
checkpoint they just trained it must have named the opt-in first, or that step
raises ``UntrustedRemoteCodeError`` on a clean install.

The old ``docs/reference/training/overview.md`` carried the load as a runnable
data-loop fence copied from the example, and this test exec'd the fence's
``os.environ`` statements. Its successor, ``docs/learn/training/lerobot.md``,
states the load as one sentence after the training fence, so the same claim is
graded on the page's text: the gate is asked first, to prove the opt-in is
still required, and then the page must name the variable before the load.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import strands_robots
from strands_robots.policies.factory import UntrustedRemoteCodeError, _check_trust_remote_code

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PAGE = _REPO_ROOT / "docs" / "learn" / "training" / "lerobot.md"
_OPT_IN = "STRANDS_TRUST_REMOTE_CODE"
#: The load-it-back step: a lerobot_local policy built from the trained checkpoint.
_LOAD = re.compile(r"""create_policy\(\s*["']lerobot_local["'][^)]*checkpoint_dir""")


def _page() -> str:
    return _PAGE.read_text(encoding="utf-8")


def test_the_page_still_tells_the_reader_to_load_the_trained_checkpoint() -> None:
    """The step this file guards is on the page; a page without it grades nothing."""
    assert _LOAD.search(_page()), (
        "docs/learn/training/lerobot.md no longer shows create_policy('lerobot_local', "
        "pretrained_name_or_path=result.checkpoint_dir, ...); if the load moved, repoint _LOAD"
    )


def test_the_gate_still_refuses_lerobot_local_without_the_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason the page has to name the variable: the load fails without it."""
    monkeypatch.delenv(_OPT_IN, raising=False)
    with pytest.raises(UntrustedRemoteCodeError):
        _check_trust_remote_code("lerobot_local")


def test_the_page_names_the_opt_in_before_it_loads_the_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader following the page top to bottom has set the variable when they reach the load."""
    text = _page()
    load = _LOAD.search(text)
    assert load is not None
    first_mention = text.find(_OPT_IN)
    assert first_mention != -1, (
        f"docs/learn/training/lerobot.md tells the reader to load the trained checkpoint with "
        f"create_policy('lerobot_local', ...) and never names {_OPT_IN}; on a clean install that step "
        "raises UntrustedRemoteCodeError. Name the opt-in (export or os.environ) before the load."
    )
    assert first_mention < load.start(), f"{_OPT_IN} is named on the page only after the checkpoint load"
    # What the page prescribes opens the gate: set it the way the page spells it.
    monkeypatch.setenv(_OPT_IN, "1")
    _check_trust_remote_code("lerobot_local")
