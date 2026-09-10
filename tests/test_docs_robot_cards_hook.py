"""docs/hooks/robot_cards.py puts every registered robot on exactly one family page.

The family pages carry ``{{robot_cards:<categories>}}`` tokens instead of hand
tables; this grader checks the tokens cover every registry category once, and
that the generated cards reference only renders that exist.
"""

from __future__ import annotations

import importlib.util
import re
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _hook():
    spec = importlib.util.spec_from_file_location("docs_robot_cards_hook", _REPO / "docs/hooks/robot_cards.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _token_categories(hook) -> Counter[str]:  # noqa: ANN001 - the loaded hook module
    """Every category named by a ``{{robot_cards:...}}`` token, with its page count.

    Uses the hook's own token pattern rather than a copy of it, so a build that
    stops expanding a token cannot leave this grader counting it.
    """
    seen: Counter[str] = Counter()
    for page in sorted((_REPO / "docs/robots").glob("*.md")):
        for match in hook._TOKEN.findall(page.read_text(encoding="utf-8")):
            seen.update(c.strip() for c in match.split(",") if c.strip())
    return seen


def test_every_registry_category_is_carded_on_exactly_one_page() -> None:
    hook = _hook()
    categories = {spec["category"] for spec in hook.registry().values()}
    seen = _token_categories(hook)
    assert set(seen) == categories, f"tokens {sorted(seen)} vs registry {sorted(categories)}"
    assert all(n == 1 for n in seen.values()), f"a category is carded twice: {seen}"


def test_every_robot_renders_one_card() -> None:
    hook = _hook()
    out = hook.cards(sorted({spec["category"] for spec in hook.registry().values()}))
    assert out.count('<article class="robot-card"') == len(hook.registry())
    for name in hook.registry():
        assert f'Robot("{name}")' in out


def test_cards_reference_only_renders_that_exist() -> None:
    hook = _hook()
    out = hook.cards(sorted({spec["category"] for spec in hook.registry().values()}))
    for src in re.findall(r'src="[^"]*?(sim_render_[a-z0-9_]+\.png)"', out):
        assert (_REPO / "docs/assets" / src).is_file(), src
