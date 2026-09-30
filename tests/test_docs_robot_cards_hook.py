"""docs/hooks/robot_pages.py puts every registered robot on exactly one family page.

The catalog (``docs/robots/index.md``) carries one ``{{robot_cards}}`` token and
each family page (``docs/robots/<family>/index.md``) one ``{{robot_cards:<family>}}``
token instead of hand tables; this grader checks the tokens cover every registry
category once, that every robot renders one card and one page of its own with
its constructor line, and that the generated cards reference only thumbnails
that exist.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from tests._docs_hooks import docs_hook

_REPO = Path(__file__).resolve().parents[1]
_ROBOTS = _REPO / "docs" / "robots"


def _hook():  # noqa: ANN202 - the loaded hook module
    return docs_hook("robot_pages")


def _token_categories(hook) -> tuple[Counter[str], int]:  # noqa: ANN001 - the loaded hook module
    """Every family named by a ``{{robot_cards:<family>}}`` token, with its page count, and the bare-token count.

    Uses the hook's own token pattern rather than a copy of it, so a build that
    stops expanding a token cannot leave this grader counting it.
    """
    seen: Counter[str] = Counter()
    bare = 0
    for page in sorted(_ROBOTS.rglob("*.md")):
        for match in hook._TOKEN_CARDS.findall(page.read_text(encoding="utf-8")):
            if match:
                seen[match] += 1
            else:
                bare += 1
    return seen, bare


def test_every_registry_category_is_carded_on_exactly_one_family_page() -> None:
    hook = _hook()
    categories = {spec["category"] for spec in hook.registry().values()}
    seen, bare = _token_categories(hook)
    assert set(seen) == categories, f"tokens {sorted(seen)} vs registry {sorted(categories)}"
    assert all(n == 1 for n in seen.values()), f"a family is carded twice: {seen}"
    assert bare == 1, f"the catalog carries the bare {{{{robot_cards}}}} token once, found {bare}"
    for category in categories:
        page = _ROBOTS / category / "index.md"
        assert page.is_file(), f"no family page for {category!r}"
        assert f"{{{{robot_cards:{category}}}}}" in page.read_text(encoding="utf-8"), page


def test_every_robot_renders_one_card_and_one_page() -> None:
    hook = _hook()
    out = hook.cards(None, "")
    assert out.count('<article class="sr-robot"') == len(hook.registry())
    for name in hook.registry():
        assert f'href="robots/{name}/"' in out, f"the {name} card does not link to its page"
        page = _ROBOTS / f"{name}.md"
        assert page.is_file(), f"no generated page for {name}"
        assert f'Robot("{name}")' in page.read_text(encoding="utf-8"), f"{page.name} never shows the constructor line"


def test_family_cards_are_the_family_and_nothing_else() -> None:
    hook = _hook()
    for category in sorted({spec["category"] for spec in hook.registry().values()}):
        out = hook.cards(category, "")
        families = set(re.findall(r'data-family="([a-z_]+)"', out))
        assert families == {category}, f"cards for {category!r} carry {sorted(families)}"


def test_cards_reference_only_thumbnails_that_exist() -> None:
    hook = _hook()
    out = hook.cards(None, "")
    sources = re.findall(r'<img src="([^"]+)"', out)
    assert sources, "no card carries a thumbnail; the guard would prove nothing"
    for src in sources:
        assert (_REPO / "docs" / src).is_file(), src


def test_each_native_driver_is_described_once_and_every_robot_page_links_there() -> None:
    """The drivers page renders every ``DRIVERS`` entry once; a robot page links to it instead of restating it."""
    hook = _hook()
    drivers_page = _REPO / "docs" / "learn" / "hardware" / "drivers.md"
    assert hook._TOKEN_FACTS.search(drivers_page.read_text(encoding="utf-8")), "drivers.md lost {{driver_facts}}"
    rendered = hook.driver_facts()
    assert re.findall(r"^### (\w+)$", rendered, re.M) == list(hook.DRIVERS)
    for name in hook.registry():
        text = (_ROBOTS / f"{name}.md").read_text(encoding="utf-8")
        cls = hook._load_coverage().row(name).native_driver
        assert "| Other kwargs |" not in text, f"{name}.md restates its driver's facts"
        if cls:
            assert f"](../learn/hardware/drivers.md#{cls.lower()})" in text, f"{name}.md does not link {cls}"


def test_every_robot_page_names_its_own_chips_and_the_build_renders_them() -> None:
    """A robot page carries ``{{robot_chips:<its name>}}``, not chip markup, and the token expands to the chips."""
    hook = _hook()
    for name in hook.registry():
        text = (_ROBOTS / f"{name}.md").read_text(encoding="utf-8")
        tokens = hook._TOKEN_CHIPS.findall(text)
        assert tokens == [name], f"{name}.md carries chip tokens {tokens}, expected [{name!r}]"
        assert "sr-chip" not in text, f"{name}.md restates chip markup the token renders"
        rendered = hook.substitute(text, "../")
        assert "{{robot_chips" not in rendered and hook.chips(name) in rendered, f"{name}.md chips did not render"
