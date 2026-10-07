"""Every trainer name the README and docs pass to ``create_trainer`` is registered.

A page that says ``create_trainer("lerobot")`` sends the reader straight into a
``ValueError``; the registry, not the prose, decides which names exist.
"""

import re
from pathlib import Path

from strands_robots.training import list_trainers

ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
# A call's first argument, including the `"ppo" | "fast_sac"` alternation form,
# plus the bare quoted names that follow it in the README's Train row.
CALL = re.compile(r'create_trainer\(((?:"[a-z0-9_ ]+"\s*\|?\s*)+)')
README_TRAIN_ROW = re.compile(r"^\| \*\*Train\*\*.*$", re.MULTILINE)
NAME = re.compile(r'"([^"]+)"')


def test_every_trainer_name_in_the_docs_is_one_create_trainer_takes() -> None:
    registered = set(list_trainers())
    named: dict[str, str] = {}
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        spans = [m.group(1) for m in CALL.finditer(text)]
        if page.name == "README.md":
            spans += README_TRAIN_ROW.findall(text)
        for span in spans:
            for name in NAME.findall(span):
                named.setdefault(name, str(page.relative_to(ROOT)))
    assert named, "no create_trainer call found; the pattern no longer matches the docs"
    unknown = {name: page for name, page in named.items() if name not in registered}
    assert not unknown, f"docs name trainers create_trainer refuses: {unknown}"
