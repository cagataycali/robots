"""The GR00T page's ``data_configs`` block must name every shipped config.

``docs/reference/policies/groot.md`` carries a fenced block under the ``## 27
data_configs`` heading that is the reader's catalog of ``data_config=`` values.
``strands_robots/policies/groot/data_configs.json`` is the vocabulary owner, so
the block is graded against it rather than against a copied list: a config
shipped there and absent from the page is a value the reader cannot discover.
A trailing ``*`` in the block stands for every config sharing that prefix.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PAGE = _REPO_ROOT / "docs" / "reference" / "policies" / "groot.md"
_DATA_CONFIGS = _REPO_ROOT / "strands_robots" / "policies" / "groot" / "data_configs.json"


def _page_block_tokens() -> list[str]:
    """Whitespace-split tokens of the fenced block under the data_configs heading."""
    text = _PAGE.read_text()
    match = re.search(r"^## \d+ data_configs\s*\n```\n(.*?)\n```", text, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, "docs/reference/policies/groot.md has no fenced block under the data_configs heading"
    return match.group(1).split()


def _named(config: str, tokens: list[str]) -> bool:
    return any(config == token or (token.endswith("*") and config.startswith(token[:-1])) for token in tokens)


def test_every_shipped_data_config_is_named_on_the_page() -> None:
    configs = sorted(json.loads(_DATA_CONFIGS.read_text())["configs"])
    tokens = _page_block_tokens()
    missing = [config for config in configs if not _named(config, tokens)]
    assert missing == [], (
        f"data_configs.json ships {missing} but the docs/reference/policies/groot.md data_configs block does not name them"
    )
