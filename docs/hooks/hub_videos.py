#!/usr/bin/env python3
"""Expand ``{{hub_videos:<set>}}`` into a grid of the Hub's own playback clips, numbers included.

A set is an entry in ``docs/hooks/data/hub_videos.json``: a Hugging Face collection and
the model repos in it. For each repo the figure's ``<video>`` streams the repo's
``playback.mp4`` with ``frame.png`` as poster (nothing is copied into this tree), and the
caption is built from numbers this module READ from the repo's ``verify.json``,
``record.json`` and ``train_curve.json`` when ``--refresh`` last ran. The page spends one
word on the token; nobody types a success rate by hand::

    {{hub_videos:isaaclab-zoo}}

``python docs/hooks/hub_videos.py --refresh`` re-reads every repo over the network and
rewrites the data file; the build itself is offline. ``--check`` reports a set whose
repos lack a cached row. Figures are ``<figure class="sr-sim">`` like the sim clips, in a
``<div class="sr-hub">`` grid, every video ``preload="none"``.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import urllib.request
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.hub_videos")

_DOCS = Path(__file__).resolve().parents[1]
DATA = _DOCS / "hooks" / "data" / "hub_videos.json"
HUB = "https://huggingface.co"
_TOKEN = re.compile(r"\{\{hub_videos:([A-Za-z0-9_-]+)\}\}")

#: The files a repo row is read from, and the keys kept out of each.
FILES = ("verify.json", "record.json", "train_curve.json")


def load() -> dict:
    """The data file, ``{}`` when it does not exist yet."""
    return json.loads(DATA.read_text(encoding="utf-8")) if DATA.is_file() else {}


def resolve(repo: str, filename: str) -> str:
    """The Hub URL of one file at a model repo's ``main``."""
    return f"{HUB}/{repo}/resolve/main/{filename}"


def _fetch_json(url: str) -> dict | list:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - https Hub URL built above
        return json.load(resp)


def read_row(repo: str) -> dict:
    """One repo's numbers, straight from its ``verify.json`` / ``record.json`` / ``train_curve.json``."""
    verify = _fetch_json(resolve(repo, "verify.json"))
    record = _fetch_json(resolve(repo, "record.json"))
    curve = _fetch_json(resolve(repo, "train_curve.json"))
    last = curve[-1]
    return {
        "task": record.get("task"),
        "task_text": (verify.get("tasks") or [""])[0],
        "physics": next((o.split("=", 1)[1] for o in record.get("overrides", []) if o.startswith("physics=")), None),
        "episodes": verify.get("num_episodes"),
        "frames": verify.get("num_frames"),
        "fps": verify.get("fps"),
        "action_dim": verify.get("action.dim"),
        "iterations": int(last["iter"]) + 1,
        "success_rate": last.get("Metrics/success_rate"),
        "mean_reward": last.get("Mean reward"),
        "verified": bool(verify.get("OK")),
    }


def refresh(data: dict) -> dict:
    """Re-read every repo of every set; the labels (``name``) are kept, the numbers replaced."""
    for set_id, entry in data.items():
        for item in entry["items"]:
            item.update(read_row(item["repo"]))
            print(f"{set_id}: {item['repo']} success {item['success_rate']} reward {item['mean_reward']}")
    return data


def caption(item: dict) -> str:
    """The sentence under one clip, every number from the cached row."""
    parts = [f"{item['name']}, {item['task']} on {item.get('physics') or 'Isaac Sim'}:"]
    if item.get("success_rate") is not None:
        parts.append(f"success {item['success_rate'] * 100:.0f} percent,")
    if item.get("mean_reward") is not None:
        parts.append(f"mean reward {item['mean_reward']:.1f}")
    parts.append(f"after {item['iterations']:,} iterations;")
    parts.append(f"{item['episodes']} recorded episodes, {item['fps']} fps")
    return " ".join(parts)


def figure(item: dict) -> str:
    """One ``<figure>``: the Hub's mp4 and poster, the caption, nothing preloaded."""
    repo = item["repo"]
    text = html.escape(caption(item))
    return (
        f'<figure class="sr-sim"><video src="{resolve(repo, "playback.mp4")}" '
        f'poster="{resolve(repo, "frame.png")}" muted loop playsinline controls preload="none"></video>'
        f'<figcaption>{text} (<a href="{HUB}/{repo}">{html.escape(repo.split("/")[-1])}</a>)</figcaption></figure>'
    )


def grid_html(set_id: str, data: dict | None = None) -> str | None:
    """The grid for one set, or None when the set or a repo's numbers are missing."""
    data = load() if data is None else data
    entry = data.get(set_id)
    if not entry or any(item.get("iterations") is None for item in entry["items"]):
        return None
    figures = "".join(figure(item) for item in entry["items"])
    link = f'<p class="sr-hub-link"><a href="{entry["collection"]}">{html.escape(entry["title"])}</a> on the Hub.</p>'
    return f'<div class="sr-hub">{figures}</div>{link}'


def substitute(markdown: str, page_path: str = "<string>") -> str:
    """Replace every ``{{hub_videos:...}}`` token; warn (strict build: fail) on an unknown set."""

    def _one(match: re.Match[str]) -> str:
        out = grid_html(match.group(1))
        if out is None:
            log.warning("%s: {{hub_videos:%s}} has no rows in %s", page_path, match.group(1), DATA.name)
            return match.group(0)
        return out

    return _TOKEN.sub(_one, markdown)


def referenced_sets(markdown: str) -> set[str]:
    """Every set a page references."""
    return set(_TOKEN.findall(markdown))


def check(data: dict) -> list[str]:
    """Problems ``--check`` reports: a set with no items, an item with no cached numbers."""
    problems = []
    for set_id, entry in data.items():
        if not entry.get("items"):
            problems.append(f"{set_id}: no items")
        for item in entry.get("items", []):
            if item.get("iterations") is None or item.get("episodes") is None:
                problems.append(f"{set_id}: {item.get('repo')} has no cached numbers; run --refresh")
    return problems


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """mkdocs hook entry point."""
    return substitute(markdown, page.file.src_path)


def main(argv: list[str] | None = None) -> int:
    """``--refresh`` rewrites the data file from the Hub; ``--check`` reports missing numbers."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--refresh", action="store_true", help="re-read every repo from the Hub and rewrite the data file"
    )
    parser.add_argument("--check", action="store_true", help="report sets whose repos lack cached numbers")
    args = parser.parse_args(argv)
    data = load()
    if args.refresh:
        DATA.write_text(json.dumps(refresh(data), indent=2) + "\n", encoding="utf-8")
    problems = check(data if not args.refresh else load())
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
