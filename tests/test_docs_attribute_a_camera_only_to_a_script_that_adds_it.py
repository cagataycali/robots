# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A documented sentence that says a script mounts a camera links a script that does.

A camera recipe is the one part of a recording page a reader cannot check by
reading: ``add_camera`` returns ``status="error"`` rather than raising, so a
wrong mount body or a pose that frames nothing produces a rollout that records
from somewhere else and no traceback. Docs therefore point at a shipped script
as the worked version - "X records a walking robot this way" - and that pointer
is only worth following if the script really does it.

Measured on ``82169fa4``, before this module: ``docs/policies/wbc.md`` closed its
new "Recording it" subsection with

    [`examples/locomotion/scripted_g1.py`](...) does the pelvis mount before its
    first segment.

and that example contains zero ``add_camera`` calls - it records
``video={"camera": "default"}``, the very fixed view the paragraph above tells
the reader to stop using. The same paragraph attributed a pose
(``position=[1.7, -4.6, 1.4]``, ``fov=42``) to the clip rendered further down the
page; the pose appears nowhere in the repository, and that clip is rendered by
``examples/wbc/wbc_g1_torque_deploy.py`` through ``mujoco.Renderer`` with
``camera=-1``, the free camera. Both are invisible to the link grader
(``tests/test_markdown_links_resolve.py``): the paths resolve, the claims about
them do not hold.

What is graded, for each markdown link to a repository ``.py`` file whose
sentence claims a camera is added or mounted:

* the linked file calls ``add_camera`` at all;
* every pose literal the sentence shows in backticks (``position=[...]``,
  ``target=[...]``) appears in that file, so a pose cannot be invented for a
  script that uses another one;
* every namespaced mount body it names (``microduck/trunk_base``) appears there
  too, since the body is what makes the camera ride along.

Prose is not graded beyond that trigger, for the reason the sibling
``tests/test_docs_camera_names_are_addressable.py`` gives: a rule stated in one
paragraph and shown in twenty places drifts in the twenty. A sentence that makes
no camera claim about the script it links is none of this module's business.
"""

from __future__ import annotations

import re
from pathlib import Path

import strands_robots

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent

#: A markdown link to a Python file in this repository, written either as a
#: repo-relative path or as the ``blob/main`` URL the docs use for examples.
_LINK = re.compile(
    r"\[`?[^\]`]+?`?\]\((?:https://github\.com/strands-labs/robots/blob/main/)?([\w./-]+?\.py)\)",
)

#: The sentence claims the linked script places a camera, rather than merely
#: mentioning one: it names the door or the mounting.
_CLAIMS_A_CAMERA = re.compile(r"add_camera|mount(?:s|ed|ing)?\b")

#: ``position=[-2.6, -1.6, 1.1]`` and friends, inside backticks.
_POSE_LITERAL = re.compile(r"`(\w+=\[[^\]]*\])`")

#: A namespaced mount body such as ``microduck/trunk_base``, inside backticks.
#: Read only from a sentence that says something is mounted, so a backticked
#: directory path in a sentence about a fixed camera is not mistaken for a body.
_MOUNT_BODY = re.compile(r"`([a-z][\w-]*/[a-z][\w-]*)`")
_MOUNTS = re.compile(r"mount(?:s|ed|ing)?\b")


def _documentation_files() -> list[Path]:
    files = sorted((_REPO_ROOT / "docs").rglob("*.md"))
    readme = _REPO_ROOT / "README.md"
    return [*files, readme] if readme.is_file() else files


#: A sentence boundary: a full stop followed by any whitespace. Markdown wraps
#: prose, so a sentence ends at ".\n" as often as at ". " - reading only the
#: latter joined a page's sentences and credited each link with its neighbour's
#: pose.
_SENTENCE_END = re.compile(r"\.\s")


def _sentence_around(text: str, start: int, end: int) -> str:
    """The sentence holding ``text[start:end]``, bounded by a full stop or a blank line."""
    opens = max((m.end() for m in _SENTENCE_END.finditer(text, 0, start)), default=0)
    opens = max(opens, text.rfind("\n\n", 0, start) + 2, 0)
    closes = next((m.start() + 1 for m in _SENTENCE_END.finditer(text, end)), len(text))
    blank = text.find("\n\n", end)
    if blank != -1:
        closes = min(closes, blank)
    return " ".join(text[opens:closes].split())


def _camera_claims(files: list[Path] | None = None) -> list[tuple[str, str, str]]:
    """Every ``(where, script, sentence)`` that credits a script with a camera."""
    claims: list[tuple[str, str, str]] = []
    for path in files if files is not None else _documentation_files():
        text = path.read_text(encoding="utf-8")
        for match in _LINK.finditer(text):
            sentence = _sentence_around(text, match.start(), match.end())
            if not _CLAIMS_A_CAMERA.search(sentence):
                continue
            line = text.count("\n", 0, match.start()) + 1
            claims.append((f"{_relative(path)}:{line}", match.group(1), sentence))
    return claims


def _squeezed(text: str) -> str:
    return " ".join(text.split())


def _relative(path: Path) -> str:
    """``path`` under the repository root, or its own name if it lies outside."""
    return str(path.relative_to(_REPO_ROOT)) if path.is_relative_to(_REPO_ROOT) else path.name


def _offenders(claims: list[tuple[str, str, str]]) -> list[str]:
    """Every claim whose script does not hold up, in the order the pages state them."""
    offenders: list[str] = []
    for where, script, sentence in claims:
        source = _REPO_ROOT / script
        if not source.is_file():
            offenders.append(f"{where} credits {script}, which does not exist")
            continue
        body = _squeezed(source.read_text(encoding="utf-8"))
        if "add_camera" not in body:
            offenders.append(f"{where} credits {script} with a camera it never adds")
            continue
        bodies = _MOUNT_BODY.findall(sentence) if _MOUNTS.search(sentence) else []
        for literal in _POSE_LITERAL.findall(sentence) + bodies:
            if _squeezed(literal) not in body:
                offenders.append(f"{where} attributes {literal} to {script}, which does not use it")
    return offenders


def test_a_script_credited_with_a_camera_adds_the_camera_the_page_shows() -> None:
    offenders = _offenders(_camera_claims())
    assert not offenders, (
        "a page tells the reader to copy a camera recipe from a script that does "
        "something else, and add_camera fails by returning status=error, so the "
        "reader's rollout records from the wrong view with no traceback:\n  " + "\n  ".join(offenders)
    )


def test_the_reader_resolves_the_shapes_the_pages_are_written_in(tmp_path: Path) -> None:
    """An empty offender list and a reader that resolves nothing are the same list."""
    page = tmp_path / "page.md"
    page.write_text(
        "A `chase` camera mounted on `microduck/trunk_base` -\n"
        "[`examples/microduck/eval_rl_policy.py`](examples/microduck/eval_rl_policy.py)\n"
        "adds it with `position=[0.0, -0.8, 0.4]` before the rollout.\n\n"
        "[`examples/locomotion/scripted_g1.py`](https://github.com/strands-labs/robots/"
        "blob/main/examples/locomotion/scripted_g1.py) does the pelvis mount too.\n\n"
        "The clip was shot from `add_camera` at `position=[1.7, -4.6, 1.4]` in\n"
        "[`examples/kimodo/kimodo_g1_walking.py`](examples/kimodo/kimodo_g1_walking.py).\n",
        encoding="utf-8",
    )
    claims = _camera_claims([page])
    assert [script for _, script, _ in claims] == [
        "examples/microduck/eval_rl_policy.py",
        "examples/locomotion/scripted_g1.py",
        "examples/kimodo/kimodo_g1_walking.py",
    ], claims
    sentence = claims[0][2]
    assert _POSE_LITERAL.findall(sentence) == ["position=[0.0, -0.8, 0.4]"]
    assert _MOUNT_BODY.findall(sentence) == ["microduck/trunk_base"]
    # The checker accepts the script that does it, and names both ways a claim
    # fails: a script that adds no camera, and a pose it does not use.
    assert _offenders(claims) == [
        "page.md:5 credits examples/locomotion/scripted_g1.py with a camera it never adds",
        "page.md:8 attributes position=[1.7, -4.6, 1.4] to examples/kimodo/kimodo_g1_walking.py, which does not use it",
    ]
