# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A documented sentence that says a script mounts a camera links a script that does.

A camera recipe is the one part of a recording page a reader cannot check by
reading: ``add_camera`` returns ``status="error"`` rather than raising, so a
wrong mount body or a pose that frames nothing produces a rollout that records
from somewhere else and no traceback. Docs therefore point at a shipped script
as the worked version - "X records a walking robot this way" - and that pointer
is only worth following if the script really does it.

Measured on ``82169fa4``, before this module: a draft of ``docs/policies/wbc.md``
closed its new "Recording it" subsection with

    [`examples/locomotion/scripted_g1.py`](...) does the pelvis mount before its
    first segment.

and that example then contained zero ``add_camera`` calls - it recorded
``video={"camera": "default"}``, the very fixed view the paragraph above tells
the reader to stop using. That one was settled from the other end: #3708 gave
``scripted_g1.py`` the pelvis mount the sentence had promised, which is the
argument for grading the pair rather than either half - the same claim was
false in the page and true in the script within one day, and nothing said so.
The same paragraph attributed a pose (``position=[1.7, -4.6, 1.4]``, ``fov=42``)
to the clip rendered further down the page; that pose appears nowhere in the
repository, and the clip is rendered by
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


def _documentation_files(root: Path) -> list[Path]:
    files = sorted((root / "docs").rglob("*.md"))
    readme = root / "README.md"
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


def _camera_claims(files: list[Path] | None = None, root: Path = _REPO_ROOT) -> list[tuple[str, str, str]]:
    """Every ``(where, script, sentence)`` that credits a script with a camera."""
    claims: list[tuple[str, str, str]] = []
    for path in files if files is not None else _documentation_files(root):
        text = path.read_text(encoding="utf-8")
        for match in _LINK.finditer(text):
            sentence = _sentence_around(text, match.start(), match.end())
            if not _CLAIMS_A_CAMERA.search(sentence):
                continue
            line = text.count("\n", 0, match.start()) + 1
            claims.append((f"{_relative(path, root)}:{line}", match.group(1), sentence))
    return claims


def _squeezed(text: str) -> str:
    return " ".join(text.split())


def _relative(path: Path, root: Path) -> str:
    """``path`` under ``root``, or its own name if it lies outside."""
    return str(path.relative_to(root)) if path.is_relative_to(root) else path.name


def _offenders(claims: list[tuple[str, str, str]], root: Path = _REPO_ROOT) -> list[str]:
    """Every claim whose script does not hold up, in the order the pages state them."""
    offenders: list[str] = []
    for where, script, sentence in claims:
        source = root / script
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
    """An empty offender list and a reader that resolves nothing are the same list.

    The tree is written here rather than pointed at shipped examples. An earlier
    revision made its negative case a real script - ``scripted_g1.py``, which
    added no camera - and #3708 then gave that script a pelvis mount, so an
    offender this test pins stopped being produced and the reader was reported
    broken by an unrelated improvement to an example. A grader's own shapes have
    to be fixed for the assertion to mean anything; whether the shipped pages
    hold up is the other test in this module, which reads the real tree.
    """
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "mounts.py").write_text(
        'sim.add_camera(name="chase", parent_body="microduck/trunk_base",\n               position=[0.0, -0.8, 0.4])\n',
        encoding="utf-8",
    )
    (tmp_path / "examples" / "fixed.py").write_text(
        'sim.add_camera(name="side", position=[3.0, 0.0, 1.2])\n', encoding="utf-8"
    )
    (tmp_path / "examples" / "no_camera.py").write_text(
        'sim.run_policy(video={"camera": "default"})\n', encoding="utf-8"
    )
    page = tmp_path / "page.md"
    page.write_text(
        "A `chase` camera mounted on `microduck/trunk_base` -\n"
        "[`examples/mounts.py`](examples/mounts.py)\n"
        "adds it with `position=[0.0, -0.8, 0.4]` before the rollout.\n\n"
        "[`examples/no_camera.py`](https://github.com/strands-labs/robots/"
        "blob/main/examples/no_camera.py) does the pelvis mount too.\n\n"
        "The clip was shot from `add_camera` at `position=[1.7, -4.6, 1.4]`, a pose\n"
        "this page keeps beside the others under `docs/policies`, in\n"
        "[`examples/fixed.py`](examples/fixed.py).\n\n"
        "[`examples/gone.py`](examples/gone.py) mounts one as well.\n",
        encoding="utf-8",
    )
    claims = _camera_claims([page], root=tmp_path)
    assert [script for _, script, _ in claims] == [
        "examples/mounts.py",
        "examples/no_camera.py",
        "examples/fixed.py",
        "examples/gone.py",
    ], claims
    sentence = claims[0][2]
    assert _POSE_LITERAL.findall(sentence) == ["position=[0.0, -0.8, 0.4]"]
    assert _MOUNT_BODY.findall(sentence) == ["microduck/trunk_base"]
    # The checker accepts the script that does it, and names all three ways a
    # claim fails: a script that adds no camera, a pose it does not use, and a
    # path that is not there at all. `docs/policies` sits in a sentence about a
    # fixed camera, so it is a directory the page mentions and not a mount body:
    # reading it as one would credit examples/fixed.py with a body too.
    assert _offenders(claims, root=tmp_path) == [
        "page.md:5 credits examples/no_camera.py with a camera it never adds",
        "page.md:9 attributes position=[1.7, -4.6, 1.4] to examples/fixed.py, which does not use it",
        "page.md:11 credits examples/gone.py, which does not exist",
    ]
