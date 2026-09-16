# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The build reads its version from release tags only.

``[tool.hatch.version] source = "vcs"`` asks git for the nearest tag. Left at
its default the describe command matches ``*[0-9]*``, so any tag containing a
digit is a version candidate, and a clone that also carries a fork's artifact
tags fails or misreports its own version two ways:

* ``artifact-902-action-keys`` cannot be parsed at all, so ``pip install -e .``
  fails outright with ``Can't parse version from tag``;
* ``artifact-bench-video-1783229263`` parses, and the package quietly takes
  version ``1783229263`` - above every real release, and a wheel built from
  such a clone shadows the real one.

A ``tag_regex`` cannot repair either: the regex runs after describe has already
chosen the tag. The describe command in ``raw-options`` restricts candidates to
``v<digit>...`` so only release tags are ever asked to carry a version. ``v*``
would not be enough - ``visual-d035`` and ``viz-...`` also start with a ``v``.

The pins below run the command read back from ``pyproject.toml`` against a real
repository tagged the way a fork is, so git's own glob decides which tags are
candidates rather than a restatement of the pattern here.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

#: Real non-version tags carried by forks of this repository. Each contains a
#: digit, so each is a version candidate under the default ``--match``; the two
#: ending in digits are the ones that quietly parse as a version.
ARTIFACT_TAGS = (
    "artifact-902-action-keys",
    "artifacts-pc020-obstruction",
    "artifact-bench-video-1783229263",
    "viz-move-object-orientation-1783155746",
    "visual-d035",
)

#: The describe command setuptools-scm runs when nothing overrides it, and the
#: one whose ``--match`` chose an artifact tag.
DEFAULT_COMMAND = ("describe", "--dirty", "--tags", "--long", "--match", "*[0-9]*")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _describe_arguments() -> list[str]:
    """The describe command the build runs, read back from ``pyproject.toml``."""
    with PYPROJECT.open("rb") as handle:
        version = tomllib.load(handle)["tool"]["hatch"]["version"]
    assert version["source"] == "vcs", version
    raw_options = version.get("raw-options", {})
    assert "git_describe_command" in raw_options, (
        "the build must pin its own git describe command; on the default, any "
        f"tag containing a digit carries the version: {raw_options}"
    )
    command = list(raw_options["git_describe_command"])
    assert command[:2] == ["git", "describe"], command
    return command[1:]


def _repository_tagged_like_a_fork(tmp_path: Path) -> Path:
    """A release tag, then a fork's artifact tags on the commits after it."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "checks@example.invalid")
    _git(root, "config", "user.name", "Release Tag Tests")
    _git(root, "config", "commit.gpgsign", "false")
    for message in ("the release", *ARTIFACT_TAGS):
        (root / "README.md").write_text(f"# {message}\n", encoding="utf-8")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", message)
        _git(root, "tag", "v0.5.1" if message == "the release" else message)
    return root


def test_describe_reads_the_release_tag_past_a_forks_artifact_tags(tmp_path: Path) -> None:
    """The artifact tags are nearer to HEAD, and the release tag still wins."""
    repo = _repository_tagged_like_a_fork(tmp_path)
    on_the_default = _git(repo, *DEFAULT_COMMAND)
    assert on_the_default.startswith(f"{ARTIFACT_TAGS[-1]}-"), on_the_default

    described = _git(repo, *_describe_arguments())
    assert described.startswith("v0.5.1-"), described


def test_a_release_tag_is_the_version_in_every_shape_a_release_takes(tmp_path: Path) -> None:
    """Patch, minor, major and pre-release tags are all candidates."""
    repo = _repository_tagged_like_a_fork(tmp_path)
    arguments = _describe_arguments()
    for release_tag in ("v0.5.2", "v0.6.0", "v1.0.0", "v0.5.2rc1"):
        _git(repo, "tag", release_tag)
        described = _git(repo, *arguments)
        _git(repo, "tag", "-d", release_tag)
        assert described.startswith(f"{release_tag}-0-g"), (release_tag, described)
