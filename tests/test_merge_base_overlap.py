"""Contract pins for the merge-base overlap check.

``scripts/check_merge_base_overlap.py`` exists because four green signals are not
enough to merge safely. #1766 and #1763 both edited
``strands_robots/simulation/mujoco/scene_ops.py``; #1766 landed first; #1763 then
read ``APPROVED`` / ``SUCCESS`` / ``MERGEABLE`` / ``CLEAN`` and its squash still
broke ``main``, because it carried a premise test asserting the defect #1766 had
just fixed. Every one of those signals is computed against the base the branch
was tested on, so none of them can see it.

The checks below pin the three properties that make the script worth having:

- it **flags** the #1763/#1766 topology, replayed here as real commits in a real
  repository rather than as a hand-built path set;
- it is **self-clearing** -- merging the base branch makes it pass, so the
  remedy it asks for is the remedy that satisfies it, with no override to add;
- it is **quiet** where an overlap cannot matter: disjoint edits and prose-only
  overlaps do not block.

The git-topology tests are the load-bearing ones. The overlap is a statement
about merge bases, and a merge base is exactly the thing a hand-built fixture
would assume rather than exercise -- including the one way to get this wrong in
CI, pinned by ``test_a_merge_commit_head_defeats_the_check``.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_merge_base_overlap.py"

#: The file both #1766 and #1763 edited. Used verbatim in the replay so the
#: pinned scenario is recognisable as the incident it came from.
_SHARED = "strands_robots/simulation/mujoco/scene_ops.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_merge_base_overlap", _SCRIPT_PATH)
    assert spec and spec.loader, f"cannot load {_SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_merge_base_overlap"] = module
    spec.loader.exec_module(module)
    return module


check = _load_module()


# --- git fixtures ---------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def _write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


#: A file with enough distinct lines that two branches can edit it in regions far
#: enough apart for git to merge them without a conflict -- which is the whole
#: point of the incident: the text merged cleanly and the meaning did not.
_SHARED_BODY = "\n".join(f"line {index}" for index in range(1, 41)) + "\n"


def _edit_line(repo: Path, relative: str, line_number: int, text: str) -> None:
    path = repo / relative
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[line_number - 1] = text
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository on ``main`` with one commit, ready to branch from."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "checks@example.invalid")
    _git(root, "config", "user.name", "Merge Base Overlap Tests")
    _git(root, "config", "commit.gpgsign", "false")
    _write(root, _SHARED, _SHARED_BODY)
    _write(root, "docs/simulation/world-building.md", "# World building\n\nOriginal prose.\n")
    _commit(root, "initial commit")
    return root


def _branch_editing_shared(repo: Path) -> None:
    """Branch off ``main`` and edit the shared file, as #1763 did."""
    _git(repo, "checkout", "-q", "-b", "pr")
    _edit_line(repo, _SHARED, 40, "line 40  # edited by the pull request")
    _write(repo, "tests/simulation/mujoco/test_add_robot_preserves_scene_state.py", "# premise test\n")
    _commit(repo, "the pull request's commit")


def _land_on_main_editing_shared(repo: Path) -> None:
    """Land a commit on ``main`` touching the shared file, as #1766 did."""
    _git(repo, "checkout", "-q", "main")
    _edit_line(repo, _SHARED, 1, "line 1  # edited by the commit that landed first")
    _commit(repo, "the commit that landed on main first")
    _git(repo, "checkout", "-q", "pr")


def _run_at(repo: Path, head: str) -> int:
    """Check a named commit, which is what CI does: the head SHA, never ``HEAD``."""
    return int(check.main(["--repo", str(repo), "--base-ref", "main", "--head", head]))


def _run(repo: Path) -> int:
    return _run_at(repo, "HEAD")


# --- the incident, replayed -----------------------------------------------------


def test_the_1763_1766_pair_is_flagged(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The acceptance criterion: replayed, the pair that broke main is caught.

    Both branches edit the shared file in regions git merges without complaint,
    so this is not a conflict the existing gate would have stopped.
    """
    _branch_editing_shared(repo)
    _land_on_main_editing_shared(repo)

    assert _run(repo) == 1, "an untested combination must not report success"
    out = capsys.readouterr().out
    assert _SHARED in out, "the report must name the overlapping path"
    assert "merge" in out and "main" in out, "the report must state the remedy"


def test_the_pair_merges_without_a_text_conflict(repo: Path) -> None:
    """Non-vacuity for the test above: git itself has nothing to report here.

    If the two edits conflicted, the existing merge gate would already block the
    merge and this check would be redundant. The incident happened precisely
    because they did not.
    """
    _branch_editing_shared(repo)
    _land_on_main_editing_shared(repo)

    _git(repo, "merge", "--no-edit", "-q", "main")  # raises CalledProcessError on conflict
    assert "line 1  # edited by the commit that landed first" in (repo / _SHARED).read_text(encoding="utf-8")
    assert "line 40  # edited by the pull request" in (repo / _SHARED).read_text(encoding="utf-8")


def test_merging_the_base_clears_the_check(repo: Path) -> None:
    """Self-clearing: the remedy the report asks for is what makes it pass.

    This is why the check needs no bypass label. Merging the base advances the
    merge base to the base tip, which empties the base-side path set, and the
    re-run then happens against a base containing the landed commits.
    """
    _branch_editing_shared(repo)
    _land_on_main_editing_shared(repo)
    assert _run(repo) == 1

    _git(repo, "merge", "--no-edit", "-q", "main")

    assert _run(repo) == 0, "after merging the base there is nothing untested left to flag"


# --- where an overlap cannot matter ---------------------------------------------


def test_an_unmoved_base_does_not_overlap(repo: Path) -> None:
    _branch_editing_shared(repo)
    assert _run(repo) == 0


def test_disjoint_edits_do_not_overlap(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A base that moved under *other* files invalidates nothing."""
    _branch_editing_shared(repo)
    _git(repo, "checkout", "-q", "main")
    _write(repo, "strands_robots/policies/mock.py", "# unrelated landing\n")
    _commit(repo, "an unrelated commit on main")
    _git(repo, "checkout", "-q", "pr")

    assert _run(repo) == 0
    assert "No overlap" in capsys.readouterr().out


def test_a_prose_only_overlap_is_reported_but_does_not_block(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Documentation cannot change what the suite does, so it does not gate."""
    doc = "docs/simulation/world-building.md"
    _git(repo, "checkout", "-q", "-b", "pr")
    _write(repo, doc, "# World building\n\nOriginal prose.\n\nAdded by the pull request.\n")
    _commit(repo, "docs from the pull request")

    _git(repo, "checkout", "-q", "main")
    _write(repo, doc, "# World building\n\nRewritten on main.\n")
    _commit(repo, "docs on main")
    _git(repo, "checkout", "-q", "pr")

    assert _run(repo) == 0, "a prose overlap must not demand a full re-run"
    out = capsys.readouterr().out
    assert doc in out, "it is still reported - the reader may want to know"
    assert "not blocking" in out


# --- the way to get this wrong in CI -------------------------------------------


def test_a_merge_commit_head_defeats_the_check(repo: Path) -> None:
    """Pins why the workflow must check out the head SHA, not the merge commit.

    ``actions/checkout`` defaults to ``refs/pull/<n>/merge`` on a pull request.
    That commit already contains the base tip, so the merge base *is* the base
    tip, the base-side set is empty, and the check would pass unconditionally --
    silently, with no signal that it had stopped testing anything.
    """
    _branch_editing_shared(repo)
    _land_on_main_editing_shared(repo)
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert _run(repo) == 1

    _git(repo, "checkout", "-q", "-b", "simulated-merge-ref")
    _git(repo, "merge", "--no-edit", "-q", "main")
    merge_commit = _git(repo, "rev-parse", "HEAD").strip()
    assert merge_commit != head

    against_merge_commit = int(check.main(["--repo", str(repo), "--base-ref", "main", "--head", merge_commit]))
    assert against_merge_commit == 0, (
        "documents the trap: run against the merge commit and the check is vacuous, "
        "which is why the workflow pins the head SHA"
    )


def test_a_rename_on_the_base_still_overlaps_the_old_path(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--no-renames`` keeps a renamed file's old name in the base-side set."""
    _branch_editing_shared(repo)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "mv", _SHARED, "strands_robots/simulation/mujoco/scene_operations.py")
    _commit(repo, "rename on main")
    _git(repo, "checkout", "-q", "pr")

    assert _run(repo) == 1
    assert _SHARED in capsys.readouterr().out


def test_an_unresolvable_base_ref_fails_loudly(repo: Path) -> None:
    """A check that cannot compute its answer must not report the reassuring one."""
    _branch_editing_shared(repo)
    assert int(check.main(["--repo", str(repo), "--base-ref", "no-such-branch", "--head", "HEAD"])) == 1


def test_the_remote_tracking_ref_wins_over_a_local_branch(repo: Path) -> None:
    """CI has only ``origin/<base>``, so that is what must be preferred.

    Pinned by pointing ``origin/main`` at a commit that overlaps while the local
    ``main`` does not: the check must follow the remote-tracking ref.
    """
    _branch_editing_shared(repo)
    _git(repo, "checkout", "-q", "main")
    _edit_line(repo, _SHARED, 1, "line 1  # only on the remote-tracking ref")
    landed = _commit(repo, "a commit reachable only via origin/main")
    _git(repo, "update-ref", "refs/remotes/origin/main", landed)
    _git(repo, "reset", "-q", "--hard", "HEAD~1")  # local main no longer has it
    _git(repo, "checkout", "-q", "pr")

    assert check.resolve_base_ref("main", repo=repo) == "origin/main"
    assert _run(repo) == 1, "the overlap is only visible through origin/main"


# --- the pure core --------------------------------------------------------------


def test_overlap_is_the_sorted_intersection() -> None:
    assert check.overlapping_paths(["b.py", "a.py", "c.py"], ["c.py", "a.py"]) == ("a.py", "c.py")


def test_overlap_is_empty_for_disjoint_sets() -> None:
    assert check.overlapping_paths(["a.py"], ["b.py"]) == ()


@pytest.mark.parametrize("path", ["CHANGELOG.md", "docs/guide.md", "notes.rst", "requirements.txt", "A.MD"])
def test_prose_paths_are_classified_as_prose(path: str) -> None:
    assert check.is_prose(path)


@pytest.mark.parametrize(
    "path",
    [
        "strands_robots/simulation/mujoco/scene_ops.py",
        "tests/test_thing.py",
        "strands_robots/registry/robots.json",
        "pyproject.toml",
        "mkdocs.yml",
        ".github/workflows/test-lint.yml",
    ],
)
def test_behaviour_bearing_paths_are_not_classified_as_prose(path: str) -> None:
    """Non-vacuity: the prose carve-out must not swallow anything that runs."""
    assert not check.is_prose(path)


def test_partition_splits_and_sorts() -> None:
    blocking, prose = check.partition_overlap(["z.md", "b.py", "a.py", "a.md", "b.py"])
    assert blocking == ("a.py", "b.py"), "deduplicated and sorted"
    assert prose == ("a.md", "z.md")


def test_report_names_every_blocking_path_and_the_remedy() -> None:
    report = check.render_report(
        base_ref="main",
        merge_base_sha="32dc3f5b2ca5f226842e4f8e40aaa8e64108e383",
        blocking=[_SHARED],
        prose=["CHANGELOG.md"],
        base_change_count=5,
    )
    assert _SHARED in report
    assert "CHANGELOG.md" in report
    assert "not blocking" in report
    assert "32dc3f5b" in report, "the merge base is what makes the claim checkable"
    assert "merge" in report


def test_report_says_so_when_there_is_no_overlap() -> None:
    report = check.render_report(
        base_ref="main",
        merge_base_sha="0123456789abcdef",
        blocking=[],
        prose=[],
        base_change_count=0,
    )
    assert "No overlap" in report


def test_no_emoji_in_the_script_or_its_output() -> None:
    """Project rule: plain ASCII in anything an agent reads programmatically."""
    source = _SCRIPT_PATH.read_text(encoding="utf-8")
    offenders = [(index, char) for index, char in enumerate(source) if ord(char) > 0x7F]
    assert offenders == [], f"non-ASCII in {_SCRIPT_PATH.name}: {offenders[:5]}"


# --- the tree the gate is read from ---------------------------------------------


def test_the_overlap_does_not_depend_on_the_checked_out_tree(repo: Path) -> None:
    """Same commits, same overlap, whichever tree happens to be checked out.

    This is the property that lets CI run the script from the *base* checkout while
    judging the branch, and CI must: a branch that forked before a gate landed
    carries no copy of that gate's script, so running it out of the head tree exits
    2 before the check begins -- a red X indistinguishable from exit 1, which the
    script reserves for a real untested overlap. That failure was measured on the
    sibling changelog gate in issue #1791; this one shares its shape and is latent
    only because every currently open branch happens to postdate it.

    It holds because nothing here reads the working tree: the path sets come from
    ``git diff --name-only <a>..<b>``, so only reachability matters.
    """
    _branch_editing_shared(repo)
    _land_on_main_editing_shared(repo)
    head = _git(repo, "rev-parse", "HEAD").strip()

    # The branch checked out, as the workflow used to do.
    assert _run_at(repo, head) == 1

    # The base checked out, as the workflow now does. The branch's edit is not in
    # the tree on disk, and the overlap is still reported.
    _git(repo, "checkout", "-q", "main")
    assert "line 40  # edited by the pull request" not in (repo / _SHARED).read_text(encoding="utf-8")
    assert _run_at(repo, head) == 1


def test_a_branch_with_no_overlap_still_passes_from_the_base_checkout(repo: Path) -> None:
    """The other direction: reading from the base does not fail every branch.

    Without this, the test above is satisfied by a check that refuses everything.
    """
    _branch_editing_shared(repo)
    head = _git(repo, "rev-parse", "HEAD").strip()

    _git(repo, "checkout", "-q", "main")
    assert _run_at(repo, head) == 0


def test_the_workflow_reads_the_script_from_the_base_and_names_the_head() -> None:
    """The workflow must not require its own script in the tree under review.

    A ``pull_request`` workflow definition is read from the merge commit, so a gate
    runs against heads that contain neither it nor its script. The sibling changelog
    gate exited 2 for exactly that reason (issue #1791); this workflow had the same
    shape.
    """
    workflow = (_REPO_ROOT / ".github" / "workflows" / "merge-base-overlap.yml").read_text(encoding="utf-8")

    assert "ref: ${{ github.base_ref }}" in workflow, "the gate's script must come from the base branch"
    assert "ref: ${{ github.event.pull_request.head.sha }}" not in workflow, (
        "checking the head out is what made the script's presence a precondition"
    )
    assert "HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in workflow
    assert '--head "$HEAD_SHA"' in workflow, "the commit under test is named, not checked out"


# --------------------------------------------------------------------------
# The sweep over pairs of open pull requests.
#
# Everything above compares one branch against its base. Two *open* pull
# requests editing the same file have the same failure mode and no signal at
# all: neither has an `M..base` overlap, so the check reads green on both.
#
# Measured over the 19 open pull requests on 2026-08-13: 171 pairs, 7 sharing a
# path, 3 prose-only, 4 worth reporting. Two of the four had already been found
# by hand at the cost of a composition run each -- #2233/#2235 share
# `tests/simulation/isaac/test_motion_primitives.py`, where one carries a strict
# xfail for the defect the other fixes, and #2224/#2227 share
# `strands_robots/simulation/safe_output.py`. These pin the caller, and in
# particular that an incomplete read is named rather than folded in as clean.
# --------------------------------------------------------------------------


def _pull(number: int, *, draft: bool = False) -> dict[str, Any]:
    return {"number": number, "draft": draft}


def _files(*names: str) -> list[dict[str, Any]]:
    return [{"filename": name} for name in names]


def _sweep_fixture(
    monkeypatch: pytest.MonkeyPatch,
    pulls: list[dict[str, Any]],
    files: dict[int, list[dict[str, Any]]],
    *,
    unreadable: tuple[int, ...] = (),
) -> None:
    """Route the two reads a sweep makes to plain fixtures.

    ``unreadable`` names pull requests whose file list raises, standing in for a
    rate limit on one branch while the rest of the sweep proceeds.
    """
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    def fake_get(url: str, _token: str) -> object:
        if "/pulls?state=open" in url:
            return pulls if "&page=1" in url else []
        match = re.search(r"/pulls/(\d+)/files", url)
        assert match is not None, f"unexpected url {url}"
        number = int(match.group(1))
        if number in unreadable:
            raise check.urllib.error.URLError("rate limited")
        # Anchored on the separator: an unanchored `page=` also matches inside
        # `per_page=100`, which silently resolves every read to page 100 and
        # returns an empty slice for every pull request.
        page_match = re.search(r"[?&]page=(\d+)", url)
        assert page_match is not None, f"no page in {url}"
        page = int(page_match.group(1))
        rows = files.get(number, [])
        # 100 rows per page, so a fixture can express a list longer than one page.
        return rows[(page - 1) * 100 : page * 100]

    monkeypatch.setattr(check, "_get", fake_get)


def test_the_sweep_reports_the_measured_pairs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The four measured pairs are reported and the disjoint branch is not.

    This is the sweep's reason to exist: #2233 and #2235 both edit the isaac
    motion-primitive suite and compose to a red one, and no signal on either
    branch says so.
    """
    isaac = "tests/simulation/isaac/test_motion_primitives.py"
    safe = "strands_robots/simulation/safe_output.py"
    _sweep_fixture(
        monkeypatch,
        [_pull(2224), _pull(2227), _pull(2233), _pull(2235), _pull(2999)],
        {
            2224: _files(safe, "strands_robots/simulation/mujoco/physics.py"),
            2227: _files(safe, "tests/simulation/test_output_path_sandbox.py"),
            2233: _files(isaac),
            2235: _files(isaac, "strands_robots/simulation/isaac/motion_primitives.py"),
            2999: _files("docs/unrelated_module.py"),
        },
    )

    assert check.main(["--all-open", "--repo-slug", "o/r", "--token", "t"]) == 1
    report = capsys.readouterr().out
    assert "#2224+#2227, #2233+#2235" in report
    assert f"| #2233 + #2235 | `{isaac}` |" in report
    assert f"| #2224 + #2227 | `{safe}` |" in report
    assert "Compared 10 pair(s) across 5 open non-draft pull request(s)" in report
    assert "#2999" not in report.split("| pair |")[1]


def test_a_prose_only_pair_is_reported_but_not_counted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The measured prose case: three pull requests sharing one README.

    The exemption is what takes the live report from 7 pairs to 4, so it earns a
    pin: prose cannot change what the suite does, and a real collision inside it
    surfaces as a merge conflict the merge gate already stops.
    """
    _sweep_fixture(
        monkeypatch,
        [_pull(2188), _pull(2191)],
        {2188: _files("examples/fleet/README.md"), 2191: _files("examples/fleet/README.md")},
    )

    assert check.main(["--all-open", "--repo-slug", "o/r", "--token", "t"]) == 0
    report = capsys.readouterr().out
    assert "No two open pull requests edit the same behaviour-bearing file." in report
    assert "1 pair(s) sharing only documentation" in report
    assert "- #2188 + #2191: `examples/fleet/README.md`" in report


def test_a_truncated_file_list_is_named_not_reported_as_clean(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pull request read only in part is excluded and named.

    This is the failure mode worth guarding: the shared path may be on a page
    nobody read, so comparing a partial set would report the reassuring answer.
    The pull request is named instead, and the pair it would have formed is not
    silently absent.
    """
    monkeypatch.setattr(check, "MAX_PAGES", 1)
    shared = "strands_robots/simulation/base.py"
    long_list = _files(*[f"pkg/mod_{index}.py" for index in range(100)])
    _sweep_fixture(
        monkeypatch,
        [_pull(10), _pull(11)],
        {10: long_list + _files(shared), 11: _files(shared)},
    )

    assert check.main(["--all-open", "--repo-slug", "o/r", "--token", "t"]) == 0
    report = capsys.readouterr().out
    assert "Not evaluated (1): #10." in report
    assert "Compared 0 pair(s) across 1 open non-draft pull request(s)" in report
    assert "let a partial sweep read as a clean one" in report


def test_one_unreadable_pull_request_does_not_suppress_the_others(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rate limit on one branch must not take the report with it."""
    shared = "strands_robots/mesh/__init__.py"
    _sweep_fixture(
        monkeypatch,
        [_pull(1035), _pull(1722), _pull(9999)],
        {1035: _files(shared), 1722: _files(shared), 9999: _files(shared)},
        unreadable=(9999,),
    )

    assert check.main(["--all-open", "--repo-slug", "o/r", "--token", "t"]) == 1
    report = capsys.readouterr().out
    assert f"| #1035 + #1722 | `{shared}` |" in report
    assert "Not evaluated (1): #9999." in report


def test_a_draft_pull_request_is_not_swept(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A draft cannot be the branch that lands first and hands over an overlap."""
    shared = "strands_robots/simulation/base.py"
    _sweep_fixture(
        monkeypatch,
        [_pull(1), _pull(2, draft=True)],
        {1: _files(shared), 2: _files(shared)},
    )

    assert check.main(["--all-open", "--repo-slug", "o/r", "--token", "t"]) == 0
    assert "Compared 0 pair(s) across 1 open" in capsys.readouterr().out


def test_a_renamed_path_contributes_both_names(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Measured on #2057: the new name alone misses the overlap.

    That pull request moved ``tests/simulation/test_args_docstring_completeness.py``
    to ``tests/``. The API reports the rename as one entry naming only the new
    path, so reading ``filename`` alone gives 6 paths and drops the old name,
    while adding ``previous_filename`` gives 7 and matches
    ``git diff --no-renames`` exactly. Taking the new name only would lose the
    overlap against a branch still editing the old path -- the missed-overlap
    direction ``changed_paths`` already refuses.
    """
    old = "tests/simulation/test_args_docstring_completeness.py"
    new = "tests/test_args_docstring_completeness.py"
    _sweep_fixture(
        monkeypatch,
        [_pull(2057), _pull(2100)],
        {
            2057: [{"filename": new, "previous_filename": old, "status": "renamed"}],
            2100: _files(old),
        },
    )

    assert check.main(["--all-open", "--repo-slug", "o/r", "--token", "t"]) == 1
    assert f"| #2057 + #2100 | `{old}` |" in capsys.readouterr().out


def test_the_paths_of_a_renamed_entry_include_the_previous_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader itself returns both names, and reports the read as complete."""
    monkeypatch.setattr(
        check,
        "_get",
        lambda _url, _token: [{"filename": "b.py", "previous_filename": "a.py", "status": "renamed"}],
    )
    paths, complete = check.pull_request_paths("o/r", 1, "t")
    assert paths == frozenset({"a.py", "b.py"})
    assert complete is True


def test_pairs_are_ordered_and_disjoint_branches_are_absent() -> None:
    """Pairs come out sorted by number so two reports diff on verdicts."""
    overlaps = check.pairwise_overlaps({3: ["x.py"], 1: ["x.py"], 2: ["y.py"]})
    assert [(pair.lower, pair.higher) for pair in overlaps] == [(1, 3)]


def test_a_pair_splits_its_shared_paths_into_behaviour_and_prose() -> None:
    """One pair sharing both kinds reports both, and counts as a finding."""
    both = ["strands_robots/x.py", "docs/y.md"]
    (pair,) = check.pairwise_overlaps({1: both, 2: both})
    assert pair.blocking == ("strands_robots/x.py",)
    assert pair.prose == ("docs/y.md",)
    assert pair.is_finding is True


def test_a_pair_sharing_only_prose_is_not_a_finding() -> None:
    (pair,) = check.pairwise_overlaps({1: ["a.md"], 2: ["a.md"]})
    assert pair.blocking == ()
    assert pair.is_finding is False


def test_the_sweep_uses_the_same_prose_rule_as_the_single_branch_mode() -> None:
    """The exemption has one owner, so the two modes cannot drift apart.

    ``pairwise_overlaps`` partitions through ``partition_overlap`` rather than
    re-deriving the suffix set; if it grew its own copy, a suffix added to
    ``PROSE_SUFFIXES`` would be honoured in one mode and not the other.
    """
    source = inspect.getsource(check.pairwise_overlaps)
    assert "partition_overlap(" in source
    assert "PROSE_SUFFIXES" not in source
    assert "is_prose" not in source


def test_all_open_and_head_are_mutually_exclusive(capsys: pytest.CaptureFixture[str]) -> None:
    """Both name what to evaluate, so a conflicting pair is an error."""
    with pytest.raises(SystemExit) as excinfo:
        check.main(["--all-open", "--head", "deadbeef"])
    assert excinfo.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_a_sweep_without_a_repository_slug_fails_loudly(capsys: pytest.CaptureFixture[str]) -> None:
    """A sweep that cannot name its repository must not report an empty one."""
    assert check.main(["--all-open", "--repo-slug", "", "--token", "t"]) == 1
    assert "needs --repo-slug" in capsys.readouterr().err


def test_listing_the_pull_requests_failing_is_not_a_clean_sweep(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one read with no partial result to report fails rather than reassures."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    def boom(_url: str, _token: str) -> object:
        raise check.urllib.error.URLError("no network")

    monkeypatch.setattr(check, "_get", boom)
    assert check.main(["--all-open", "--repo-slug", "o/r", "--token", "t"]) == 1
    assert "could not list open pull requests" in capsys.readouterr().err


def test_the_single_branch_mode_still_reads_git_only(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No API read may leak into the mode that answers from the object database.

    The sweep is the only caller of ``_get``; making it fatal here keeps the
    single-branch check runnable with no token and no network.
    """

    def fatal(_url: str, _token: str) -> object:
        raise AssertionError("the single-branch mode must not reach the API")

    monkeypatch.setattr(check, "_get", fatal)
    _branch_editing_shared(repo)
    _land_on_main_editing_shared(repo)
    assert _run(repo) == 1
