"""Pins the checkout classifier against the pull request that produced it.

The observation is #2678. A run fetched what it believed was the pull request's
branch and got a mirror ref that was one commit behind the commit the API was
already serving::

    git fetch origin pull/2678/head    ->  33f8bcf4      the mirror
    pullRequest { headRefOid }         ->  0b070a05      the tip
    git merge-base --is-ancestor 33f8bcf4 0b070a05  ->  true, a pure lag

On that tree the symbol its review thread named (an "unused global" finding) was
genuinely unused, so deletion was the correct local fix and the suite agreed --
18 passed, ruff clean. At the tip the same finding meant the opposite: the
constant was unused because the behavioural tests spelled its value out at each
call site, and the fix was to wire it up. Deletion would have passed CI while
removing the invariant the symbol carried, which is why the fixtures below are a
*worse answer* case and not a duplicated-work case.

Two properties carry the check, and both are pinned here rather than assumed:

- Ahead is not stale. A run that has committed its own work is at a commit the
  tip does not have, which is the ordinary state between a commit and its push.
  A check that reported it would fire exactly when it is being used correctly,
  so the comparison is ancestry and not equality.
- A tip absent from the local object database is stale, not indeterminate. A
  clone that never fetched a commit cannot contain it, and folding that into a
  pass is the one failure mode that would turn this into a check agreeing with
  everything.

See scripts/check_checkout_is_pr_head.py, issue #2520, and the "PR Workflow"
section of AGENTS.md.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_checkout_is_pr_head.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_checkout_is_pr_head", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()

# The two commits #2678 carried at the moment of the lag, verbatim.
MIRROR = "33f8bcf4749a9146111294b4274be81326b50672"
TIP = "0b070a05c5f0c3ca240f0ef092d3fe5fb0fa7a78"

# A commit the tip does not contain, standing for this clone's own unpushed work.
LOCAL_WORK = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


class TestTheMeasuredPullRequest:
    """#2678, at the lag and after it."""

    def test_a_checkout_the_branch_has_moved_past_is_the_finding(self) -> None:
        verdict = mod.classify(MIRROR, TIP, tip_is_ancestor=False)
        assert verdict.outcome == mod.STALE_CHECKOUT
        assert verdict.is_finding

    def test_the_same_clone_reads_current_once_it_fetches_the_branch(self) -> None:
        """The control. Only the checkout moved; the tip did not."""
        verdict = mod.classify(TIP, TIP, tip_is_ancestor=True)
        assert verdict.outcome == mod.CURRENT
        assert not verdict.is_finding

    def test_the_finding_names_both_commits(self) -> None:
        summary = mod.classify(MIRROR, TIP, tip_is_ancestor=False).summary
        assert MIRROR[:8] in summary
        assert TIP[:8] in summary

    def test_the_finding_says_a_thread_may_mean_something_else_at_the_tip(self) -> None:
        """The expensive part of #2678 was not lost time, it was a wrong answer.

        A reader told only "your checkout is behind" re-fetches and keeps the fix
        they derived. The summary has to say that the *meaning* of the thread can
        differ at the tip, which is what makes re-reading it non-optional.
        """
        summary = mod.classify(MIRROR, TIP, tip_is_ancestor=False).summary
        assert "may mean the opposite" in summary
        assert "re-read the threads" in summary


class TestAheadIsNotStale:
    """The clone's own unpushed work must not read as a finding."""

    def test_a_checkout_containing_the_tip_is_ahead_not_stale(self) -> None:
        verdict = mod.classify(LOCAL_WORK, TIP, tip_is_ancestor=True)
        assert verdict.outcome == mod.AHEAD
        assert not verdict.is_finding

    def test_the_ahead_summary_says_the_difference_is_unpushed_work(self) -> None:
        summary = mod.classify(LOCAL_WORK, TIP, tip_is_ancestor=True).summary
        assert "unpushed work" in summary
        assert "Not a finding" in summary

    def test_ahead_and_stale_are_told_apart_only_by_ancestry(self) -> None:
        """Identical commits, opposite verdicts. Ancestry is the whole decision."""
        ahead = mod.classify(LOCAL_WORK, TIP, tip_is_ancestor=True)
        stale = mod.classify(LOCAL_WORK, TIP, tip_is_ancestor=False)
        assert (ahead.outcome, stale.outcome) == (mod.AHEAD, mod.STALE_CHECKOUT)


class TestAnAbsentTipIsStaleRatherThanUnknown:
    """A clone that never fetched the tip cannot contain it."""

    def test_unknown_ancestry_is_the_finding(self) -> None:
        verdict = mod.classify(MIRROR, TIP, tip_is_ancestor=None)
        assert verdict.outcome == mod.STALE_CHECKOUT
        assert verdict.is_finding

    def test_an_unreadable_tip_is_not_a_finding_and_not_current(self) -> None:
        """A deleted fork is a different problem, but it is not a pass either."""
        verdict = mod.classify(MIRROR, None, tip_is_ancestor=None)
        assert verdict.outcome == mod.UNRESOLVABLE_HEAD
        assert not verdict.is_finding

    def test_an_unreadable_checkout_is_its_own_outcome(self) -> None:
        verdict = mod.classify(None, TIP, tip_is_ancestor=None)
        assert verdict.outcome == mod.UNKNOWN_CHECKOUT
        assert not verdict.is_finding

    def test_two_absent_commits_do_not_read_as_current(self) -> None:
        assert mod.classify(None, None, tip_is_ancestor=None).outcome != mod.CURRENT


class TestTheRemedyIsToFetchTheBranchByName:
    """The remedy has to refuse the mirror, or it recreates the finding."""

    def test_it_refuses_refs_pull_by_name(self) -> None:
        text = "\n".join(mod.WHAT_CLEARS_THIS)
        assert "refs/pull/N/head" in text

    def test_it_names_the_fetch_and_the_assertion_against_the_api(self) -> None:
        text = "\n".join(mod.WHAT_CLEARS_THIS)
        assert "git fetch https://github.com/$HEAD_REPO.git $HEAD_REF" in text
        assert "must equal headRefOid" in text

    def test_it_refuses_rebase_and_amend(self) -> None:
        """Both shapes destroy a commit that may already answer the thread."""
        text = "\n".join(mod.WHAT_CLEARS_THIS)
        assert "rebase" in text
        assert "--amend --force-with-lease" in text

    def test_the_remedy_only_appears_on_a_finding(self) -> None:
        anchor = mod.WHAT_CLEARS_THIS[1]
        stale = mod.Report(1, mod.classify(MIRROR, TIP, False), "o/r", "b", TIP).render("o/r")
        clean = mod.Report(1, mod.classify(TIP, TIP, True), "o/r", "b", TIP).render("o/r")
        assert anchor in stale
        assert anchor not in clean


class TestAncestryIsReadFromTheLocalRepository:
    """The one part that touches git, pinned against a repository built here."""

    def test_a_parent_is_an_ancestor_of_its_child(self, tmp_path: Path) -> None:
        cwd, first, second = _two_commit_repo(tmp_path)
        assert mod.tip_is_ancestor_of_checkout(first, second, cwd) is True

    def test_a_child_is_not_an_ancestor_of_its_parent(self, tmp_path: Path) -> None:
        """The stale direction: the tip has a commit the checkout does not."""
        cwd, first, second = _two_commit_repo(tmp_path)
        assert mod.tip_is_ancestor_of_checkout(second, first, cwd) is False

    def test_a_tip_absent_from_the_object_database_is_none(self, tmp_path: Path) -> None:
        cwd, _first, second = _two_commit_repo(tmp_path)
        assert mod.tip_is_ancestor_of_checkout(LOCAL_WORK, second, cwd) is None

    def test_local_head_reads_the_checkout(self, tmp_path: Path) -> None:
        cwd, _first, second = _two_commit_repo(tmp_path)
        assert mod.local_head(cwd) == second

    def test_local_head_of_a_non_repository_is_none(self, tmp_path: Path) -> None:
        plain = tmp_path / "not-a-clone"
        plain.mkdir()
        assert mod.local_head(str(plain)) is None


class TestTheTipIsReadFromTheHeadRepository:
    """Neither suspect answer may stand in for the tip."""

    def test_branch_tip_queries_the_head_repository_ref(self, monkeypatch) -> None:
        seen: dict[str, object] = {}

        def fake(query: str, variables: dict, token: str) -> dict:
            seen.update(variables)
            seen["query"] = query
            return {"repository": {"ref": {"target": {"oid": TIP}}}}

        monkeypatch.setattr(mod, "_graphql", fake)
        assert mod.branch_tip("someone/robots", "topic", "t") == TIP
        assert seen["owner"] == "someone"
        assert seen["name"] == "robots"
        assert seen["ref"] == "refs/heads/topic"
        assert "pullRequest" not in str(seen["query"])

    def test_an_unreadable_ref_returns_none_rather_than_raising(self, monkeypatch) -> None:
        monkeypatch.setattr(mod, "_graphql", lambda *a, **k: {"repository": {"ref": None}})
        assert mod.branch_tip("someone/robots", "gone", "t") is None

    def test_a_malformed_head_repository_is_unresolvable(self, monkeypatch) -> None:
        monkeypatch.setattr(mod, "_graphql", lambda *a, **k: {})
        assert mod.branch_tip("not-a-repo", "topic", "t") is None


class TestEvaluate:
    """The wiring, including the note about its sibling check."""

    NODE = {
        "number": 2678,
        "headRefName": "fix/velocity-write-names-conflicting-rate-drive",
        "headRefOid": TIP,
        "headRepository": {"nameWithOwner": "cagataycali/robots"},
    }

    def test_the_mirror_checkout_is_a_finding(self, monkeypatch) -> None:
        monkeypatch.setattr(mod, "branch_tip", lambda *a: TIP)
        monkeypatch.setattr(mod, "tip_is_ancestor_of_checkout", lambda *a: False)
        report = mod.evaluate(dict(self.NODE), "t", MIRROR, ".")
        assert report.verdict.outcome == mod.STALE_CHECKOUT
        assert report.pr == 2678

    def test_ancestry_is_not_consulted_when_the_checkout_is_the_tip(self, monkeypatch) -> None:
        """No git call for the ordinary case, and no way for one to change it."""
        monkeypatch.setattr(mod, "branch_tip", lambda *a: TIP)

        def explode(*_a: object) -> bool:
            raise AssertionError("ancestry must not be consulted when the commits are equal")

        monkeypatch.setattr(mod, "tip_is_ancestor_of_checkout", explode)
        assert mod.evaluate(dict(self.NODE), "t", TIP, ".").verdict.outcome == mod.CURRENT

    def test_a_null_head_repository_is_unresolvable_not_a_finding(self, monkeypatch) -> None:
        node = dict(self.NODE, headRepository=None)
        report = mod.evaluate(node, "t", MIRROR, ".")
        assert report.verdict.outcome == mod.UNRESOLVABLE_HEAD
        assert not report.verdict.is_finding

    def test_a_recorded_head_behind_the_tip_is_named_as_the_sibling_defect(self, monkeypatch) -> None:
        """#2508's defect and #2678's are independent, and both can be live at once."""
        monkeypatch.setattr(mod, "branch_tip", lambda *a: TIP)
        monkeypatch.setattr(mod, "tip_is_ancestor_of_checkout", lambda *a: False)
        node = dict(self.NODE, headRefOid=MIRROR)
        rendered = mod.evaluate(node, "t", MIRROR, ".").render("strands-labs/robots")
        assert "check_pr_head_is_current.py" in rendered

    def test_no_sibling_note_when_the_record_matches_the_tip(self, monkeypatch) -> None:
        monkeypatch.setattr(mod, "branch_tip", lambda *a: TIP)
        monkeypatch.setattr(mod, "tip_is_ancestor_of_checkout", lambda *a: False)
        rendered = mod.evaluate(dict(self.NODE), "t", MIRROR, ".").render("strands-labs/robots")
        assert "check_pr_head_is_current.py" not in rendered


class TestExitStatus:
    """A missing input must not fail the caller; a finding must."""

    def test_a_missing_repo_does_not_fail_the_caller(self) -> None:
        assert mod.main(["--repo", "", "--pr", "1", "--token", "t"]) == 0

    def test_a_missing_pr_does_not_fail_the_caller(self) -> None:
        assert mod.main(["--repo", "o/r", "--pr", "", "--token", "t"]) == 0

    def test_a_missing_token_does_not_fail_the_caller(self) -> None:
        assert mod.main(["--repo", "o/r", "--pr", "1", "--token", ""]) == 0

    def test_an_unreachable_api_fails_open(self, monkeypatch) -> None:
        def boom(*_a: object, **_k: object) -> dict:
            raise ValueError("GraphQL errors: nope")

        monkeypatch.setattr(mod, "fetch_pull_request", boom)
        assert mod.main(["--repo", "o/r", "--pr", "1", "--token", "t", "--checkout", MIRROR]) == 0

    def test_a_stale_checkout_exits_one(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(mod, "fetch_pull_request", lambda *a: dict(TestEvaluate.NODE))
        monkeypatch.setattr(mod, "branch_tip", lambda *a: TIP)
        monkeypatch.setattr(mod, "tip_is_ancestor_of_checkout", lambda *a: False)
        assert mod.main(["--repo", "o/r", "--pr", "2678", "--token", "t", "--checkout", MIRROR]) == 1
        assert "::warning" in capsys.readouterr().out

    def test_a_current_checkout_exits_zero_and_warns_nothing(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(mod, "fetch_pull_request", lambda *a: dict(TestEvaluate.NODE))
        monkeypatch.setattr(mod, "branch_tip", lambda *a: TIP)
        assert mod.main(["--repo", "o/r", "--pr", "2678", "--token", "t", "--checkout", TIP]) == 0
        assert "::warning" not in capsys.readouterr().out

    def test_an_ahead_checkout_exits_zero(self, monkeypatch) -> None:
        """The shape a mid-cycle run is in, and it must not stop the run."""
        monkeypatch.setattr(mod, "fetch_pull_request", lambda *a: dict(TestEvaluate.NODE))
        monkeypatch.setattr(mod, "branch_tip", lambda *a: TIP)
        monkeypatch.setattr(mod, "tip_is_ancestor_of_checkout", lambda *a: True)
        assert mod.main(["--repo", "o/r", "--pr", "2678", "--token", "t", "--checkout", LOCAL_WORK]) == 0


def _two_commit_repo(tmp_path: Path) -> tuple[str, str, str]:
    """Build a hermetic two-commit repository. Returns (path, first, second).

    Built here rather than reusing this repository's own history: the #2678
    commits live on a fork branch that has since been deleted, so a test reading
    them from the object database would pass or fail on what the clone happened
    to fetch.
    """
    path = tmp_path / "repo"
    path.mkdir()
    cwd = str(path)

    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init", "--quiet", "--initial-branch", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    git("config", "commit.gpgsign", "false")
    (path / "f").write_text("one\n", encoding="utf-8")
    git("add", "f")
    git("commit", "--quiet", "-m", "first")
    first = git("rev-parse", "HEAD")
    (path / "f").write_text("two\n", encoding="utf-8")
    git("add", "f")
    git("commit", "--quiet", "-m", "second")
    second = git("rev-parse", "HEAD")
    return cwd, first, second
