"""No recording test resolves its dataset root from the developer's shared cache.

A `repo_id` without a `root` is not a neutral shorthand in a unit test. Both
`DatasetRecorder.create` and every backend's `start_recording` resolve the pair
through :func:`~strands_robots.dataset_recorder.resolve_dataset_dir`, which
falls back to ``$HF_LEROBOT_HOME/{repo_id}`` -- by default
``~/.cache/huggingface/lerobot/{repo_id}`` -- and `_prepare_create_target` then
**resolves and inspects that shared path** before any injected fake dataset
class is reached. So the test writes nothing there, and still reads it.

Measured on ``9e0b77b9``, before the call sites this module guards were given a
root. Instrumenting ``_lerobot_home`` across the whole unit suite recorded
**65 test instances in 4 modules** resolving the shared home, 58 of them onto
the single id ``local/probe``. Planting one unrelated dataset there::

    mkdir -p ~/.cache/huggingface/lerobot/local/probe/meta
    echo '{"fps":30}' > ~/.cache/huggingface/lerobot/local/probe/meta/info.json

turned ``tests/test_dataset_recorder_fps_domain.py`` plus
``tests/test_dataset_schema_frame_shape_domain.py`` from 133 passed into
**22 failed, 111 passed**, every failure the same ``FileExistsError`` naming a
path in ``$HOME``; removing it returned 133 passed. ``local/probe`` is a name
any scratch script reaches for, so the exposure is latent rather than
theoretical -- and it is hard to attribute, because the failure implicates the
dataset stack and a directory in ``$HOME``, not the test's own resolution.

A `repo_id` is supplied positionally as often as by keyword, and the two forms
are the same exposure. `DatasetRecorder.create("user/data", joint_names=["j1"])`
in ``tests/test_dataset_recorder.py`` resolves ``$HF_LEROBOT_HOME/user/data``
exactly as the keyword form resolves ``local/probe``: measured on the same
commit, one unrelated dataset planted at ``user/data`` took that module from
4 failed to **10 failed** -- six tests, each on the same ``FileExistsError``.
So the rule reads the first positional argument too; a keyword-only rule passes
a file with six live offenders in it.

Why the rule is keyed on the *call site* and not on the resolution:

- Rebinding the dataset home for the whole suite (an autouse fixture pointing
  ``_lerobot_home`` at ``tmp_path``) would close the class in one line, and it
  would also break the one test that legitimately depends on the real default:
  ``test_resolve_dataset_dir_falls_back_to_default_home_when_lerobot_absent``
  asserts the resolved path equals ``Path.home() / ".cache" / ...``. That test
  resolves a path and never touches the disk, which is precisely the
  distinction a home-level guard cannot draw and a call-site rule gets for
  free: it is not a recording call, so this module never looks at it.
- Some guarded call sites are refused *before* the root is resolved (the
  missing-lerobot-extra and no-world guards), so they are inert today.
  Requiring a root of them anyway keeps this rule one line with no exemptions;
  the alternative has to model which guard fires first, which is a fact about
  the implementation and not about the test.

With the roots below supplied, the same instrumentation records exactly **one**
test resolving the shared home -- the fallback test named above, which compares
a path and reads nothing -- and the full unit suite's failure set is unchanged
with a stray dataset planted at all six ids the offenders used.

``DatasetRecorder.resume`` is deliberately not an entry point here: unlike
``create`` it forwards ``repo_id`` / ``root`` to ``LeRobotDataset`` without
resolving the pair itself and without ``_prepare_create_target``, so it neither
resolves the shared home in this repo nor inspects the target first. The
instrumentation agrees - no ``resume`` test reached the home.

``tests_integ/`` deliberately records real datasets and may want the shared
home, so the scan is scoped to ``tests/``.

Naming a root is necessary and not sufficient. The rule above reads the keyword
and not the value it carries, so ``root="/tmp/ds"`` satisfies it while
reproducing verbatim the failure this module exists to prevent: a fixed absolute
path is shared between every test that names it and with every other process on
the machine, so the verdict still depends on what the host already holds. That
is not hypothetical - ``tests/test_dataset_recorder.py`` named ``/tmp/ds`` and
the test failed on any host where that directory existed, with the same
``FileExistsError`` from the same ``_prepare_dataset_target`` inspection, for a
test whose subject is codec normalisation and which writes nothing. Pointing the
same call at a fresh directory turned it green with no other change.

So the second rule reads the value, and it refuses exactly one shape: a ``root``
given as a string literal. The boundary is deliberate. A literal cannot be
per-test unique, which is decidable from the syntax alone with no false
positives; whereas ``root=root`` with ``root`` bound from ``tmp_path`` earlier in
the function is the idiomatic form 101 of the 194 recording calls use, and
telling that apart from a shared local would need dataflow this module has no
business doing. ``root=None`` is left alone for a different reason: it means
"resolve the shared home", which is a question about the home rather than about
the call site, and the modules that pass it are the ones whose subject is that
resolution.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parent

# `start_recording` is reimplemented per backend and always reached through an
# instance, so it is matched on the attribute name for any receiver.
_ENTRY_ATTRS = frozenset({"start_recording"})
_ENTRY_QUALIFIED = frozenset({("DatasetRecorder", "create")})


def _is_entry_call(call: ast.Call) -> bool:
    """Is this call a dataset-recording entry point?"""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr in _ENTRY_ATTRS:
        return True
    return isinstance(func.value, ast.Name) and (func.value.id, func.attr) in _ENTRY_QUALIFIED


def _kwarg_forwarders(tree: ast.Module) -> frozenset[str]:
    """Names of module-level helpers that funnel ``**kwargs`` into an entry point.

    Three of the guarded modules route their calls through a local
    ``_create(**kwargs)`` so the type suppression for a deliberately-wrong
    keyword is stated once. The `repo_id` is supplied at the *caller*, so the
    caller is where the `root` has to be required - resolving the funnel by AST
    keeps that automatic instead of hard-coding a helper name.
    """
    forwarders = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and _is_entry_call(call)
                and any(keyword.arg is None for keyword in call.keywords)
            ):
                forwarders.add(node.name)
                break
    return frozenset(forwarders)


def _names_a_repo_id(call: ast.Call, *, direct: bool) -> bool:
    """Does this call supply a `repo_id`?

    `repo_id` is the first parameter of `DatasetRecorder.create` and of every
    backend's `start_recording`, so a *direct* call supplies it either by
    keyword or as the first positional argument, and the positional form is the
    one `tests/test_dataset_recorder.py` uses.

    A call routed through a local helper is read by keyword only. The helper's
    first parameter is whatever that helper declares -- `_record_episode(sim,
    tmp_path / "ds")` and `_start(factory, fps=fps)` both lead a positional
    with something other than a `repo_id` -- so reading position through a
    forwarder would flag 15 calls that name no dataset at all.
    """
    if direct and call.args:
        return True
    return any(kw.arg == "repo_id" for kw in call.keywords)


def _sole_call(source: str) -> ast.Call:
    """The one call expression in ``source``, for the exemplars below."""
    statement = ast.parse(source).body[0]
    assert isinstance(statement, ast.Expr), source
    call = statement.value
    assert isinstance(call, ast.Call), source
    return call


def _literal_root(call: ast.Call) -> str | None:
    """The ``root`` this call names, if it names one as a string literal.

    Returns the literal so the failure message can quote the path a reader has
    to go and find, rather than only the line it is on.
    """
    for keyword in call.keywords:
        if keyword.arg == "root" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
    return None


def _recording_calls(path: Path) -> list[tuple[ast.Call, bool, set[str]]]:
    """Every recording-entry call in ``path``, with how it was reached."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forwarders = _kwarg_forwarders(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        direct = _is_entry_call(node)
        is_forwarded = isinstance(node.func, ast.Name) and node.func.id in forwarders
        if direct or is_forwarded:
            found.append((node, direct, {kw.arg for kw in node.keywords if kw.arg}))
    return found


@functools.cache
def _scan() -> dict[Path, list[tuple[ast.Call, bool, set[str]]]]:
    return {path: _recording_calls(path) for path in sorted(_TESTS_ROOT.rglob("test_*.py"))}


def test_a_recording_call_that_names_a_repo_id_also_names_a_root() -> None:
    """A unit test's dataset directory is its own, not the developer's."""
    offenders = [
        f"{path.relative_to(_TESTS_ROOT)}:{call.lineno}"
        for path, calls in _scan().items()
        for call, direct, keywords in calls
        if _names_a_repo_id(call, direct=direct) and "root" not in keywords
    ]
    assert not offenders, (
        "these recording calls pass a repo_id with no root, so they resolve a "
        "dataset directory under $HF_LEROBOT_HOME (by default "
        "~/.cache/huggingface/lerobot) and their verdict depends on what is "
        "already in the developer's cache - pass root=str(tmp_path / 'dataset'):\n" + "\n".join(offenders)
    )


def test_the_scan_reaches_the_modules_that_record() -> None:
    """Non-vacuity, keyed on modules rather than on a call count.

    A count would have to be edited by whoever adds a recording test, which
    makes it a number kept in step by hand rather than a floor. The modules
    below own the recording entry points this rule exists for: if the matcher
    stops recognising a call shape, they are what stops being scanned.
    """
    scanned = {str(path.relative_to(_TESTS_ROOT)) for path, calls in _scan().items() if calls}
    for module in (
        "test_dataset_recorder_fps_domain.py",
        "test_dataset_schema_frame_shape_domain.py",
        "test_dataset_schema_column_names_distinct.py",
        "simulation/mujoco/test_recording_paths.py",
        "simulation/newton/test_dataset_recording.py",
        "simulation/isaac/test_dataset_recording.py",
        "test_dataset_recorder.py",
    ):
        assert module in scanned, (
            f"{module} contains recording calls the scan no longer sees, so the "
            "rule above passes because it inspects nothing"
        )


def test_the_scan_still_reads_a_positional_repo_id() -> None:
    """The positional half of the rule is pinned separately from the keyword half.

    Six live offenders passed `repo_id` positionally, so a matcher that quietly
    stopped reading position would take the rule back to where it started while
    both tests above still passed.
    """
    positional = [
        call
        for path, calls in _scan().items()
        if path.name == "test_dataset_recorder.py"
        for call, direct, _keywords in calls
        if direct and call.args
    ]
    assert positional, (
        "no direct recording call in test_dataset_recorder.py is seen to pass a "
        "positional repo_id, so the rule now rests on the keyword form alone"
    )


def test_a_named_root_is_not_a_fixed_absolute_path() -> None:
    """A unit test's dataset directory is unique to the test, not just non-default.

    The rule above requires a root; this one requires it to be the test's own. A
    string literal is shared by construction - with every other test naming it and
    with every process on the host - so it carries the same exposure as the shared
    cache the module is named for.
    """
    offenders = [
        f"{path.relative_to(_TESTS_ROOT)}:{call.lineno}: root={literal!r}"
        for path, calls in _scan().items()
        for call, direct, _keywords in calls
        if _names_a_repo_id(call, direct=direct) and (literal := _literal_root(call)) is not None
    ]
    assert not offenders, (
        "these recording calls name a root as a fixed absolute path, so it is "
        "shared with every other test that names it and with every process on "
        "the host, and the verdict depends on what is already there - pass "
        "root=str(tmp_path / 'dataset'):\n" + "\n".join(offenders)
    )


def test_the_literal_root_rule_reads_the_value_and_not_the_keyword() -> None:
    """Non-vacuity, on constructed calls rather than on the corpus.

    With the offenders fixed the corpus can no longer exercise a rejection, so a
    rule graded only against it would pass by having nothing to look at. These
    two exemplars are the shapes the rule has to separate, and the accepted one
    also pins that the rule does not simply refuse every root.
    """
    refused = _sole_call('DatasetRecorder.create("user/data", root="/tmp/ds")')
    accepted = _sole_call('DatasetRecorder.create("user/data", root=str(tmp_path / "ds"))')

    assert _literal_root(refused) == "/tmp/ds"
    assert _literal_root(accepted) is None
    # Both are recording calls, so the rule is separating them on the value.
    assert _is_entry_call(refused) and _is_entry_call(accepted)
