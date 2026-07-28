"""Regression tests: every ``_verify_resume_schema`` call passes its required kwargs.

``DatasetRecordingMixin._verify_resume_schema`` takes ``fps`` as a KEYWORD-ONLY
argument with no default, and ALL THREE backend call sites omitted it (mujoco, isaac,
newton):

    self._verify_resume_schema(resumed, state_names, camera_keys, camera_dims, action_names)
      -> TypeError: _verify_resume_schema() missing 1 required
                    keyword-only argument: 'fps'

so *every* dataset resume raised, and the append path could not run at all. The check
being skipped is not a cosmetic loss either - its own docstring explains that a
resumed dataset keeps the rate it was created at, so appending at a different
requested rate writes a wrong timebase: episodes recorded at different cadences
become indistinguishable and a policy trained on them reads the wrong dt (and so the
wrong velocities) for every appended episode.

These assertions are STATIC (ast) on purpose. The resume path cannot be exercised at
runtime in this environment - importing the LeRobot dataset stack dies on
``AttributeError: module 'av' has no attribute 'option'`` (av 17.1.0 removed the
module torchcodec 0.14.0 needs) - and a signature/call-site mismatch is exactly the
kind of defect a static check catches without that dependency.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from strands_robots.simulation import recording as shared_recording


def _repo_root() -> Path:
    return Path(shared_recording.__file__).resolve().parents[2]


def _caller_paths() -> list[Path]:
    """Every source file that CALLS ``_verify_resume_schema``, discovered.

    Deliberately not a hardcoded list. A first version of this test enumerated the
    MuJoCo and Isaac backends and therefore missed a THIRD offender in
    ``newton/recording.py`` - the same missing-``fps`` crash, invisible because the
    test only looked where the bug had already been found. Discovering call sites
    means a new backend (or a new caller in an existing one) is covered the moment
    it lands.
    """
    package = _repo_root() / "strands_robots"
    out = []
    for path in sorted(package.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        if "_verify_resume_schema" not in source:
            continue
        if _call_nodes(path):
            out.append(path)
    return out


def _required_kwonly_args() -> list[str]:
    """Keyword-only parameters of ``_verify_resume_schema`` that have no default."""
    signature = inspect.signature(shared_recording.DatasetRecordingMixin._verify_resume_schema)
    return [
        name
        for name, param in signature.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is inspect.Parameter.empty
    ]


def _call_nodes(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "_verify_resume_schema"
    ]


def test_fps_is_still_a_required_keyword_only_argument() -> None:
    """Pin the premise; if it gains a default the call-site checks are moot."""
    assert "fps" in _required_kwonly_args()


def test_at_least_the_known_backends_are_discovered() -> None:
    """Guard the discovery itself: a broken walk would vacuously pass everything."""
    found = {path.name for path in _caller_paths()}
    parents = {path.parent.name for path in _caller_paths()}
    assert found == {"recording.py"}, found
    for backend in ("mujoco", "isaac", "newton"):
        assert backend in parents, f"no _verify_resume_schema caller found under {backend}/"


def test_every_call_site_passes_the_required_kwargs() -> None:
    """The core defect: the callers omitted the required ``fps``.

    All THREE backends did - mujoco, isaac and newton - so this iterates discovered
    call sites rather than a list, and reports every offender at once.
    """
    required = _required_kwonly_args()
    offenders = []
    for path in _caller_paths():
        for call in _call_nodes(path):
            passed = {kw.arg for kw in call.keywords if kw.arg is not None}
            double_starred = any(kw.arg is None for kw in call.keywords)
            for name in required:
                if name not in passed and not double_starred:
                    rel = path.relative_to(_repo_root())
                    offenders.append(f"{rel}:{call.lineno} omits {name!r}")
    assert not offenders, "these raise TypeError on every dataset resume: " + "; ".join(offenders)


def test_calling_without_fps_still_raises() -> None:
    """Documents WHY the call sites must pass it - no default is supplied."""

    class _Stub(shared_recording.DatasetRecordingMixin):
        pass

    with pytest.raises(TypeError, match="fps"):
        _Stub()._verify_resume_schema(object(), ["j1"], [], {}, None)  # type: ignore[call-arg]


def test_the_positional_arity_matches_too() -> None:
    """A caller must supply every positional parameter the signature requires."""
    signature = inspect.signature(shared_recording.DatasetRecordingMixin._verify_resume_schema)
    positional = [
        name
        for name, param in signature.parameters.items()
        if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and param.default is inspect.Parameter.empty
        and name != "self"
    ]
    for path in _caller_paths():
        for call in _call_nodes(path):
            supplied = len(call.args) + len({kw.arg for kw in call.keywords if kw.arg is not None})
            assert supplied >= len(positional), (
                f"{path.relative_to(_repo_root())}:{call.lineno} passes {supplied} arguments; "
                f"{len(positional)} positional parameters are required"
            )
