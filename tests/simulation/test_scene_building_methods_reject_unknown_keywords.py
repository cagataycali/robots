"""Every backend's scene-building method refuses a keyword it cannot use.

``tests/simulation/test_add_object_unknown_keyword_across_backends.py`` pins the
rule for ``add_object`` alone: a backend method may declare ``**kwargs`` only if
its body calls :func:`strands_robots.simulation.base.unknown_kwargs_error`,
because the action dispatcher skips its own unknown-key check for ``**kwargs``
methods ("those methods own the check instead"). A discarding sink on any
other scene-building verb has the same failure mode, and it was found on one:
the mjlab backend declared ``**kwargs`` on ``create_world``, ``add_robot`` and
``add_camera`` and read none of it, so ``add_robot("so101", positon=[0.3, 0, 0])``
reported success and spawned at the origin, and ``create_world(terrian="rough")``
reported success on flat ground. Every dataset recorded afterwards was silently
wrong with no trace in any error envelope.

This grader walks the same four verbs on every backend package under
``strands_robots/simulation/``. A method without ``**kwargs`` is fine (Python
raises ``TypeError`` and the dispatcher's own check names the key); a method
with ``**kwargs`` must call the helper. It is GL free and needs no optional
backend: the scan is over source.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap

from strands_robots.simulation import base as sim_base

SCENE_VERBS = ("create_world", "add_robot", "add_object", "add_camera")


def _scan_scene_sinks(root: pathlib.Path) -> tuple[set[tuple[str, str]], list[str]]:
    """Return ``(found, adrift)`` over every backend class method named in :data:`SCENE_VERBS`."""
    found: set[tuple[str, str]] = set()
    adrift: list[str] = []
    for backend in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for path in sorted(backend.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - not expected in-tree
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for member in node.body:
                    if not isinstance(member, ast.FunctionDef) or member.name not in SCENE_VERBS:
                        continue
                    found.add((backend.name, member.name))
                    if member.args.kwarg is None:
                        continue
                    calls = {
                        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
                        for n in ast.walk(member)
                        if isinstance(n, ast.Call)
                    }
                    if "unknown_kwargs_error" not in calls:
                        adrift.append(f"{backend.name}/{path.name}::{node.name}.{member.name}")
    return found, adrift


class TestNoSceneBuildingKeywordSinkDrifts:
    def test_every_backend_scene_verb_rejects_unknown_keywords(self) -> None:
        root = pathlib.Path(inspect.getfile(sim_base)).parent
        found, adrift = _scan_scene_sinks(root)
        backends = {backend for backend, _verb in found}
        assert {"isaac", "mjlab", "mujoco", "newton"} <= backends, f"a backend stopped defining scene verbs: {found}"
        for backend in ("isaac", "mjlab", "mujoco", "newton"):
            missing = [verb for verb in SCENE_VERBS if (backend, verb) not in found]
            assert missing == [], f"{backend} no longer defines {missing}; the scan would silently skip them"
        assert adrift == [], "these drop unknown keywords silently: " + ", ".join(adrift)

    def test_the_scanner_reports_a_planted_sink_on_each_verb(self, tmp_path: pathlib.Path) -> None:
        backend = tmp_path / "planted"
        backend.mkdir()
        body = "\n".join(
            f"    def {verb}(self, name=None, **kwargs):\n        return {{'status': 'success'}}\n"
            for verb in SCENE_VERBS
        )
        (backend / "simulation.py").write_text("class Engine:\n" + body, encoding="utf-8")
        found, adrift = _scan_scene_sinks(tmp_path)
        assert found == {("planted", verb) for verb in SCENE_VERBS}
        assert sorted(adrift) == sorted(f"planted/simulation.py::Engine.{verb}" for verb in SCENE_VERBS)

    def test_the_scanner_accepts_a_guarded_sink_and_a_plain_signature(self, tmp_path: pathlib.Path) -> None:
        backend = tmp_path / "planted"
        backend.mkdir()
        (backend / "simulation.py").write_text(
            textwrap.dedent(
                """
                class Engine:
                    def add_robot(self, name, position=None, **kwargs):
                        if err := unknown_kwargs_error("add_robot", kwargs, ("name", "position")):
                            return err
                        return {"status": "success"}

                    def create_world(self, timestep=None):
                        return {"status": "success"}
                """
            ),
            encoding="utf-8",
        )
        found, adrift = _scan_scene_sinks(tmp_path)
        assert found == {("planted", "add_robot"), ("planted", "create_world")}
        assert adrift == []
