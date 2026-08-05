"""Regression tests: every backend refuses an ``add_object`` keyword it cannot use.

All three backends declare the *same ten* ``add_object`` parameters. Two of them
declare no ``**kwargs``, so Python refuses an unknown keyword with a
``TypeError``. The Isaac backend alone declared a ``**kwargs`` sink, read exactly
one key out of it -- ``scale``, the documented ``size`` alias -- and dropped
every other key silently.

``unknown_kwargs_error`` in :mod:`strands_robots.simulation.base` exists for
precisely that shape, and its docstring names the two kinds a ``**kwargs`` method
can be:

    For a *forwarding* sink (``attach_teleop``, ``stream_dataset``) dropping is
    right - the keys belong to the callee. For a *discarding* sink it turns a
    misspelled or invented parameter into a successful no-op [...] Discarding
    sinks call this helper instead, so an unusable parameter is named rather than
    swallowed.

The action dispatcher states the same delegation from the other side, in
``MuJoCoSimEngine._validate_and_build_kwargs``:

    ``**kwargs`` methods accept arbitrary inputs, so we skip the unknown-key
    check for them. Those methods own the check instead [...] otherwise skipping
    here would make a misspelled parameter a silent no-op on exactly those
    actions.

Isaac's ``add_object`` is a discarding sink that never called the helper, so it
was the one action the dispatcher's skip left uncovered.

Measured on the probe set below, one ``add_object`` per case, with no
``isaacsim`` / ``newton`` / ``warp`` installed -- six of eight keywords diverged:

===========================  ==========  ==========  ====================
keyword                      MuJoCo      Newton      Isaac (before)
===========================  ==========  ==========  ====================
(none -- control)            success     success     success
``scale`` (documented)       TypeError   TypeError   success, honored
``heigth``                   TypeError   TypeError   success, dropped
``positon``                  TypeError   TypeError   success, dropped
``colour``                   TypeError   TypeError   success, dropped
``density``                  TypeError   TypeError   success, dropped
``friction``                 TypeError   TypeError   success, dropped
``rgba``                     TypeError   TypeError   success, dropped
===========================  ==========  ==========  ====================

Every dropped row compiled the *default* extents and reported them back in the
result ``json``, so nothing downstream could tell the request from a call that
never carried the keyword at all.

``scale`` is the reason the sink exists and is a real Isaac-only capability, so
deleting ``**kwargs`` is not the fix: the sink stays and rejects everything it
does not read. ``TestTheScaleAliasIsStillHonored`` pins that half, so a future
"just drop the ``**kwargs``" change fails here.

``TestNoAddObjectKeywordSinkDrifts`` keeps it structural: a backend
``add_object`` may declare ``**kwargs`` only if it calls the shared helper.

These tests are GL-free and need no optional backend: the guard is the method's
first statement, so calling the unbound method with a small stand-in for ``self``
exercises it in every environment.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap
from typing import Any

import pytest

from strands_robots.simulation import base as sim_base

# Imported as a module rather than by symbol so the accepted-keyword tuple is
# reached through it: one import of this module keeps py/import-and-import-from
# quiet, and a test that names a symbol the module does not define reports that
# directly instead of failing the whole file at collection.
from strands_robots.simulation.isaac import simulation as isaac_sim
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from tests.simulation.test_pose_vector_domain_across_backends import _isaac_stub, _newton_stub

#: Keywords no ``add_object`` can use: two misspellings of a real parameter, a
#: British spelling of one, an MJCF attribute name, and two plausible physics
#: options this API does not expose. Each was accepted and discarded.
UNKNOWN_KEYWORDS: tuple[tuple[str, Any], ...] = (
    ("heigth", 0.30),
    ("positon", [1.0, 2.0, 0.3]),
    ("colour", [1.0, 0.0, 0.0]),
    ("density", 500.0),
    ("friction", 0.9),
    ("rgba", [1.0, 0.0, 0.0, 1.0]),
)

#: The one residual keyword Isaac's sink exists to read.
HONORED_EXTRA = "scale"


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


def _json(result: dict[str, Any]) -> dict[str, Any]:
    return next(block["json"] for block in result["content"] if "json" in block)


def _isaac_counting() -> tuple[Any, dict[str, int]]:
    """An Isaac stand-in counting what a refused call must never reach."""
    stub = _isaac_stub()
    calls = {"construct": 0, "scene_add": 0}

    def construct(**kwargs: Any) -> tuple[Any, Any]:
        calls["construct"] += 1
        return object(), list(kwargs.get("size") or [])

    def scene_add(handle: Any) -> None:
        calls["scene_add"] += 1

    stub._construct_shape_prim = construct
    stub._world.scene.add = scene_add
    return stub, calls


def _isaac_add(stub: Any, **kwargs: Any) -> dict[str, Any]:
    """Call the unbound method so a deliberately invalid keyword reaches it."""
    return isaac_sim.IsaacSimulation.add_object(stub, **kwargs)


def _newton_add(stub: Any, **kwargs: Any) -> dict[str, Any]:
    return NewtonSimEngine.add_object(stub, **kwargs)


# --------------------------------------------------------------------------- #
# The Isaac backend: the keyword is named rather than swallowed               #
# --------------------------------------------------------------------------- #


class TestIsaacAddObjectRefusesAKeywordItCannotUse:
    """Each unknown keyword is refused, naming it and the accepted vocabulary."""

    @pytest.mark.parametrize(("key", "value"), UNKNOWN_KEYWORDS, ids=[k for k, _ in UNKNOWN_KEYWORDS])
    def test_an_unknown_keyword_is_refused_by_name(self, key: str, value: Any) -> None:
        stub, _ = _isaac_counting()
        result = _isaac_add(stub, name="crate", shape="box", size=[0.1, 0.1, 0.1], **{key: value})
        assert result["status"] == "error"
        text = _text(result)
        assert key in text, text
        assert "add_object" in text, text
        # The accepted vocabulary is the actionable half: without it the caller
        # only learns the key was wrong, not what to write instead.
        assert "size" in text and "position" in text, text

    def test_several_unknown_keywords_are_all_named(self) -> None:
        """Reporting one at a time would need one round trip per typo."""
        stub, _ = _isaac_counting()
        result = _isaac_add(stub, name="crate", heigth=0.3, positon=[0, 0, 1], density=500.0)
        assert result["status"] == "error"
        text = _text(result)
        for key in ("density", "heigth", "positon"):
            assert key in text, text

    def test_a_usable_call_is_unaffected(self) -> None:
        stub, calls = _isaac_counting()
        result = _isaac_add(stub, name="crate", shape="box", size=[0.1, 0.1, 0.1])
        assert result["status"] == "success"
        assert calls["construct"] == 1
        assert _json(result)["size"] == [0.1, 0.1, 0.1]


class TestTheRefusalPrecedesEveryEffect:
    """A refused keyword must leave the scene, the registry and the name alone."""

    def test_no_prim_is_constructed_and_nothing_is_registered(self) -> None:
        stub, calls = _isaac_counting()
        result = _isaac_add(stub, name="crate", shape="box", size=[0.1, 0.1, 0.1], heigth=0.3)
        assert result["status"] == "error"
        assert calls == {"construct": 0, "scene_add": 0}
        assert stub._objects == {}
        assert stub._prim_registry == []

    def test_the_name_is_still_usable_afterwards(self) -> None:
        """A refused call must not consume the name the caller asked for."""
        stub, calls = _isaac_counting()
        assert _isaac_add(stub, name="crate", size=[0.1, 0.1, 0.1], colour=[1, 0, 0])["status"] == "error"
        retry = _isaac_add(stub, name="crate", size=[0.1, 0.1, 0.1], color=[1.0, 0.0, 0.0])
        assert retry["status"] == "success", _text(retry)
        assert calls["construct"] == 1
        assert "crate" in stub._objects


class TestTheScaleAliasIsStillHonored:
    """``scale`` is why the sink exists; the guard must not cost that capability."""

    def test_scale_supplies_the_extent_when_size_is_omitted(self) -> None:
        stub, calls = _isaac_counting()
        result = _isaac_add(stub, name="crate", shape="box", scale=[0.2, 0.3, 0.4])
        assert result["status"] == "success", _text(result)
        assert _json(result)["size"] == [0.2, 0.3, 0.4]
        assert calls["construct"] == 1

    def test_an_explicit_size_still_wins_over_scale(self) -> None:
        stub, _ = _isaac_counting()
        result = _isaac_add(stub, name="crate", shape="box", size=[0.1, 0.1, 0.1], scale=[0.9, 0.9, 0.9])
        assert result["status"] == "success", _text(result)
        assert _json(result)["size"] == [0.1, 0.1, 0.1]

    def test_scale_beside_an_unknown_keyword_is_still_refused(self) -> None:
        """The honored key must not license the rest of the mapping."""
        stub, calls = _isaac_counting()
        result = _isaac_add(stub, name="crate", scale=[0.2, 0.2, 0.2], heigth=0.3)
        assert result["status"] == "error"
        assert "heigth" in _text(result)
        assert calls["construct"] == 0


# --------------------------------------------------------------------------- #
# Cross-backend parity                                                        #
# --------------------------------------------------------------------------- #


def _refuses(call: Any, **kwargs: Any) -> bool:
    """Did this backend refuse the call, by either documented channel?"""
    try:
        return bool(call(**kwargs)["status"] == "error")
    except TypeError:
        # A backend with no ``**kwargs`` refuses an unknown keyword here.
        return True


class TestEveryBackendRefusesAnUnknownAddObjectKeyword:
    """One API, one verdict: no backend accepts a keyword it cannot honor."""

    @pytest.mark.parametrize(("key", "value"), UNKNOWN_KEYWORDS, ids=[k for k, _ in UNKNOWN_KEYWORDS])
    def test_no_backend_accepts_it(self, key: str, value: Any) -> None:
        isaac, _ = _isaac_counting()
        verdicts = {
            "isaac": _refuses(lambda **kw: _isaac_add(isaac, **kw), name="crate", size=[0.1] * 3, **{key: value}),
            "newton": _refuses(
                lambda **kw: _newton_add(_newton_stub(), **kw), name="crate", size=[0.1] * 3, **{key: value}
            ),
        }
        assert all(verdicts.values()), f"{key} was accepted by: {[b for b, v in verdicts.items() if not v]}"

    def test_a_usable_call_is_accepted_by_every_backend(self) -> None:
        """Non-vacuity: the parity above must not hold by refusing everything."""
        isaac, _ = _isaac_counting()
        assert _isaac_add(isaac, name="crate", size=[0.1] * 3)["status"] == "success"
        assert _newton_add(_newton_stub(), name="crate", size=[0.1] * 3)["status"] == "success"


# --------------------------------------------------------------------------- #
# The accepted set, and no drift                                              #
# --------------------------------------------------------------------------- #


class TestTheAcceptedSetIsDerivedFromTheSignature:
    """The vocabulary must be the live signature plus the keys the sink reads."""

    def test_it_is_the_declared_parameters_plus_scale(self) -> None:
        declared = {
            name
            for name, param in inspect.signature(isaac_sim.IsaacSimulation.add_object).parameters.items()
            if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
        }
        assert set(isaac_sim._ADD_OBJECT_PARAMS) == declared | {HONORED_EXTRA}

    def test_every_declared_parameter_is_accepted_by_keyword(self) -> None:
        """A declared parameter passed by keyword must never read as unknown."""
        for name in isaac_sim._ADD_OBJECT_PARAMS:
            if name == HONORED_EXTRA:
                continue
            assert sim_base.unknown_kwargs_error("add_object", {name: None}, isaac_sim._ADD_OBJECT_PARAMS) is None


def _scan_add_object_sinks(root: pathlib.Path) -> tuple[set[tuple[str, str]], list[str]]:
    """Find every backend ``add_object`` with a ``**kwargs`` sink; flag the unguarded.

    Returns ``(found, adrift)`` where ``found`` is every public ``add_object``
    defined on a class under a backend package and ``adrift`` names those that
    declare ``**kwargs`` without calling the shared helper.
    """
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
                    if not isinstance(member, ast.FunctionDef) or member.name != "add_object":
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
                        adrift.append(f"{backend.name}/{path.name}::{node.name}.add_object")
    return found, adrift


#: Every backend package that defines an ``add_object``. ``SpecBuilder`` in the
#: MuJoCo package takes an already-validated ``SimObject``, so the scan keys on
#: the backend rather than the class and both MuJoCo definitions collapse here.
_KNOWN_ADD_OBJECT_BACKENDS = {("isaac", "add_object"), ("mujoco", "add_object"), ("newton", "add_object")}


class TestNoAddObjectKeywordSinkDrifts:
    """An ``add_object`` may declare ``**kwargs`` only if it rejects the residue."""

    def test_every_backend_sink_rejects_unknown_keywords(self) -> None:
        root = pathlib.Path(inspect.getfile(sim_base)).parent
        found, adrift = _scan_add_object_sinks(root)
        assert found == _KNOWN_ADD_OBJECT_BACKENDS, f"the set of add_object backends changed: {found}"
        assert adrift == [], "these drop unknown keywords silently: " + ", ".join(adrift)

    def test_the_scanner_reports_a_planted_sink(self, tmp_path: pathlib.Path) -> None:
        """Without this, an empty result could mean a scanner matching nothing."""
        backend = tmp_path / "planted"
        backend.mkdir()
        (backend / "simulation.py").write_text(
            textwrap.dedent(
                """
                class Engine:
                    def add_object(self, name, size=None, **kwargs):
                        kwargs.pop("scale", None)
                        return {"status": "success"}
                """
            ),
            encoding="utf-8",
        )
        found, adrift = _scan_add_object_sinks(tmp_path)
        assert found == {("planted", "add_object")}
        assert len(adrift) == 1
        assert "Engine.add_object" in adrift[0]

    def test_the_scanner_accepts_a_guarded_sink(self, tmp_path: pathlib.Path) -> None:
        """The negative control: declaring ``**kwargs`` is not itself the defect."""
        backend = tmp_path / "planted"
        backend.mkdir()
        (backend / "simulation.py").write_text(
            textwrap.dedent(
                """
                class Engine:
                    def add_object(self, name, size=None, **kwargs):
                        if err := unknown_kwargs_error("add_object", kwargs, ("name", "size")):
                            return err
                        return {"status": "success"}
                """
            ),
            encoding="utf-8",
        )
        found, adrift = _scan_add_object_sinks(tmp_path)
        assert found == {("planted", "add_object")}
        assert adrift == []
