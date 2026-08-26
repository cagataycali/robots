"""``ReachyMiniDriver``'s ``prefix`` is a sequence of mesh identifiers.

The prefix is the Wireless variant's whole namespace. It is interpolated
verbatim into the three key expressions the robot lives on -- ``joint_positions``
and ``imu_data`` in :meth:`ZenohLink.start`, and ``command`` in
:meth:`ZenohLink.send_cmd` -- and nothing downstream narrows it. Two unusable
shapes reach the wire from there, failing in opposite directions:

* Zenoh reads ``*`` and ``**`` as key-expression wildcards and accepts them on
  a **publisher** key as readily as on a subscriber one. So a wildcard prefix
  widens ``<prefix>/command`` from one robot's inbox into a match-any key, and
  a single ``look()`` is delivered to every Mini beneath the pattern. Nothing
  reports it: the widened call succeeds.
* The shapes Zenoh refuses outright (an empty segment, a stray ``?``) were not
  refused here either. They surfaced from inside the transport at
  :meth:`ZenohLink.start`, after ``connect()`` had already probed the daemon
  and logged a connection -- not from the constructor call that named them.

:func:`~strands_robots.mesh.security.validate_mesh_identifier` is the shared
domain for one key-expression segment, and its own docstring gives the reason
this surface needs it: an unvalidated segment "silently widens a point-to-point
subscription into a match-any one". A prefix is a ``/``-joined sequence of those
segments, so each one goes through it. The ``/`` itself stays legitimate --
``TestMultiSegmentNamespacingIsPreserved`` pins that namespacing two Minis apart
as ``reachy_mini/robot_a`` and ``reachy_mini/robot_b`` is unaffected, which is
what keeps this a refusal of the wildcard rather than of multi-instance use.

``TestWhyTheConstructorOwnsTheDomain`` and
``TestMultiSegmentNamespacingIsPreserved`` pass on both trees: the first pins
the premises that make the constructor the right owner, the second pins what
must not change.
"""

from __future__ import annotations

import ast
import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

import strands_robots
from strands_robots.mesh.security import MAX_PEER_ID_LEN
from tests.test_reachy_mini_driver import _force_real_device_connect_edge

# Prefixes that cannot address a single robot's key expressions. The four
# wildcards are the dangerous half -- Zenoh accepts every one of them, so they
# are not refused anywhere downstream. ``True`` is listed because ``bool`` is
# not a ``str``, and ``None`` because it would otherwise be interpolated as the
# literal segment ``"None"``.
WILDCARD_PREFIXES: list[str] = ["*", "**", "reachy_mini/*", "a$*"]

UNUSABLE_PREFIXES: list[Any] = [
    *WILDCARD_PREFIXES,
    "",
    "a//b",
    "trailing/",
    "/leading",
    "a?b",
    "a$b",
    "a#b",
    "reachy mini",
    "reachy\nmini",
    "x" * (MAX_PEER_ID_LEN + 1),
    "a/" + "x" * (MAX_PEER_ID_LEN + 1),
    7,
    None,
    True,
    b"reachy_mini",
]

# Prefixes that address exactly one robot. ``dev/reachy`` is the multi-segment
# prefix the sibling ``ZenohLink`` suite already drives.
USABLE_PREFIXES: list[str] = [
    "reachy_mini",
    "dev/reachy",
    "reachy_mini/robot_a",
    "a.b-c_d",
    "x" * MAX_PEER_ID_LEN,
]


def _label(value: Any) -> str:
    """A single-token parametrize id for a prefix that may hold spaces."""
    text = repr(value)
    if len(text) > 24:
        text = f"{text[:12]}..len{len(value) if isinstance(value, str) else 0}"
    return text.replace(" ", "-")


@pytest.fixture
def rmd():
    """The reachy_mini_driver module bound to the real device_connect_edge."""
    _force_real_device_connect_edge()
    import strands_robots.device_connect.reachy_mini_driver as module

    return module


class _RecordingTransport:
    """Device Connect transport stand-in that records the keys it is handed.

    The real Zenoh transport hands each key straight to ``zenoh``, so recording
    them is what lets a unit test read which key expression a prefix produced
    without a live session.
    """

    def __init__(self) -> None:
        self.subscribed: list[str] = []
        self.published: list[str] = []

    async def subscribe(self, key: str, _handler: Any) -> None:
        self.subscribed.append(key)

    async def publish(self, key: str, _payload: bytes) -> None:
        self.published.append(key)


def _keys_for(rmd, prefix: str) -> tuple[list[str], list[str]]:
    """The keys a driver built with *prefix* subscribes to and publishes on."""
    driver = rmd.ReachyMiniDriver(host="reachy.local", prefix=prefix)
    transport = _RecordingTransport()
    link = rmd.ZenohLink(transport, driver._prefix)
    asyncio.run(link.start(on_joints=lambda _d: None, on_imu=lambda _d: None))
    asyncio.run(link.send_cmd({"head_pose": [[1.0]]}))
    return transport.subscribed, transport.published


class TestKeyPrefixDomain:
    """A prefix must be a ``/``-joined sequence of mesh identifiers."""

    @pytest.mark.parametrize("prefix", UNUSABLE_PREFIXES, ids=_label)
    def test_an_unusable_prefix_is_refused_at_construction(self, rmd, prefix):
        with pytest.raises(ValueError):
            rmd.ReachyMiniDriver(host="reachy.local", prefix=prefix)

    @pytest.mark.parametrize("prefix", WILDCARD_PREFIXES, ids=_label)
    def test_a_wildcard_prefix_is_refused(self, rmd, prefix):
        """The half nothing downstream refuses: Zenoh accepts every one."""
        with pytest.raises(ValueError, match="prefix segment"):
            rmd.ReachyMiniDriver(host="reachy.local", prefix=prefix)

    @pytest.mark.parametrize("prefix", USABLE_PREFIXES, ids=_label)
    def test_a_usable_prefix_is_accepted_and_stored(self, rmd, prefix):
        assert rmd.ReachyMiniDriver(host="reachy.local", prefix=prefix)._prefix == prefix

    def test_the_default_prefix_is_usable(self, rmd):
        assert rmd.ReachyMiniDriver(host="reachy.local")._prefix == "reachy_mini"

    def test_the_refusal_names_the_class_the_parameter_and_the_segment(self, rmd):
        """A multi-segment prefix must say which segment was unusable."""
        with pytest.raises(ValueError) as caught:
            rmd.ReachyMiniDriver(host="reachy.local", prefix="reachy_mini/*")
        message = str(caught.value)
        assert "ReachyMiniDriver" in message
        assert "prefix" in message
        assert "'*'" in message

    def test_a_non_string_is_refused_rather_than_stringified(self, rmd):
        """``None`` would otherwise namespace the robot under ``"None"``."""
        with pytest.raises(ValueError, match="must be a string"):
            rmd.ReachyMiniDriver(host="reachy.local", prefix=None)

    def test_a_boolean_is_refused_rather_than_read_as_a_segment(self, rmd):
        with pytest.raises(ValueError, match="must be a string"):
            rmd.ReachyMiniDriver(host="reachy.local", prefix=True)


class TestWhyTheConstructorOwnsTheDomain:
    """The premises that make the constructor, not the transport, the owner."""

    def test_the_command_key_interpolates_the_prefix_verbatim(self, rmd):
        _subscribed, published = _keys_for(rmd, "reachy_mini")
        assert published == ["reachy_mini/command"]

    def test_the_sensor_keys_interpolate_the_prefix_verbatim(self, rmd):
        subscribed, _published = _keys_for(rmd, "dev/reachy")
        assert subscribed == ["dev/reachy/joint_positions", "dev/reachy/imu_data"]

    @pytest.mark.parametrize("key", ["**/command", "*/command", "reachy_mini/*/command"])
    def test_zenoh_accepts_a_wildcard_as_a_publisher_key(self, key):
        """So a wildcard prefix is not refused downstream -- it just widens."""
        zenoh = pytest.importorskip("zenoh")
        assert str(zenoh.KeyExpr(key)) == key

    def test_a_wildcard_publish_reaches_a_concrete_subscriber(self):
        """A command published on a widened key lands in another Mini's inbox."""
        zenoh = pytest.importorskip("zenoh")
        received: list[str] = []
        session = zenoh.open(zenoh.Config())
        try:
            session.declare_subscriber(
                "reachy_mini/robot_b/command",
                lambda sample: received.append(str(sample.key_expr)),
            )
            time.sleep(0.4)
            session.put("reachy_mini/**/command", b'{"head_pose": [[1.0]]}')
            deadline = time.monotonic() + 3.0
            while not received and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            try:
                session.close()
            except Exception:  # pragma: no cover - teardown can time out
                pass
        assert received == ["reachy_mini/**/command"]

    def test_an_empty_segment_is_refused_only_inside_the_transport(self):
        """The other half: Zenoh does refuse these, but far from the caller."""
        zenoh = pytest.importorskip("zenoh")
        with pytest.raises(Exception, match="Invalid Key Expr"):
            zenoh.KeyExpr("/joint_positions")


class TestTheGuardIsSoundAgainstZenoh:
    """The accepted domain is a strict subset of what Zenoh will carry."""

    @staticmethod
    def _zenoh_accepts(zenoh, prefix: Any) -> bool:
        try:
            zenoh.KeyExpr(f"{prefix}/command")
        except Exception:
            return False
        return True

    def test_nothing_the_guard_accepts_is_refused_by_zenoh(self, rmd):
        """Soundness: no accepted prefix can reach the late-refusal path."""
        zenoh = pytest.importorskip("zenoh")
        accepted = [p for p in USABLE_PREFIXES if rmd._key_prefix_error(p, "prefix", "C") is None]
        assert accepted == USABLE_PREFIXES
        assert [p for p in accepted if not self._zenoh_accepts(zenoh, p)] == []

    def test_the_extra_refusals_are_wildcards_or_unprintable(self, rmd):
        """What the guard refuses beyond Zenoh is the hazard, not arbitrary."""
        zenoh = pytest.importorskip("zenoh")
        extra = [
            p
            for p in UNUSABLE_PREFIXES
            if self._zenoh_accepts(zenoh, p) and rmd._key_prefix_error(p, "prefix", "C") is not None
        ]
        assert set(WILDCARD_PREFIXES) <= set(extra)
        for prefix in extra:
            wild = isinstance(prefix, str) and ("*" in prefix)
            unprintable = isinstance(prefix, str) and (not prefix.isprintable() or " " in prefix)
            too_long = isinstance(prefix, str) and any(len(s) > MAX_PEER_ID_LEN for s in prefix.split("/"))
            assert wild or unprintable or too_long or not isinstance(prefix, str), prefix

    def test_the_wildcards_really_are_valid_zenoh_keys(self):
        """Non-vacuity: if Zenoh refused them the class would be trivial."""
        zenoh = pytest.importorskip("zenoh")
        assert [p for p in WILDCARD_PREFIXES if not self._zenoh_accepts(zenoh, p)] == []


class TestTheRefusalPrecedesAnyState:
    """A refused prefix leaves nothing built and reaches no daemon."""

    def test_a_refused_prefix_allocates_no_base_driver_state(self, rmd, monkeypatch):
        """The guard runs before ``DeviceDriver.__init__``."""
        calls: list[int] = []
        original = rmd.DeviceDriver.__init__

        def recording_init(self, *args: Any, **kwargs: Any) -> None:
            calls.append(1)
            original(self, *args, **kwargs)

        monkeypatch.setattr(rmd.DeviceDriver, "__init__", recording_init)

        with pytest.raises(ValueError):
            rmd.ReachyMiniDriver(host="reachy.local", prefix="**")
        assert calls == []

        rmd.ReachyMiniDriver(host="reachy.local", prefix="reachy_mini")
        assert calls == [1]

    def test_a_refused_prefix_never_reaches_the_daemon(self, rmd, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(rmd, "api", lambda *a, **k: calls.append("api") or {})
        with pytest.raises(ValueError):
            rmd.ReachyMiniDriver(host="reachy.local", prefix="**")
        assert calls == []

    def test_an_unusable_port_is_still_reported_when_both_are_unusable(self, rmd):
        """The port guard stays first, so its message is not displaced."""
        with pytest.raises(ValueError, match="api_port"):
            rmd.ReachyMiniDriver(host="reachy.local", prefix="**", api_port=0)


class TestTheDomainIsTheSharedMeshIdentifier:
    """One rule, held in one place, applied per segment."""

    @staticmethod
    def _source(rmd) -> str:
        return Path(rmd.__file__).read_text(encoding="utf-8")

    @staticmethod
    def _function(source: str, name: str) -> ast.FunctionDef:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"no function named {name!r}")

    def test_the_helper_delegates_to_the_shared_mesh_identifier(self, rmd):
        helper = self._function(self._source(rmd), "_key_prefix_error")
        called = {
            node.func.id for node in ast.walk(helper) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "validate_mesh_identifier" in called

    def test_the_helper_splits_the_prefix_into_segments(self, rmd):
        helper = self._function(self._source(rmd), "_key_prefix_error")
        assert '.split("/")' in ast.unparse(helper).replace("'", '"')

    def test_the_module_keeps_no_second_copy_of_the_identifier_charset(self, rmd):
        """A local re-spelling of the charset is what makes two rules drift."""
        assert "A-Za-z0-9_.\\-" not in self._source(rmd)

    def test_the_constructor_calls_the_helper(self, rmd):
        init = self._function(self._source(rmd), "__init__")
        called = {
            node.func.id for node in ast.walk(init) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert {"tcp_port_error", "_key_prefix_error"} <= called


class TestMultiSegmentNamespacingIsPreserved:
    """What must not change: the ``/`` is how two Minis are kept apart."""

    def test_two_minis_can_be_namespaced_apart(self, rmd):
        _subs_a, published_a = _keys_for(rmd, "reachy_mini/robot_a")
        _subs_b, published_b = _keys_for(rmd, "reachy_mini/robot_b")
        assert published_a == ["reachy_mini/robot_a/command"]
        assert published_b == ["reachy_mini/robot_b/command"]
        assert published_a != published_b

    def test_the_sibling_suite_s_multi_segment_prefixes_are_still_accepted(self, rmd):
        for prefix in ("dev/reachy", "dev/r", "p"):
            assert rmd.ReachyMiniDriver(host="reachy.local", prefix=prefix)._prefix == prefix

    def test_a_dot_bearing_segment_is_accepted(self, rmd):
        """``.`` is in the identifier charset, so a versioned name is fine."""
        assert rmd.ReachyMiniDriver(host="reachy.local", prefix="reachy.mini.v2")._prefix == "reachy.mini.v2"


def _exported_names(package_init: Path) -> list[str]:
    """The names listed in ``__all__`` of a package's ``__init__``."""
    for node in ast.parse(package_init.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            continue
        if not isinstance(node.value, ast.List | ast.Tuple):
            continue
        return [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    raise AssertionError(f"no __all__ in {package_init}")


def _exported_prefix_constructors(source: str, exported: list[str]) -> dict[str, list[str]]:
    """Map each exported class to the key-prefix ``__init__`` params it takes.

    Scoped to classes the package exports, because that is the surface a caller
    constructs: the hardware links are built only from an already-validated
    prefix, so requiring them to re-check it would institutionalize a second
    copy of the rule.
    """
    found: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef) or node.name not in exported:
            continue
        for member in node.body:
            if not isinstance(member, ast.FunctionDef) or member.name != "__init__":
                continue
            prefixes = [
                arg.arg
                for arg in member.args.args + member.args.kwonlyargs
                if arg.arg == "prefix" or arg.arg.endswith("_prefix")
            ]
            if prefixes:
                found[node.name] = prefixes
    return found


def _validates_prefix(source: str, class_name: str) -> bool:
    """True when the class's ``__init__`` calls the shared prefix domain."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "__init__":
                    return any(
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "_key_prefix_error"
                        for call in ast.walk(member)
                    )
    return False


class TestNoExportedDeviceConnectKeyPrefixSurfaceDrifts:
    """Every exported driver taking a key prefix routes it through one domain."""

    @staticmethod
    def _package_dir() -> Path:
        return Path(strands_robots.__file__).parent / "device_connect"

    def _surfaces(self) -> dict[str, tuple[Path, list[str]]]:
        package = self._package_dir()
        exported = _exported_names(package / "__init__.py")
        surfaces: dict[str, tuple[Path, list[str]]] = {}
        for module in sorted(package.rglob("*.py")):
            source = module.read_text(encoding="utf-8")
            for class_name, prefixes in _exported_prefix_constructors(source, exported).items():
                surfaces[class_name] = (module, prefixes)
        return surfaces

    def test_the_scan_finds_the_known_prefix_surface(self):
        """Non-vacuity: a scan resolving elsewhere would report nothing."""
        assert {name: p for name, (_, p) in self._surfaces().items()} == {"ReachyMiniDriver": ["prefix"]}

    def test_every_exported_prefix_constructor_validates_it(self):
        """A future exported driver cannot namespace itself unvalidated."""
        adrift = {
            name: prefixes
            for name, (module, prefixes) in self._surfaces().items()
            if not _validates_prefix(module.read_text(encoding="utf-8"), name)
        }
        assert adrift == {}, f"exported constructors taking a key prefix without the shared domain: {adrift}"

    def test_the_scan_detects_a_planted_unguarded_prefix(self):
        """Meta: an empty result must mean clean sources, not a dead scanner."""
        planted = 'class Planted:\n    def __init__(self, prefix: str = "p"):\n        self._p = prefix\n'
        assert _exported_prefix_constructors(planted, ["Planted"]) == {"Planted": ["prefix"]}
        assert not _validates_prefix(planted, "Planted")
