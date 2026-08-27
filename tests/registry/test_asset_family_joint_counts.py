"""Robots compiled from the same model must declare the same joint count.

``robots.json`` gives every robot a ``joints`` figure, and two discovery
surfaces report it verbatim: :func:`~strands_robots.registry.get_robot` returns
it and :func:`~strands_robots.registry.list_robots` prints it in the ``Joints``
column an agent reads to size an action vector. Two entries disagreed with a
sibling built from an indistinguishable model:

* ``ur5e`` declared ``8``. Its asset has six hinge joints, six actuators and no
  free joint - and ``ur10e``, whose compiled model has byte-identical joint
  names, joint types and actuator names, declared ``6``.
* ``unitree_a1`` declared ``16`` where ``aliengo`` and ``go1`` - same
  indistinguishable model - both declared ``13``, which is that model's
  ``njnt`` (twelve movable joints and the floating base). ``anymal_b`` and
  ``anymal_c``, a separate family whose model has the same 13/12/12 shape,
  declare ``13`` too.

One compiled shape, two answers, in both cases.

What ``joints`` MEANS across the whole registry is deliberately not settled
here. The figure follows no single rule today: ``docs/robots/arms.md`` says
"Joint counts include any free joints / gripper actuators", which reads as
MuJoCo's ``njnt`` and holds for ``anymal_b``/``anymal_c`` (13 against a 12-DOF
description, the extra one being the floating base), while ``panda`` declares
``7`` against an ``njnt`` of 9 - the arm without its two finger joints - and
``arx_l5``/``piper`` both declare ``11`` against an ``njnt`` of 8. Of the 50
registry robots whose asset loads, 22 declare a figure that is neither their
``njnt`` nor their movable-joint count. Picking one convention would rewrite
those 22 numbers on a guess about what each was counting, so this file grades a
weaker property that needs no such decision:

    two robots whose compiled models are indistinguishable must be described
    by the same number, whatever that number is counting.

That holds under every convention above - ``njnt``, movable joints, actuated
DOF, hardware DOF - because the models agree on all of them. It is the in-family
control that makes both figures decidable without settling the registry-wide
question: ``ur5e``'s own description ("6-DOF industrial") agrees with the
sibling it disagreed with, and ``unitree_a1``'s two siblings agree with each
other, with their shared model and with a second quadruped family of the same
shape.

Two layers, because the oracle is not available everywhere:

* :class:`TestEveryAssetFamilyAgrees` reads ``robots.json`` alone, so it grades
  the property on any install - including one with no MuJoCo and no downloaded
  assets, which is where the registry is most often read.
* :class:`TestTheFrozenFamiliesStillMatchTheAssets` re-derives the grouping from
  the real compiled models and asserts :data:`_ASSET_FAMILIES` still describes
  them. Without it the frozen table could drift into agreeing with a wrong
  registry, which is the failure a hand-maintained copy always has. It skips
  where the assets are absent rather than downloading them, which is what
  ``allow_download=False`` in :func:`_asset_signatures` buys: this walks the
  whole registry, so the downloading resolver fetches every asset the machine
  does not already have.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "strands_robots" / "registry" / "robots.json"

#: Registry robots whose compiled MJCF models are indistinguishable - identical
#: joint names, identical joint types and identical actuator names. Measured
#: against the shipped assets on mujoco 3.5.0 over the 50 registry entries whose
#: model loads; :class:`TestTheFrozenFamiliesStillMatchTheAssets` re-derives this
#: from the assets so the copy cannot go stale.
_ASSET_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("aliengo", "go1", "unitree_a1"),
    ("anymal_b", "anymal_c"),
    ("arx_l5", "piper"),
    ("toddlerbot_2xc", "toddlerbot_2xm"),
    ("ur10e", "ur5e"),
    ("vx300s", "wx250s"),
)

#: Families this file does not require to agree yet, each with the reason.
#:
#: ``vx300s``/``wx250s`` declare ``19`` and ``16`` against an ``njnt`` of 8 and
#: an actuator count of 7, and their descriptions state the same shape as each
#: other ("6-DOF + gripper"). So they are the same defect as ``ur5e`` - but
#: unlike ``ur5e``, whose sibling and description both name ``6``, nothing here
#: says which of the two figures is right, or whether either is. Choosing needs
#: the registry-wide convention decision this file declines to make, so the pair
#: is recorded rather than guessed at. Removing an entry from this set is how
#: that decision gets enforced.
_UNRESOLVED_FAMILIES: frozenset[tuple[str, ...]] = frozenset({("vx300s", "wx250s")})


@pytest.fixture(scope="module")
def registry() -> dict[str, Any]:
    """Load the shipped robot registry once."""
    data = json.loads(REGISTRY_PATH.read_text())
    return dict(data.get("robots", data))


@pytest.fixture
def host_asset_cache(monkeypatch: Any) -> None:
    """Let the re-derivation read the machine's real asset cache.

    ``tests/registry/conftest.py`` repoints ``STRANDS_ASSETS_DIR`` at a per-test
    temp dir, so a developer's user robots cannot leak into registry assertions.
    That isolation is right for the registry reads it was written for, and it
    also empties the asset cache - the one input this class needs. Under it the
    grouping is unconfirmable on every machine, and the resolver's downloading
    default hid that by fetching the whole corpus into the temp dir instead of
    reporting that there was nothing to read.

    Restoring the host value for this class alone gives the two honest outcomes
    the class documents: confirm the table where the assets are present, skip
    where they are not.
    """
    monkeypatch.delenv("STRANDS_ASSETS_DIR", raising=False)


def _graded_families() -> tuple[tuple[str, ...], ...]:
    """The families this file requires to agree."""
    return tuple(f for f in _ASSET_FAMILIES if f not in _UNRESOLVED_FAMILIES)


def _asset_signatures(names: Iterable[str]) -> dict[str, tuple[Any, ...]]:
    """Compile each asset already on disk and read its joint/actuator signature.

    ``allow_download=False`` is load-bearing rather than a speed-up. This walks
    the whole registry, and the resolver downloads any asset it cannot find, so
    the default fetches every entry a machine does not already have - 63 of the
    72 on a fresh checkout. Declining is equivalent to a download that fails, so
    it cannot change the grouping on a machine that has the assets, and leaves it
    empty on one that does not.

    Args:
        names: Registry robot names to look for on disk.

    Returns:
        The signature of every named robot whose asset is present and compiles;
        robots whose asset is absent or unloadable are simply left out.
    """
    mujoco = pytest.importorskip("mujoco")
    from strands_robots.assets.manager import resolve_model_path

    signatures: dict[str, tuple[Any, ...]] = {}
    for name in names:
        try:
            path = resolve_model_path(name, allow_download=False)
        except Exception:  # pragma: no cover - a resolver failure is not this test's subject
            continue
        if path is None or not Path(path).exists():
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
        except Exception:  # pragma: no cover - an unloadable asset is not this test's subject
            continue
        signatures[name] = (
            tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)),
            tuple(int(model.jnt_type[i]) for i in range(model.njnt)),
            tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)),
        )
    return signatures


class TestTheFamilyTableIsUsable:
    """Premises the graded property rests on, so it cannot pass vacuously."""

    def test_every_family_names_at_least_two_robots(self) -> None:
        """A one-robot family agrees with itself and grades nothing."""
        assert _ASSET_FAMILIES
        for family in _ASSET_FAMILIES:
            assert len(family) >= 2, family
            assert len(set(family)) == len(family), family

    def test_every_named_robot_is_in_the_registry(self, registry: dict[str, Any]) -> None:
        """A typo would silently drop a robot out of the graded set."""
        missing = sorted({n for f in _ASSET_FAMILIES for n in f} - set(registry))
        assert not missing, f"families name robots absent from the registry: {missing}"

    def test_something_is_actually_graded(self) -> None:
        """Exempting every family would leave the agreement rule inert."""
        graded = _graded_families()
        assert graded, "every family is exempt, so nothing is graded"
        for family in (("ur10e", "ur5e"), ("aliengo", "go1", "unitree_a1")):
            assert family in graded, f"{family} carried a corrected value and must stay graded"

    def test_every_exemption_names_a_real_family(self) -> None:
        """An exemption for a family that no longer exists hides a regression."""
        stale = sorted(f for f in _UNRESOLVED_FAMILIES if f not in _ASSET_FAMILIES)
        assert not stale, f"exemptions for families that are not declared: {stale}"


class TestEveryAssetFamilyAgrees:
    """The graded property, on the registry alone."""

    @pytest.mark.parametrize("family", _graded_families(), ids=lambda f: "-".join(f))
    def test_one_compiled_shape_is_described_by_one_number(
        self, family: tuple[str, ...], registry: dict[str, Any]
    ) -> None:
        """Robots with indistinguishable models declare the same ``joints``."""
        declared = {name: registry[name].get("joints") for name in family}
        assert len(set(declared.values())) == 1, (
            f"{' and '.join(family)} compile to the same model - identical joint names, "
            f"joint types and actuator names - so they must declare the same joint count, "
            f"but the registry says {declared}"
        )

    @pytest.mark.parametrize(
        ("name", "expected", "why"),
        [
            ("ur5e", 6, "six hinge joints, six actuators, no free joint and no gripper"),
            ("ur10e", 6, "six hinge joints, six actuators, no free joint and no gripper"),
            ("aliengo", 13, "twelve movable joints and the floating base"),
            ("go1", 13, "twelve movable joints and the floating base"),
            ("unitree_a1", 13, "twelve movable joints and the floating base"),
        ],
    )
    def test_the_corrected_entries_declare_what_their_asset_has(
        self, name: str, expected: int, why: str, registry: dict[str, Any]
    ) -> None:
        """The value, not only the agreement.

        Agreement alone is satisfied by moving the majority to the outlier -
        ``ur10e`` up to ``8``, or ``aliengo`` and ``go1`` up to ``16``. These
        pin the figure the shared asset actually has, so the family cannot
        settle on the wrong number.
        """
        assert registry[name].get("joints") == expected, (
            f"{name} declares {registry[name].get('joints')!r}; its asset has {why}"
        )

    def test_the_declared_count_reaches_the_discovery_surface(self) -> None:
        """The figure is reported, so a wrong one is read rather than inert."""
        from strands_robots.registry import get_robot, list_robots

        entry = get_robot("ur5e")
        assert entry is not None
        assert entry["joints"] == 6
        listed = {r["name"]: r["joints"] for r in list_robots()}
        assert listed["ur5e"] == 6
        assert listed["ur10e"] == 6
        assert listed["unitree_a1"] == 13


class TestTheFrozenFamiliesStillMatchTheAssets:
    """Keep :data:`_ASSET_FAMILIES` honest against the compiled models."""

    @pytest.mark.usefixtures("host_asset_cache")
    def test_the_families_are_what_the_assets_say_they_are(self, registry: dict[str, Any]) -> None:
        """Re-derive the grouping; the frozen table must still describe it.

        Restricted to the robots whose assets are present: a family whose
        members are not all loadable here cannot be confirmed or refuted, so it
        is neither.
        """
        signatures = _asset_signatures(registry)

        if len(signatures) < 2:
            pytest.skip("registry assets unavailable, so the grouping cannot be re-derived")

        derived: dict[tuple[Any, ...], list[str]] = {}
        for name, signature in signatures.items():
            derived.setdefault(signature, []).append(name)
        derived_families = {tuple(sorted(names)) for names in derived.values() if len(names) > 1}

        for family in _ASSET_FAMILIES:
            if not set(family) <= set(signatures):
                continue
            assert tuple(sorted(family)) in derived_families, (
                f"{family} is declared a family but the assets no longer group them together"
            )

        confirmable = {f for f in derived_families if set(f) <= set(signatures)}
        undeclared = sorted(confirmable - {tuple(sorted(f)) for f in _ASSET_FAMILIES})
        assert not undeclared, (
            "the assets group these robots identically but no family declares them, "
            f"so their joint counts are ungraded: {undeclared}"
        )


class TestTheRederivationReadsDiskOnly:
    """Re-deriving the grouping must not fetch an asset the machine lacks.

    :func:`_asset_signatures` walks all 72 registry entries, and the resolver's
    default is to download whatever it cannot find. That put a network fetch
    behind 63 of those entries on a fresh checkout - the case where there is no
    grouping to re-derive in the first place - and took this file past the 120s
    ``pytest-timeout`` budget.

    Both halves are needed. The behavioural cell reads the download hook, which
    is also silent on a machine that happens to hold every asset already; the
    structural cell pins the reason, so a passing run cannot be an accident of
    what is on disk.
    """

    @staticmethod
    def _download_attempts(names: Iterable[str], monkeypatch: Any) -> list[str]:
        """Names the walk asked the download hook to fetch."""
        from strands_robots.assets import manager

        attempted: list[str] = []

        def refuse(name: str, info: dict[str, Any]) -> bool:
            attempted.append(name)
            return False

        monkeypatch.setattr(manager, "_auto_download_robot", refuse)
        _asset_signatures(names)
        return attempted

    def test_no_download_when_the_cache_is_empty(self, registry: dict[str, Any], monkeypatch: Any) -> None:
        """An empty cache is a skip, not a fetch of the whole corpus.

        This is the state every machine is in under the conftest's asset
        isolation, and the one a fresh checkout is in for real: no XML anywhere,
        so the resolver's default reaches for the network 63 times.
        """
        pytest.importorskip("mujoco")
        attempted = self._download_attempts(registry, monkeypatch)
        assert not attempted, (
            f"re-deriving the grouping with an empty cache tried to download {len(attempted)} "
            f"assets (first few: {sorted(attempted)[:6]}). There is no grouping to confirm "
            "here, so it must skip - pass allow_download=False to resolve_model_path."
        )

    @pytest.mark.usefixtures("host_asset_cache")
    def test_no_download_when_the_cache_is_partial(self, registry: dict[str, Any], monkeypatch: Any) -> None:
        """A cached XML whose meshes are missing is the second fetch trigger.

        Distinct from the empty case: ``is_robot_asset_present`` is true for
        these, so guarding on presence alone still lets the resolver fetch -
        including for members of a graded family.
        """
        pytest.importorskip("mujoco")
        attempted = self._download_attempts(registry, monkeypatch)
        assert not attempted, (
            f"re-deriving the grouping against the host cache tried to download "
            f"{len(attempted)} assets (first few: {sorted(attempted)[:6]}); a mesh-less "
            "cached XML must be left to the caller, not fetched from a test."
        )

    def test_the_walk_asks_the_resolver_not_to_fetch(self) -> None:
        """Non-vacuity: the zero above is a refusal, not an absence of work."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(_asset_signatures)))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "resolve_model_path"
        ]
        assert len(calls) == 1, f"expected exactly one resolve_model_path call, found {len(calls)}"
        declined = {kw.arg: kw.value for kw in calls[0].keywords}.get("allow_download")
        assert isinstance(declined, ast.Constant) and declined.value is False, (
            "the registry walk must pass allow_download=False; without it the resolver "
            "downloads every asset this machine does not have"
        )
