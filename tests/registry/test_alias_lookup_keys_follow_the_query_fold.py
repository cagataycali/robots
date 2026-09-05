# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An alias is keyed by the same fold the query it answers is keyed by.

:func:`~strands_robots.registry.robots.resolve_name` folds a caller's query
before looking it up - lowercase, trimmed, dashes as underscores
(:func:`~strands_robots.registry.loader.normalize_robot_name`). Canonical robot
names are stored folded, so they match. Aliases were keyed as DECLARED, and that
asymmetry has two consequences, neither of them a failed lookup:

* An alias whose declared spelling is not already folded is unreachable in
  EVERY spelling, including the one it was registered with, because the query
  is folded before it reaches the map.
* The fold can carry it onto a DIFFERENT robot's key. ``aliases=["Franka-Panda"]``
  folds to ``franka_panda``, which the shipped registry already gives to
  ``panda``, so ``resolve_name("Franka-Panda")`` answered ``"panda"`` and
  ``get_robot`` returned the Franka's entry - a name declared for one robot
  resolving to another.

The uniqueness constraints are the other half of the same rule: ``_validate_robots``
compared declared spellings, so two aliases that are ONE key to every reader passed
validation, and the alias map then silently kept whichever entry merged last (the
user overlay). Both sides now compare under the fold, which is what lets the
fail-closed check in ``register_robot`` refuse such an alias at registration
instead of persisting a lookup that lands on someone else.

The carve-out: an alias that folds to its OWNER's canonical name is a second
spelling of that name, which the fold already accepts. The shipped registry
declares one (``reachy_mini`` aliases ``reachy-mini``), and
``_validate_policies`` already makes the same exception with ``alias !=
provider_name``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.registry import (
    get_robot,
    list_aliases,
    normalize_robot_name,
    register_robot,
    resolve_name,
)
from strands_robots.registry import loader as loader_mod

_MINIMAL_MJCF = '<mujoco><worldbody><body><geom size="0.1"/></body></worldbody></mujoco>'

#: A robot the shipped registry gives ``franka_panda`` to, as an alias.
_SHIPPED_ALIAS_OWNER = ("franka_panda", "panda")


@pytest.fixture
def robot_dir(tmp_path: Path) -> Path:
    """A minimal MJCF robot directory to register entries against."""
    d = tmp_path / "assets" / "bot"
    d.mkdir(parents=True)
    (d / "bot.xml").write_text(_MINIMAL_MJCF)
    return d


def _register(robot_dir: Path, name: str, aliases: list[str]) -> dict:
    """Register ``name`` with ``aliases`` against the temp asset dir."""
    return register_robot(
        name=name,
        model_xml="bot.xml",
        asset_dir=str(robot_dir),
        description="probe",
        category="arm",
        joints=6,
        aliases=aliases,
    )


class TestAnAliasIsReachableByItsDeclaredSpelling:
    """A declared alias answers the query, whatever spelling it was declared in."""

    @pytest.mark.parametrize(
        "declared",
        ["My-Arm-V2", "my-arm-v2", "MY_ARM_V2", "  my_arm_v2  ", "my_arm_v2"],
        ids=["dashed-mixed-case", "dashed", "upper", "padded", "already-folded"],
    )
    @pytest.mark.parametrize(
        "queried",
        ["My-Arm-V2", "my-arm-v2", "MY_ARM_V2", "  my_arm_v2  ", "my_arm_v2"],
        ids=["dashed-mixed-case", "dashed", "upper", "padded", "already-folded"],
    )
    def test_every_spelling_of_one_alias_reaches_the_robot_that_declared_it(
        self, robot_dir: Path, declared: str, queried: str
    ) -> None:
        """Declared spelling and queried spelling are independent of each other.

        Both go through the same fold, so the 25 pairs are one lookup. Only the
        already-folded declaration used to answer, which is the arrangement the
        pre-existing alias grader happened to pick.
        """
        _register(robot_dir, "probe_arm", [declared])

        assert resolve_name(queried) == "probe_arm"
        entry = get_robot(queried)
        assert entry is not None, f"{queried!r} reached no robot"
        assert entry["description"] == "probe", f"{queried!r} reached a different robot"

    def test_the_alias_map_is_keyed_by_the_fold_so_every_key_is_answerable(self) -> None:
        """No key of the shipped alias map is a spelling the query cannot produce.

        A key that is not its own fold can never be matched: the query is folded
        first, so nothing the caller types arrives in that form.
        """
        aliases = list_aliases()
        assert len(aliases) >= 100, f"premise: the shipped registry declares few aliases ({len(aliases)})"
        unreachable = {key: canonical for key, canonical in aliases.items() if key != normalize_robot_name(key)}
        assert not unreachable, f"alias keys no folded query can match: {unreachable}"


class TestAnAliasThatFoldsOntoAnotherRobotIsRefused:
    """The uniqueness constraints compare the keys, not the declared spellings."""

    def test_an_alias_folding_onto_a_shipped_alias_is_refused_not_silently_rerouted(self, robot_dir: Path) -> None:
        """The registration is refused, and nothing is persisted for it.

        Pre-fix this registration succeeded and ``resolve_name`` answered the
        OTHER robot, so the refusal and the absence of the entry are one claim:
        the alias never becomes a lookup that lands elsewhere.
        """
        shipped_alias, owner = _SHIPPED_ALIAS_OWNER
        assert resolve_name(shipped_alias) == owner, f"premise: {shipped_alias!r} no longer belongs to {owner!r}"

        declared = "Franka-Panda"
        assert normalize_robot_name(declared) == shipped_alias

        with pytest.raises(ValueError) as excinfo:
            _register(robot_dir, "probe_arm", [declared])
        message = str(excinfo.value)
        assert declared in message and owner in message, message
        assert shipped_alias in message, f"the shared key is not named: {message}"

        assert get_robot("probe_arm") is None, "the refused registration was persisted anyway"
        assert resolve_name(shipped_alias) == owner, "the shipped alias was rerouted"

    def test_an_alias_folding_onto_another_robots_canonical_name_is_refused(self, robot_dir: Path) -> None:
        """A canonical name wins the lookup, so such an alias could never answer."""
        assert get_robot("so100") is not None, "premise: so100 is not a shipped robot"
        assert normalize_robot_name("SO100") == "so100"

        with pytest.raises(ValueError, match="canonical robot name"):
            _register(robot_dir, "probe_arm", ["SO100"])

        assert get_robot("probe_arm") is None, "the refused registration was persisted anyway"

    def test_two_aliases_that_are_one_key_cannot_both_be_claimed(self, robot_dir: Path) -> None:
        """The pair the raw comparison passed and the alias map then collapsed."""
        _register(robot_dir, "first_arm", ["My-Shared-Alias"])
        assert resolve_name("my_shared_alias") == "first_arm"

        with pytest.raises(ValueError) as excinfo:
            _register(robot_dir, "second_arm", ["MY_SHARED_ALIAS"])
        assert "my_shared_alias" in str(excinfo.value), str(excinfo.value)

        # The first claimant keeps the key rather than losing it to the merge order.
        assert resolve_name("My-Shared-Alias") == "first_arm"


class TestTheShippedRegistryKeepsLoading:
    """Controls: the fold widens what validates, it does not narrow it."""

    def test_an_alias_that_folds_to_its_own_canonical_name_is_allowed(self) -> None:
        """The shipped self-alias, which a fold-blind collision check would refuse."""
        registry = json.loads((Path(loader_mod.__file__).parent / "robots.json").read_text())["robots"]
        self_aliased = {
            name: alias
            for name, info in registry.items()
            for alias in info.get("aliases", [])
            if normalize_robot_name(alias) == name and alias != name
        }
        assert self_aliased, "premise: no shipped alias is a second spelling of its own canonical name"

        for name, alias in self_aliased.items():
            assert resolve_name(alias) == name
            assert get_robot(alias) is not None

    def test_every_shipped_alias_still_resolves_to_the_robot_that_declares_it(self) -> None:
        """Re-keying the map moved no alias off its owner."""
        registry = json.loads((Path(loader_mod.__file__).parent / "robots.json").read_text())["robots"]
        declared = [(name, alias) for name, info in registry.items() for alias in info.get("aliases", [])]
        assert len(declared) >= 100, f"premise: few shipped aliases to check ({len(declared)})"

        misrouted = [(alias, name, resolve_name(alias)) for name, alias in declared if resolve_name(alias) != name]
        assert not misrouted, f"aliases that stopped resolving to their owner: {misrouted}"


class TestTheFoldHasOneOwner:
    """The rule is spelled once, so a reader and a validator cannot drift."""

    def test_no_registry_module_respells_the_fold(self) -> None:
        """Four modules folded names by hand; the fold now has a single owner."""
        package = Path(loader_mod.__file__).parent
        spelled = {
            path.name: path.read_text().count('.lower().strip().replace("-", "_")') for path in package.glob("*.py")
        }
        assert spelled.get("loader.py") == 1, f"premise: loader.py does not define the fold ({spelled})"
        respelled = {name: count for name, count in spelled.items() if count and name != "loader.py"}
        assert not respelled, f"modules respelling the fold instead of importing it: {respelled}"

    def test_the_fold_is_the_rule_its_callers_need(self) -> None:
        """The exported rule answers the same key the lookups are keyed by."""
        for spelling in ("My-Arm", "  MY-ARM  ", "my_arm"):
            assert normalize_robot_name(spelling) == "my_arm"
