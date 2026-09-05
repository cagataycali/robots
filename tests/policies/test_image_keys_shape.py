# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A single camera key passed as a bare string must not be read one key per character.

``image_keys`` names an ordered list of key names on two providers: the LeRobot
local provider declares model VISUAL feature keys with it
(:func:`~strands_robots.policies.lerobot_local.molmoact2.derive_image_keys`), and
the VERA provider names the observation cameras to width-concat with it
(``VeraPolicy._resolve_view_keys``). Neither validated the shape, and both reduced
the value with ``list(...)``.

``str`` is iterable, so ``list("wrist")`` is ``['w', 'r', 'i', 's', 't']`` - five
names the caller never wrote. Nothing downstream can tell that apart from a
deliberate five-entry list, so it was accepted and the consequence surfaced far
from the call:

* LeRobot, with no embodiment configured (``preflight`` returned early in that
  case): a model built declaring one bogus VISUAL feature per character;
* VERA: ``KeyError: 'w'`` raised from ``_extract_frame`` mid-rollout, after the
  policy server had been launched and the model loaded.

Two neighbouring shapes failed the same way: a non-``str`` entry became a key,
and a repeated entry could not be honored as written - the LeRobot side builds a
feature dict, where a duplicate collapses and declares fewer features than asked
for, and the VERA side concatenates one panel per entry, where a duplicate
doubles the width of the frame the model sees.

These tests pin the shared domain
(:func:`strands_robots.utils.name_list_error`), that all four surfaces that
receive the value agree on the SHAPE, and that each refusal precedes the
expensive work it guards.

The emptiness verdict is deliberately not part of that agreement, because the
two providers do not name the same kind of thing with this parameter. The
LeRobot side DECLARES the model's visual features, and absence derives them from
the embodiment, so an empty list there keeps its "not supplied" meaning. The
VERA side SELECTS a subset of the observation it was handed, so an empty
selection asks for no view and cannot be a spelling of "every view" - it is
refused, which is the verdict the shared domain reserves for the caller.
:class:`TestTheEmptinessVerdictIsPerSurface` states that divergence rather than
leaving it to be read out of a parity table.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from strands_robots.policies.lerobot_local import molmoact2
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.vera import VeraPolicy
from strands_robots.utils import name_list_error

# Shapes that cannot be honored, with the reason each is refused for. Kept as a
# table so every surface below is probed with exactly the same set.
BAD_SHAPES: list[tuple[str, Any, str]] = [
    ("a single name as a bare string", "observation.images.image", "not a single string"),
    ("a short bare string", "wrist", "not a single string"),
    ("bytes", b"wrist", "not a single string"),
    ("a mapping", {"observation.images.image": 1}, "not a mapping"),
    ("an int", 3, "must be a list of names"),
    ("a one-shot iterator", iter(["a", "b"]), "must be a list of names"),
    ("a non-str entry", ["observation.images.image", 7], "[1] must be a name (str)"),
    ("a blank entry", ["observation.images.image", "  "], "[1] must be a non-blank name"),
    ("a repeated entry", ["obs.a", "obs.b", "obs.a"], "must not repeat a name"),
]

GOOD_SHAPES: list[tuple[str, Any]] = [
    ("one name", ["observation.images.image"]),
    ("two names", ["observation.images.image", "observation.images.wrist_image"]),
    ("a tuple", ("observation.images.image",)),
]

# ``None`` means "not supplied" on every surface, which derives the list instead.
ABSENT_VALUES: list[tuple[str, Any]] = [("None", None)]

# An empty list is a usable SHAPE whose meaning is the receiving surface's to
# decide, so it is tabled separately from ``None``: it is "not supplied" to the
# LeRobot declaration and an empty SELECTION - refused - to VERA. It used to sit
# in ``ABSENT_VALUES``, which is what let the VERA cell below assert that a
# selection naming no camera view falls back to every one of them.
EMPTY_SELECTION: list[tuple[str, Any]] = [("an empty list", [])]


class _FakeVeraClient:
    """In-memory stand-in for the VERA websocket client (no server, no GPU)."""

    def __init__(self) -> None:
        self.handshakes = 0

    def get_server_metadata(self) -> dict[str, Any]:
        self.handshakes += 1
        return {"view_keys": ["front"], "needs_prompt": False}

    def infer(self, req: dict[str, Any]) -> dict[str, Any]:
        return {"action": np.zeros((1, 6), np.float32)}

    def close(self) -> None:
        pass


def _vera(image_keys: Any, client: _FakeVeraClient | None = None) -> VeraPolicy:
    return VeraPolicy(
        embodiment="pusht",
        image_keys=image_keys,
        # Structural stub: get_server_metadata / infer / close is the whole
        # surface VeraPolicy uses, and the guard under test runs before any of it.
        client=client or _FakeVeraClient(),  # type: ignore[arg-type]
        auto_launch_server=False,
    )


def _observation() -> dict[str, np.ndarray]:
    return {
        "front": np.zeros((64, 64, 3), np.uint8),
        "wrist": np.full((64, 64, 3), 200, np.uint8),
    }


# --------------------------------------------------------------------------- #
# The shared domain
# --------------------------------------------------------------------------- #
class TestNameListDomain:
    @pytest.mark.parametrize(("label", "value", "reason"), BAD_SHAPES, ids=[c[0] for c in BAD_SHAPES])
    def test_a_shape_that_cannot_be_honored_is_reported_with_its_own_reason(
        self, label: str, value: Any, reason: str
    ) -> None:
        err = name_list_error(value, "image_keys", "Surface")
        assert err is not None, f"{label} was accepted"
        assert reason in err
        assert err.startswith("Surface: image_keys")

    @pytest.mark.parametrize(("label", "value"), GOOD_SHAPES, ids=[c[0] for c in GOOD_SHAPES])
    def test_a_usable_list_is_accepted(self, label: str, value: Any) -> None:
        assert name_list_error(value, "image_keys", "Surface") is None

    def test_the_bare_string_message_quotes_the_per_character_reading_and_the_fix(self) -> None:
        """The remedy has to be copy-pasteable, and the cause has to be visible.

        Naming only the type leaves the caller to work out why a string is not a
        list of one name; quoting what it would have been read as is what makes
        the per-character split self-evident.
        """
        err = name_list_error("wrist", "image_keys", "VeraPolicy")
        assert err is not None
        assert "['w', 'r', 'i', 's', 't']" in err
        assert "5 name(s)" in err
        assert "['wrist']" in err

    def test_an_empty_sequence_is_left_to_the_caller_to_interpret(self) -> None:
        assert name_list_error([], "image_keys", "Surface") is None


# --------------------------------------------------------------------------- #
# The defect, on the LeRobot feature-declaration path
# --------------------------------------------------------------------------- #
class TestLerobotFeatureKeys:
    def test_a_bare_string_is_not_declared_as_one_feature_per_character(self) -> None:
        with pytest.raises(ValueError, match="not a single string"):
            molmoact2.derive_image_keys("observation.images.image", None)  # type: ignore[arg-type]

    def test_a_repeated_key_is_refused_rather_than_collapsing_the_feature_dict(self) -> None:
        with pytest.raises(ValueError, match="must not repeat a name"):
            molmoact2.derive_image_keys(["obs.a", "obs.a"], None)

    def test_a_usable_list_is_still_honored_verbatim(self) -> None:
        keys = ["observation.images.image", "observation.images.wrist_image"]
        assert molmoact2.derive_image_keys(keys, None) == keys

    @pytest.mark.parametrize(
        ("label", "value"),
        ABSENT_VALUES + EMPTY_SELECTION,
        ids=[c[0] for c in ABSENT_VALUES + EMPTY_SELECTION],
    )
    def test_an_absent_value_still_derives_the_default_list(self, label: str, value: Any) -> None:
        """Both spellings are "not supplied" here: this parameter declares the
        model's features rather than selecting a subset of a collection the call
        owns, so absence derives them and an empty list is that same absence."""
        assert molmoact2.derive_image_keys(value, None) == list(molmoact2.DEFAULT_IMAGE_KEYS)

    def test_the_refusal_precedes_the_weight_load(self) -> None:
        """A shape mistake must not cost a multi-minute download first."""
        with patch.object(LerobotLocalPolicy, "_load_model") as load:
            with pytest.raises(ValueError, match="not a single string"):
                LerobotLocalPolicy(pretrained_name_or_path="org/mm2", image_keys="wrist")  # type: ignore[arg-type]
        load.assert_not_called()


# --------------------------------------------------------------------------- #
# The pre-flight surface, including the path that used to return early
# --------------------------------------------------------------------------- #
class TestPreflight:
    def test_the_shape_is_checked_with_no_embodiment_configured(self) -> None:
        """``preflight`` returns early when no embodiment is set, but
        ``image_keys`` is honored on that path too - so it is checked first."""
        with pytest.raises(ValueError, match="not a single string"):
            LerobotLocalPolicy.preflight({"front"}, image_keys="wrist")

    def test_the_shape_is_checked_with_an_embodiment_configured(self) -> None:
        with pytest.raises(ValueError, match="not a single string"):
            LerobotLocalPolicy.preflight({"front", "wrist"}, embodiment="so_real", image_keys="wrist")

    def test_a_usable_list_still_reaches_the_undeclared_feature_check(self) -> None:
        """The sibling check that reports a list withholding a fed feature must
        keep working: the shape guard runs before it, not instead of it."""
        with pytest.raises(ValueError, match="does not declare them"):
            LerobotLocalPolicy.preflight(
                {"front", "wrist"},
                embodiment="so_real",
                policy_type="molmoact2",
                pretrained_name_or_path="org/mm2",
                image_keys=["observation.images.image"],
            )

    def test_an_absent_value_is_not_refused(self) -> None:
        LerobotLocalPolicy.preflight({"front"})
        LerobotLocalPolicy.preflight({"front"}, image_keys=None)
        LerobotLocalPolicy.preflight({"front"}, image_keys=[])


# --------------------------------------------------------------------------- #
# The defect, on the VERA camera-key path
# --------------------------------------------------------------------------- #
class TestVeraViewKeys:
    def test_a_bare_string_is_refused_instead_of_raising_keyerror_mid_rollout(self) -> None:
        with pytest.raises(ValueError, match="not a single string"):
            _vera("front")

    def test_the_refusal_precedes_the_server_handshake(self) -> None:
        client = _FakeVeraClient()
        with pytest.raises(ValueError, match="not a single string"):
            _vera("front", client=client)
        assert client.handshakes == 0

    def test_a_usable_list_still_selects_one_panel_per_named_view(self) -> None:
        policy = _vera(["front"])
        frame = policy._extract_frame(_observation(), {"view_keys": ["front"]})
        one_panel_width = frame.shape[1]

        policy = _vera(["front", "wrist"])
        frame = policy._extract_frame(_observation(), {"view_keys": ["front"]})
        assert frame.shape[1] == 2 * one_panel_width

    def test_a_repeated_view_is_refused_rather_than_doubling_the_frame_width(self) -> None:
        """A duplicate silently widened the frame the model sees, which is a
        different input from the single view the caller named."""
        with pytest.raises(ValueError, match="must not repeat a name"):
            _vera(["front", "front"])

    @pytest.mark.parametrize(("label", "value"), ABSENT_VALUES, ids=[c[0] for c in ABSENT_VALUES])
    def test_an_absent_value_still_falls_back_to_the_server_views(self, label: str, value: Any) -> None:
        policy = _vera(value)
        assert policy.image_keys is None
        assert policy._resolve_view_keys(_observation(), {"view_keys": ["front"]}) == ["front"]

    @pytest.mark.parametrize(("label", "value"), EMPTY_SELECTION, ids=[c[0] for c in EMPTY_SELECTION])
    def test_an_empty_selection_is_refused_rather_than_falling_back(self, label: str, value: Any) -> None:
        """This row used to be one of the ``ABSENT_VALUES`` above.

        It is retargeted rather than dropped, because the assertion it made was
        the defect: ``image_keys`` selects a subset of the observation's cameras,
        so an empty selection asks for no view, and falling back served every one
        of them - the opposite answer, under a success result. The full domain,
        with the frame it produces, is pinned by
        ``tests/policies/vera/test_vera_image_keys_selection_domain.py``.
        """
        with pytest.raises(ValueError, match="selects no camera view"):
            _vera(value)


# --------------------------------------------------------------------------- #
# The surfaces must not diverge
# --------------------------------------------------------------------------- #
def _verdict(call: Any) -> str:
    try:
        call()
    except ValueError:
        return "refused"
    return "accepted"


class TestCrossProviderParity:
    """One option name, one shape contract.

    The two providers name different vocabularies with ``image_keys`` - model
    feature keys against observation camera keys - but a value either is a list
    of distinct names or is not, so a shape refused by one surface cannot be
    accepted by another.

    The empty list is not a shape question and is therefore not in this table;
    :class:`TestTheEmptinessVerdictIsPerSurface` covers it.
    """

    @pytest.mark.parametrize(
        ("label", "value"),
        [(c[0], c[1]) for c in BAD_SHAPES] + GOOD_SHAPES + ABSENT_VALUES,
        ids=[c[0] for c in BAD_SHAPES] + [c[0] for c in GOOD_SHAPES] + [c[0] for c in ABSENT_VALUES],
    )
    def test_every_surface_reaches_the_same_verdict(self, label: str, value: Any) -> None:
        # A one-shot iterator is consumed by whichever surface reads it first,
        # so each gets its own.
        def fresh() -> Any:
            return iter(["a", "b"]) if label == "a one-shot iterator" else value

        with patch.object(LerobotLocalPolicy, "_load_model"):
            verdicts = {
                "LerobotLocalPolicy": _verdict(lambda: LerobotLocalPolicy(image_keys=fresh())),
                "preflight": _verdict(lambda: LerobotLocalPolicy.preflight({"front"}, image_keys=fresh())),
                "derive_image_keys": _verdict(lambda: molmoact2.derive_image_keys(fresh(), None)),
                "VeraPolicy": _verdict(lambda: _vera(fresh())),
            }
        assert len(set(verdicts.values())) == 1, f"surfaces disagree for {label}: {verdicts}"


# --------------------------------------------------------------------------- #
# The one value whose verdict is the surface's own
# --------------------------------------------------------------------------- #
class TestTheEmptinessVerdictIsPerSurface:
    """An empty list is a usable shape, so the shared domain returns nothing.

    ``name_list_error([])`` is ``None`` on purpose - "a surface where an absent
    value IS an error keeps that verdict its own" - and the two surfaces reach
    opposite verdicts because they name different contracts with the one option
    name. Asserted here so the divergence is deliberate and stays visible: a
    later sweep that made either side match the other would have to delete this.
    """

    def test_the_shared_domain_returns_no_verdict(self) -> None:
        assert name_list_error([], "image_keys", "Surface") is None

    def test_the_declaration_reads_it_as_not_supplied(self) -> None:
        assert molmoact2.derive_image_keys([], None) == list(molmoact2.DEFAULT_IMAGE_KEYS)
        LerobotLocalPolicy.preflight({"front"}, image_keys=[])

    def test_the_selection_refuses_it(self) -> None:
        with pytest.raises(ValueError, match="selects no camera view"):
            _vera([])
