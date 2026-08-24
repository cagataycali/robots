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
receive the value agree on it, that each refusal precedes the expensive work it
guards, and that a falsy value keeps its "not supplied" meaning.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

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

# Falsy values mean "not supplied" in both providers, which derive the list
# instead. The guards are gated on a truthy value so that meaning is preserved.
ABSENT_VALUES: list[tuple[str, Any]] = [("None", None), ("an empty list", [])]


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


# --------------------------------------------------------------------------- #
# The pre-flight surface, including the path that used to return early
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# The surfaces must not diverge
# --------------------------------------------------------------------------- #
def _verdict(call: Any) -> str:
    try:
        call()
    except ValueError:
        return "refused"
    return "accepted"


