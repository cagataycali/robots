"""``VeraPolicy.image_keys`` is a subset selector, so it is read by membership.

``image_keys`` names a SUBSET of the observation's own image keys - the cameras
to width-concat into the single ``(H, W, 3)`` frame the video planner acts on.
``None`` is documented as "the server's ``view_keys`` (from the connect
handshake), or every image key in the observation". Read by truthiness, every
other falsy value took that same branch, and for a selector that is not a wider
default but the *opposite* answer: a caller who selected no camera view was
served every one of them.

Four sites spelled the one parameter, and the two reads were independent
mistakes - the shape guard was gated on a truthy value, so it never saw the
empty selection; the store erased ``[]`` to ``None``, so the selection was gone
before the resolver ran; and the resolver read the attribute by truthiness too.

Measured before the fix, one ``VeraPolicy(embodiment="pusht",
auto_launch_server=False, image_keys=X)`` per row against a two-camera
observation, offline - no server, no ``vera`` package, no GPU:

| ``image_keys=`` | stored | resolved views (no server ``view_keys``) |
| --- | --- | --- |
| ``None`` | ``None`` | ``['...images.top', '...images.wrist']`` |
| ``[]``   | **``None``** | **``['...images.top', '...images.wrist']``** |
| ``""``   | **``None``** | **``['...images.top', '...images.wrist']``** |
| ``['...images.wrist']`` | ``['...images.wrist']`` | ``['...images.wrist']`` |

``[]`` and ``""`` were byte-identical to ``None`` at both surfaces, under a
server that advertised ``view_keys`` and under one that did not.

The consequence is silent in the direction that cannot be noticed: the excluded
cameras are concatenated into a *wider* frame and the rollout succeeds, so the
mistake presents as a policy that does not solve. The one refusal that could
have caught it - ``_extract_frame``'s "requires at least one camera frame" -
was unreachable for an empty selection, because by then it had been widened.

``image_keys`` reaches this from configuration: it is a ``VeraPolicy``
constructor keyword, so it arrives through ``create_policy("vera", ...)`` and
``policy_config``, and an empty list is what a filter that matched nothing
produces.

The list shape stays with the shared name-list domain (a bare string is
iterable per character, a repeat concatenates one panel twice); only the
emptiness verdict is local, which is what that domain's docstring reserves for
the caller. ``""`` is refused by the shape half, as the bare string it is.

Everything here is offline - no server, no socket, no ``vera`` package, no GPU.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.vera import VeraPolicy

TOP = "observation.images.top"
WRIST = "observation.images.wrist"


def _two_camera_observation() -> dict[str, Any]:
    """An observation carrying two distinguishable camera frames."""
    return {
        TOP: np.zeros((8, 8, 3), dtype=np.uint8),
        WRIST: np.full((8, 8, 3), 255, dtype=np.uint8),
        "observation.state": np.zeros((6,), dtype=np.float32),
    }


def _policy(**kwargs: Any) -> VeraPolicy:
    """A provider built without launching or contacting a server."""
    return VeraPolicy(embodiment="pusht", auto_launch_server=False, **kwargs)


class TestAnEmptySelectionIsRefusedRatherThanWidened:
    """``[]`` asks for no view, so it cannot resolve to every view."""

    @pytest.mark.parametrize("empty", [[], ()], ids=["list", "tuple"])
    def test_an_empty_selection_is_refused(self, empty: Any) -> None:
        with pytest.raises(ValueError) as exc:
            _policy(image_keys=empty)
        assert "image_keys" in str(exc.value)

    def test_the_refusal_names_the_spelling_that_means_every_view(self) -> None:
        """The remedy has to name ``None``, which is the branch it stopped taking."""
        with pytest.raises(ValueError) as exc:
            _policy(image_keys=[])
        message = str(exc.value)
        assert "image_keys=None" in message
        assert "no camera view" in message

    def test_the_refusal_precedes_the_server_runner(self) -> None:
        """A refused selection leaves no runner to stop and no server to reap."""
        started: list[str] = []

        class _SpyRunner:
            def start(self) -> None:
                started.append("start")

            def stop(self) -> None:
                started.append("stop")

        with pytest.raises(ValueError):
            VeraPolicy(embodiment="pusht", image_keys=[], server_runner=_SpyRunner())  # type: ignore[arg-type]
        assert started == []


class TestAnEmptySelectionIsNeverServedAnUnselectedCamera:
    """The behaviour the refusal exists for, asserted on the frame itself."""

    def test_a_one_camera_selection_builds_a_one_camera_frame(self) -> None:
        policy = _policy(image_keys=[WRIST])
        frame = policy._extract_frame(_two_camera_observation(), {})
        both = _policy()._extract_frame(_two_camera_observation(), {})
        assert frame.shape[1] * 2 == both.shape[1]
        # The selected camera is the one in the frame, not the excluded one.
        assert int(frame.min()) == 255

    def test_an_empty_attribute_resolves_to_no_camera_and_the_frame_is_refused(self) -> None:
        """The resolver's own read, reached by assigning the attribute directly.

        The constructor keeps ``[]`` from arriving here, so this pins the second
        of the two independent truthiness reads: an empty selection resolves to
        no camera and ``_extract_frame`` refuses, rather than the resolver
        answering it with every camera in the observation.
        """
        policy = _policy()
        policy.image_keys = []
        assert policy._resolve_view_keys(_two_camera_observation(), {}) == []
        with pytest.raises(ValueError):
            policy._extract_frame(_two_camera_observation(), {})


class TestTheDocumentedSpellingsAreUnchanged:
    """Controls: the narrowing is confined to the empty selection."""

    def test_none_resolves_to_every_image_key_when_the_server_names_no_view(self) -> None:
        policy = _policy(image_keys=None)
        assert policy.image_keys is None
        assert policy._resolve_view_keys(_two_camera_observation(), {}) == [TOP, WRIST]

    def test_none_resolves_to_the_server_views_when_the_handshake_names_them(self) -> None:
        policy = _policy(image_keys=None)
        resolved = policy._resolve_view_keys(_two_camera_observation(), {"view_keys": [TOP]})
        assert resolved == [TOP]

    def test_a_real_subset_is_stored_and_resolved_as_written(self) -> None:
        policy = _policy(image_keys=[WRIST])
        assert policy.image_keys == [WRIST]
        assert policy._resolve_view_keys(_two_camera_observation(), {}) == [WRIST]
        # An explicit selection outranks the server's advertised views.
        assert policy._resolve_view_keys(_two_camera_observation(), {"view_keys": [TOP]}) == [WRIST]

    def test_the_selection_is_copied_so_a_later_mutation_cannot_reach_the_policy(self) -> None:
        selection = [WRIST]
        policy = _policy(image_keys=selection)
        selection.append(TOP)
        assert policy.image_keys == [WRIST]


class TestTheShapeVerdictStaysWithTheSharedDomain:
    """Controls: the spellings the name-list domain already refused still are."""

    def test_a_bare_string_is_refused_as_one_name_per_character(self) -> None:
        with pytest.raises(ValueError) as exc:
            _policy(image_keys=WRIST)  # type: ignore[arg-type]
        assert "must be a list of names, not a single string" in str(exc.value)

    def test_an_empty_string_is_refused_by_the_same_half(self) -> None:
        """``""`` was widened exactly like ``[]``; it is a bare string, so it takes that verdict."""
        with pytest.raises(ValueError) as exc:
            _policy(image_keys="")  # type: ignore[arg-type]
        assert "must be a list of names, not a single string" in str(exc.value)

    def test_a_repeated_name_is_still_refused(self) -> None:
        with pytest.raises(ValueError) as exc:
            _policy(image_keys=[WRIST, WRIST])
        assert "repeat" in str(exc.value)


class TestTheRuleDoesNotReachActionMapping:
    """Recorded so the narrowing is not re-proposed one line further down.

    ``action_mapping`` reads the same way (``dict(action_mapping) if
    action_mapping else None``) and is correct: it is a rename mapping rather
    than a subset of a collection the call owns, so ``{}`` and ``None`` both
    honestly mean "no rename, columns keep their server names".
    """

    def test_an_empty_action_mapping_is_accepted_and_means_no_rename(self) -> None:
        assert _policy(action_mapping={}).action_mapping is None
        assert _policy(action_mapping=None).action_mapping is None


def _reads_image_keys_bare(test: ast.expr) -> bool:
    """``image_keys`` used as a condition on its own, i.e. read by truthiness."""
    return (isinstance(test, ast.Name) and test.id == "image_keys") or (
        isinstance(test, ast.Attribute) and test.attr == "image_keys"
    )


def _image_keys_reads(func: Any) -> tuple[list[ast.expr], list[ast.Compare]]:
    """Every conditional read of ``image_keys`` in ``func``, split by how it reads."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    bare = [
        node.test
        for node in ast.walk(tree)
        if isinstance(node, (ast.If, ast.IfExp)) and _reads_image_keys_bare(node.test)
    ]
    membership = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops)
        and _reads_image_keys_bare(node.left)
    ]
    return bare, membership


class TestEveryReadOfTheSelectionUsesMembership:
    """The third read, which no value can reach any more, graded structurally.

    The three reads this module names are pinned by behaviour above except one.
    The shape guard is pinned by the refusal; the resolver is pinned by an
    attribute assigned after construction. The store runs only at construction,
    *below* a guard that now refuses every empty value, so no value can reach it
    to tell the two spellings apart - yet it is the line that erased ``[]`` to
    ``None`` and it is the reason the resolver never saw a selection to widen.

    Its spelling is a ternary rather than an ``if``, so both conditional forms
    are scanned: a check written for statement conditionals alone leaves the
    store free to return to truthiness, and a later change that relaxed or moved
    the refusal would then restore the widening with nothing failing.
    """

    @pytest.mark.parametrize(
        "func",
        [VeraPolicy.__init__, VeraPolicy._resolve_view_keys],
        ids=["__init__", "_resolve_view_keys"],
    )
    def test_no_conditional_reads_the_selection_by_truthiness(self, func: Any) -> None:
        bare, _ = _image_keys_reads(func)
        assert not bare, (
            f"{func.__qualname__} reads image_keys by truthiness at line(s) "
            f"{[node.lineno for node in bare]}, which reads an empty selection as absent "
            "and widens it to every view"
        )

    @pytest.mark.parametrize(
        "func",
        [VeraPolicy.__init__, VeraPolicy._resolve_view_keys],
        ids=["__init__", "_resolve_view_keys"],
    )
    def test_the_scan_is_looking_at_a_membership_read(self, func: Any) -> None:
        """Non-vacuity: deleting the reads would satisfy the rule above."""
        _, membership = _image_keys_reads(func)
        assert membership, f"{func.__qualname__} does not read image_keys against None at all"

    def test_the_constructor_holds_both_conditional_forms(self) -> None:
        """The gate is an ``if`` and the store is a ternary, so both are in scope.

        Recorded because the two forms are what a partial scan splits: the counts
        below are the reason :meth:`test_no_conditional_reads_the_selection_by_truthiness`
        cannot be written against ``ast.If`` alone.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(VeraPolicy.__init__)))
        statements = [n for n in ast.walk(tree) if isinstance(n, ast.If)]
        ternaries = [n for n in ast.walk(tree) if isinstance(n, ast.IfExp)]
        assert statements, "the constructor no longer guards with an if statement"
        assert ternaries, "the constructor no longer stores the selection through a ternary"
