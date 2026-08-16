"""``VeraConfig.render_width`` is a pixel count, so it takes the shared media domain.

The per-view render width is the same quantity as the recorders' ``width`` /
``height`` and :attr:`~strands_robots.rendering.HybridCompositor.default_width`,
all of which are held to
:func:`strands_robots.utils.positive_whole_number_error` - the guard whose own
docstring names "the media knobs that count frames or pixels" as one of the two
families it exists for. ``render_width`` was not one of its callers. It was
declared ``int | None``, defaulted per embodiment, settable through
``VERA_RENDER_WIDTH`` or a pre-built :class:`VeraConfig` (it is not a
``VeraPolicy`` keyword, unlike the two ports), and then read twice under a
truthiness coalesce with an ``int()`` at each site:

    _extract_frame:  rw    = int(self.config.render_width or 128)
    _infer:          per_w = int(self.config.render_width or (context_rgb.shape[2] // n_views))

Measured on ``94de3a5``, one ``VeraConfig(embodiment="mimicgen", render_width=X)``
per row, no ``vera`` package and no server:

| ``render_width=`` | stored | what reached the planner |
| --- | --- | --- |
| ``0``      | ``0``      | **128** - the falsy width was replaced by the default |
| ``False``  | ``False``  | **128** - same, via ``bool`` being falsy |
| ``True``   | ``True``   | a **1x1 pixel** view, under a success result |
| ``2.7``    | ``2.7``    | a **2x2 pixel** view, under a success result |
| ``128.0``  | ``128.0``  | 128, but the field kept the ``float`` |
| ``"128"``  | ``'128'``  | 128, via ``int("128")`` |
| ``-1``     | ``-1``     | ``ValueError: Number of samples, -1, must be non-negative`` |
| ``"abc"``  | ``'abc'``  | ``ValueError: invalid literal for int() with base 10: 'abc'`` |
| ``nan``    | ``nan``    | ``ValueError: cannot convert float NaN to integer`` |
| ``inf``    | ``inf``    | ``OverflowError: cannot convert float infinity to integer`` |
| ``[128]``  | ``[128]``  | ``TypeError: int() argument must be ... not 'list'`` |

Two properties make this worth a domain rather than a nicer default.

The silent rows are silent in the direction that cannot be noticed: a 1x1 or 2x2
view is a *successful* rollout against the WAN/DFoT planner, so the mistake shows
up as a policy that does not solve rather than as an error.

The raising rows escape past the envelope, and they escape **late**. Every one of
them raises out of ``_extract_frame``, which ``get_actions`` calls *after*
``_ensure_started`` has launched the WAN server subprocess and completed the
handshake - so a width that was wrong at construction costs a model load (up to
``server_ready_timeout``, 600s by default) before anything reports it, and the
message names neither ``render_width`` nor ``VeraConfig``. That is the same
argument the two ports on this dataclass are already validated on, and it is why
the guard belongs at the config funnel: "refusing before any client or runner is
built leaves nothing half-configured behind".

One thing that measured *better* than expected, recorded so it is not
re-proposed as a defect: the two read sites' fallbacks (``128`` and
``context_rgb.shape[2] // n_views``) never actually disagreed. They coincide
because ``_extract_frame`` resized every view to the first expression's value
before the concatenation the second one divides back out. So the second fallback
was a redundant second definition of the same case rather than a divergence -
but nothing asserted the two must agree, which is what
:class:`TestBothReadSitesNameTheSameWidth` now does.

Everything here is offline - no server, no socket, no ``vera`` package, no GPU.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, cast

import numpy as np
import pytest

from strands_robots.policies.vera import VeraConfig, VeraPolicy

# Widths no view can be rendered at. ``0`` and ``False`` are in this set because
# a zero-width view carries no pixels; ``True`` and ``2.7`` because they were
# honored as a silent 1 and 2; ``"128"`` because the shared domain admits only a
# real scalar, which is the rule every other caller of it applies.
UNUSABLE_WIDTHS: list[Any] = [
    0,
    -1,
    True,
    False,
    2.7,
    "128",
    "abc",
    "",
    math.nan,
    math.inf,
    -math.inf,
    [128],
    {},
    object(),
]

# Widths the resize and the wire payload can both honor.
USABLE_WIDTHS: list[int] = [1, 8, 128, 252]


def _config(**kwargs: Any) -> VeraConfig:
    """Build a config through the funnel, splatted so off-type values reach it.

    mypy does not narrow a ``**dict[str, Any]`` splat, which is what lets a test
    hand a deliberately wrong type to a field annotated ``int | None``.
    """
    return VeraConfig(**kwargs)


class _FakeClient:
    """Scriptable VeraWebsocketClient stand-in (no socket)."""

    def __init__(self, metadata: dict, action_chunk: Any) -> None:
        self._meta = metadata
        self._chunk = np.asarray(action_chunk, dtype=np.float32)
        self.infer_requests: list[dict] = []

    def get_server_metadata(self) -> dict:
        return dict(self._meta)

    def infer(self, observation: dict) -> dict:
        self.infer_requests.append(observation)
        return {"action": self._chunk}

    def reset(self, reset_info: Any = None) -> None:
        pass

    def configure(self, params: dict) -> dict:
        return {"applied": params}

    def close(self) -> None:
        pass


def _cam(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _run(width: int, n_views: int = 1) -> dict:
    """Drive one ``get_actions`` at ``width`` and return the wire payload."""
    client = _FakeClient({"action_space": "pos", "context_frames": 1}, [[0.0]])
    policy = VeraPolicy(
        # The stand-in is duck-typed against the four client methods the provider
        # calls, not a ``VeraWebsocketClient`` subclass, so the parameter's type
        # is widened here rather than the fake being grown a socket it must not
        # open. Cast at the boundary keeps the rest of this helper checked.
        client=cast(Any, client),
        config=_config(embodiment="mimicgen", render_width=width, auto_launch_server=False),
    )
    obs = {f"cam{i}": _cam() for i in range(n_views)}
    asyncio.run(policy.get_actions(obs, ""))
    return client.infer_requests[-1]


# --------------------------------------------------------------------------- #
# The domain - refused at the funnel, by name
# --------------------------------------------------------------------------- #
class TestRenderWidthDomain:
    @pytest.mark.parametrize("value", UNUSABLE_WIDTHS)
    def test_an_unusable_render_width_is_refused_by_name(self, value):
        """The refusal names the field, the class and the accepted domain."""
        with pytest.raises(ValueError, match="render_width"):
            _config(embodiment="mimicgen", render_width=value)

    @pytest.mark.parametrize("value", UNUSABLE_WIDTHS)
    def test_the_refusal_replaces_a_bare_coercion_error(self, value):
        """No ``int()`` / numpy / PIL failure escapes instead of the verdict.

        Before the guard these left the provider as a ``ValueError``,
        ``OverflowError`` or ``TypeError`` from ``int()`` or from
        ``np.linspace``, none of them naming the field, and all of them raised
        per frame rather than once at construction.
        """
        try:
            _config(embodiment="mimicgen", render_width=value)
        except ValueError as exc:
            text = str(exc)
            assert "VeraConfig" in text
            assert "render_width" in text
            assert "positive whole number" in text
        else:  # pragma: no cover - the parametrization is all unusable
            pytest.fail(f"render_width={value!r} was accepted")

    @pytest.mark.parametrize("value", USABLE_WIDTHS)
    def test_a_usable_render_width_is_accepted(self, value):
        assert _config(embodiment="mimicgen", render_width=value).render_width == value

    def test_an_omitted_width_still_takes_the_per_embodiment_default(self):
        """``None`` means "apply the default", so it is defaulted, never refused.

        Same opt-out the recorders' ``width`` / ``height`` and
        ``HybridCompositor.default_width`` give ``None``: it selects a value
        rather than being one.
        """
        assert _config(embodiment="pusht").render_width == 252
        assert _config(embodiment="mimicgen").render_width == 128
        assert _config(embodiment="droid").render_width == 128


# --------------------------------------------------------------------------- #
# Normalization - the obligation the shared domain documents for its callers
# --------------------------------------------------------------------------- #
class TestTheWidthIsNormalizedForItsConsumers:
    """The domain admits any real scalar with an integral value, so passing it is
    a promise the width *can* be honored - not that it is already in the form
    ``Image.resize`` and the ``view_widths`` payload need."""

    def test_an_integral_float_is_normalized_to_a_plain_int(self):
        cfg = _config(embodiment="mimicgen", render_width=128.0)
        assert cfg.render_width == 128
        assert type(cfg.render_width) is int

    def test_a_numpy_integer_does_not_outlive_the_boundary(self):
        cfg = _config(embodiment="mimicgen", render_width=np.int64(64))
        assert cfg.render_width == 64
        assert type(cfg.render_width) is int

    def test_the_default_is_a_plain_int_too(self):
        assert type(_config(embodiment="pusht").render_width) is int


# --------------------------------------------------------------------------- #
# The property the two fallbacks used to define twice
# --------------------------------------------------------------------------- #
class TestBothReadSitesNameTheSameWidth:
    """``sum(view_widths)`` must equal the concatenated context width.

    ``_extract_frame`` resizes each view and ``_infer`` declares the per-view
    width on the wire. Those were two expressions with two different fallbacks,
    and this is the invariant that made them agree.
    """

    @pytest.mark.parametrize("width", [1, 8, 64])
    @pytest.mark.parametrize("n_views", [1, 2, 3])
    def test_the_declared_widths_span_the_context_tensor(self, width, n_views):
        req = _run(width, n_views)
        assert req["view_widths"] == [width] * n_views
        assert sum(req["view_widths"]) == req["context_rgb"].shape[2]
        assert req["context_rgb"].shape[1] == width  # each view squared

    @pytest.mark.parametrize("width", [1, 8, 64])
    def test_the_wire_width_is_a_plain_int(self, width):
        """A ``np.int64`` in ``view_widths`` would not survive msgpack cleanly."""
        req = _run(width)
        assert all(type(w) is int for w in req["view_widths"])


# --------------------------------------------------------------------------- #
# The silent substitutions are gone
# --------------------------------------------------------------------------- #
class TestTheSilentSubstitutionsAreGone:
    """Each of these was a ``status``-clean rollout at a width nobody asked for."""

    def test_zero_is_no_longer_replaced_by_the_default(self):
        """``render_width=0`` used to produce a 128-wide view, not a refusal."""
        with pytest.raises(ValueError, match="render_width"):
            _config(embodiment="mimicgen", render_width=0)

    def test_a_bool_is_no_longer_honored_as_a_one_pixel_view(self):
        """``True`` is an ``int`` subclass; it rendered a 1x1 view."""
        with pytest.raises(ValueError, match="render_width"):
            _config(embodiment="mimicgen", render_width=True)

    def test_a_fractional_width_is_no_longer_truncated(self):
        """``2.7`` rendered a 2x2 view at both read sites."""
        with pytest.raises(ValueError, match="render_width"):
            _config(embodiment="mimicgen", render_width=2.7)

    def test_a_numeric_string_is_no_longer_laundered_by_int(self):
        """``int("128")`` used to accept it; the shared domain admits only a real
        scalar, which is the rule its every other caller applies."""
        with pytest.raises(ValueError, match="render_width"):
            _config(embodiment="mimicgen", render_width="128")


# --------------------------------------------------------------------------- #
# Environment override
# --------------------------------------------------------------------------- #
class TestEnvironmentOverride:
    def test_a_zero_env_width_is_refused_rather_than_discarded(self, monkeypatch):
        """``VERA_RENDER_WIDTH=0`` is falsy, so the ``or`` this line used to
        carry dropped the override and applied the per-embodiment default in its
        place. It is now read with ``is not None``, matching ``vis_port``, so the
        caller who asked for 0 gets the refusal rather than 128."""
        monkeypatch.setenv("VERA_RENDER_WIDTH", "0")
        with pytest.raises(ValueError, match="render_width"):
            _config(embodiment="mimicgen")

    def test_a_negative_env_width_is_refused(self, monkeypatch):
        monkeypatch.setenv("VERA_RENDER_WIDTH", "-5")
        with pytest.raises(ValueError, match="render_width"):
            _config(embodiment="mimicgen")

    def test_a_usable_env_override_still_applies(self, monkeypatch):
        monkeypatch.setenv("VERA_RENDER_WIDTH", "96")
        cfg = _config(embodiment="mimicgen")
        assert cfg.render_width == 96
        assert type(cfg.render_width) is int

    def test_an_explicit_width_still_wins_over_the_environment(self, monkeypatch):
        """The env override applies only to an omitted field, as before."""
        monkeypatch.setenv("VERA_RENDER_WIDTH", "96")
        assert _config(embodiment="mimicgen", render_width=64).render_width == 64


# --------------------------------------------------------------------------- #
# Guard placement - nothing is built before the verdict
# --------------------------------------------------------------------------- #
class TestNothingIsBuiltBeforeTheVerdict:
    """The whole cost of the old placement was that the raise arrived after the
    server was up. Mirrors ``TestNothingIsBuiltBeforeTheVerdict`` in
    ``test_vera_port_domain.py``."""

    @pytest.fixture
    def no_side_effects(self, monkeypatch):
        """Make building a client or a server runner fatal.

        Yields the ``monkeypatch`` fixture so a test can also set the
        environment spelling of the width.
        """
        import strands_robots.policies.vera.provider as provider

        def _fatal(*args, **kwargs):
            raise AssertionError("a refused render_width reached a client / runner build")

        monkeypatch.setattr(provider, "VeraWebsocketClient", _fatal)
        monkeypatch.setattr(provider, "make_server_runner", _fatal)
        return monkeypatch

    def test_a_refused_env_width_builds_no_client_and_no_runner(self, no_side_effects):
        """``VeraPolicy`` builds its own config from the environment, and that
        construction is what raises - before either builder is reached."""
        no_side_effects.setenv("VERA_RENDER_WIDTH", "0")
        with pytest.raises(ValueError, match="render_width"):
            VeraPolicy(embodiment="mimicgen")

    def test_a_refused_prebuilt_config_never_reaches_the_policy(self, no_side_effects):
        """The other spelling: a config handed in as ``config=`` is refused in
        its own ``__post_init__``, so ``VeraPolicy.__init__`` never starts."""
        with pytest.raises(ValueError, match="render_width"):
            VeraPolicy(config=_config(embodiment="mimicgen", render_width=0))

    def test_the_fixture_is_not_vacuous(self, no_side_effects):
        """A usable width really does reach the patched builders."""
        with pytest.raises(AssertionError, match="refused render_width reached"):
            VeraPolicy(config=_config(embodiment="mimicgen", render_width=128))


# --------------------------------------------------------------------------- #
# Boundary - what this change deliberately leaves alone
# --------------------------------------------------------------------------- #
class TestNeighbouringWidthAxesStayOutOfScope:
    """Pins of current behaviour, not endorsements. Replace rather than delete
    when any of them is settled, per the premise-test guidance in ``AGENTS.md``.
    """

    def test_no_resource_ceiling_is_applied_to_the_width(self):
        """An outsized width is accepted, by the shared domain's stated policy.

        ``positive_whole_number_error`` documents that choosing a *resource*
        ceiling belongs to the consumer and is the per-backend decision tracked
        in #1871, so this guard deliberately does not invent one here.
        ``render_width=100000`` is a 30 GB allocation per view at the first
        resize; it passes the domain because it is a positive whole number.
        """
        assert _config(embodiment="mimicgen", render_width=100000).render_width == 100000

    def test_a_width_beyond_the_float_range_is_still_refused(self):
        """The one magnitude bound the domain does own, inherited not added."""
        with pytest.raises(ValueError, match="render_width"):
            _config(embodiment="mimicgen", render_width=10**400)

    def test_a_non_numeric_env_width_still_falls_back_to_the_default(self, monkeypatch):
        """``_env_int`` swallows a ``ValueError`` and returns ``None``, so a
        typo'd deploy variable selects the default instead of being refused.

        That is shared by every ``VERA_*`` int knob, so narrowing it is a change
        to ``_env_int`` rather than to this field, and it is out of scope here.
        Already pinned for the whole family by
        ``test_vera_unit.py::test_non_numeric_env_knobs_fall_back``; asserted
        again on this field so the two cannot drift apart silently.
        """
        monkeypatch.setenv("VERA_RENDER_WIDTH", "2.7")
        assert _config(embodiment="mimicgen").render_width == 128
        monkeypatch.setenv("VERA_RENDER_WIDTH", "abc")
        assert _config(embodiment="mimicgen").render_width == 128

    def test_the_height_is_still_implied_by_the_width(self):
        """Views are squared - ``_resize_frame`` resizes to ``(width, width)`` -
        so there is no separate height axis to hold to a domain. Recorded
        because the neighbouring media callers of this domain check ``width``
        and ``height`` as a pair, and this one legitimately does not.
        """
        req = _run(8)
        assert req["context_rgb"].shape[1:3] == (8, 8)

    def test_n_action_steps_is_gone_rather_than_unvalidated(self):
        """The neighbouring ``n_action_steps`` was documented as the deploy chunk
        size and read by nothing, so it was deleted rather than given a domain.

        This replaces the boundary pin that asserted the old behaviour
        (``n_action_steps=-7`` accepted and ignored). A domain here would have
        refused ``-7`` and then still honored nothing, which is a worse contract
        than an unvalidated knob, not a better one - so this axis is settled by
        removal. The full account, including the fourth spelling in the policy
        registry, lives in ``test_vera_n_action_steps_removed.py``; asserted here
        so the width domain cannot silently re-acquire an inert neighbour.
        """
        with pytest.raises(TypeError, match="n_action_steps"):
            _config(embodiment="mimicgen", n_action_steps=-7)
