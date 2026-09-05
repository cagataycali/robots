"""The VERA ports are one shared TCP-port domain, applied at the config funnel.

A ``server_port`` reaches three consumers under three different coercions - the
provider dials ``int(server_port or 0)``, :attr:`VeraConfig.server_uri`
interpolates the field verbatim, and the runner's argv carries
``str(server_port)`` - so an unusable value is not merely refused late, it is
*applied* as three different ports. ``server_port=2.7`` dialed ``ws://host:2``
while launching ``--port 2.7`` and reporting ``ws://host:2.7``, so the client
could not reach the server it had just started; ``True`` produced the
non-URI ``ws://host:True``. Values outside the coercions' domain
(``nan`` / ``inf`` / a list) escaped the constructor as a bare
``ValueError`` / ``OverflowError`` / ``TypeError`` naming neither the field nor
the class.

These tests pin the domain (:func:`strands_robots.utils.tcp_port_error`, the
same rule the four other port-dialing providers use), the one documented
asymmetry (``vis_port=0`` disables the viewer, so zero is a mode selector there
and not a port), and the property the divergence was: every accepted port is
named identically by all three consumers.

They also pin that a port means the same thing however it was spelled. A field
can be set as a keyword or through ``VERA_*_PORT``, and the two must reach the
one check: ``VeraConfig(server_port=0)`` was refused while
``VERA_SERVER_PORT=0`` reported success on the per-embodiment default, because
the override was read for its truth (``or``) rather than for its presence, and
``0`` is falsy. A deploy that asked for a port it cannot have was told it had
got it, on a different port.

Everything here is offline - no server, no socket, no ``vera`` package.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from strands_robots.policies.vera import VeraConfig, VeraPolicy, VeraServerRunner
from strands_robots.policies.vera.config import _viewer_port_error

# Values no TCP port can be addressed by. ``0`` is in this set for
# ``server_port`` because the client has no way to learn which ephemeral port
# the kernel handed the server, so it cannot dial one.
UNUSABLE_PORTS: list[Any] = [
    0,
    -1,
    65536,
    70000,
    2.7,
    8800.0,
    True,
    False,
    math.nan,
    math.inf,
    "8800",
    [8800],
    {},
]

# Ports every consumer can honor.
USABLE_PORTS: list[int] = [1, 8800, 8820, 65535]

# Env spellings of a ``server_port`` no consumer can honor. Every entry is one
# ``_env_int`` really parses, so each reaches the shared domain rather than
# stopping at the parser - a spelling the parser cannot read is swallowed by
# design, pinned in
# ``test_vera_unit.py::test_malformed_numeric_env_degrades_to_defaults``.
# ``"0"`` and ``"-0"`` are the rows that need the override read for its
# presence: they parse to a falsy int, which an ``or`` discards.
REFUSED_ENV_SERVER_PORTS: list[str] = ["0", "-0", "-1", "65536", "70000"]

# The same, for ``vis_port``, minus the documented zero exemption.
REFUSED_ENV_VIEWER_PORTS: list[str] = ["-9", "-1", "65536", "70000"]


def _config(**kwargs: Any) -> VeraConfig:
    """Build a config through the funnel, splatted so off-type values reach it.

    mypy does not narrow a ``**dict[str, Any]`` splat, which is what lets a test
    hand a deliberately wrong type to a field annotated ``int | None``.
    """
    return VeraConfig(**kwargs)


def _argv_port(cfg: VeraConfig, flag: str) -> str | None:
    """The value the server launch argv carries for ``flag``, or None if absent."""
    cmd = VeraServerRunner(cfg)._build_command()
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def _server_port_outcome(**kwargs: Any) -> tuple[str, int | None]:
    """What the funnel did with a ``server_port``, comparable across spellings.

    Either ``("refused", None)`` or ``("resolved", port)``, so one value set as a
    keyword and the same value set through ``VERA_SERVER_PORT`` can be compared
    for the same verdict rather than for the same exception text.
    """
    try:
        return ("resolved", _config(embodiment="pusht", **kwargs).server_port)
    except ValueError:
        return ("refused", None)


# --------------------------------------------------------------------------- #
# server_port - refused at the funnel
# --------------------------------------------------------------------------- #
class TestServerPortDomain:
    @pytest.mark.parametrize("value", UNUSABLE_PORTS)
    def test_an_unusable_server_port_is_refused_by_name(self, value):
        """The refusal names the field, the value and the accepted range."""
        with pytest.raises(ValueError, match="server_port"):
            _config(embodiment="pusht", server_port=value)

    @pytest.mark.parametrize("value", UNUSABLE_PORTS)
    def test_the_refusal_replaces_a_bare_coercion_error(self, value):
        """No ``int()`` / interpolation failure escapes instead of the verdict."""
        try:
            _config(embodiment="pusht", server_port=value)
        except ValueError as exc:
            text = str(exc)
            assert "VeraConfig" in text
            assert "expected 1-65535" in text
            assert "convert float" not in text
            assert "int() argument" not in text
        else:
            pytest.fail(f"server_port={value!r} was accepted")

    @pytest.mark.parametrize("value", UNUSABLE_PORTS)
    def test_the_provider_keyword_reaches_the_same_verdict(self, value):
        """``VeraPolicy(server_port=...)`` funnels through the same guard."""
        with pytest.raises(ValueError, match="server_port"):
            VeraPolicy(embodiment="pusht", server_port=value, auto_launch_server=False)

    @pytest.mark.parametrize("value", UNUSABLE_PORTS)
    def test_a_prebuilt_config_cannot_carry_one_past_the_provider(self, value):
        """The guard is on the config, so ``config=`` is not a way around it."""
        with pytest.raises(ValueError, match="server_port"):
            VeraPolicy(config=_config(embodiment="pusht", server_port=value))

    @pytest.mark.parametrize("value", USABLE_PORTS)
    def test_a_usable_server_port_is_applied(self, value):
        """Over-reach control: the accepted side of the range still works."""
        cfg = _config(embodiment="pusht", server_port=value)
        assert cfg.server_port == value

    def test_none_still_applies_the_per_embodiment_default(self):
        """``None`` is the documented "use the default" spelling, not a port."""
        assert _config(embodiment="pusht").server_port == 8820
        assert _config(embodiment="mimicgen").server_port == 8800


# --------------------------------------------------------------------------- #
# The three consumers agree - the property the divergence was
# --------------------------------------------------------------------------- #
class TestEveryConsumerNamesTheSamePort:
    @pytest.mark.parametrize("value", USABLE_PORTS)
    def test_client_uri_server_uri_and_argv_name_one_port(self, value):
        """``2.7`` dialed :2, reported ws://host:2.7 and launched --port 2.7."""
        policy = VeraPolicy(embodiment="pusht", server_port=value, auto_launch_server=False)
        assert policy._client.uri == f"ws://127.0.0.1:{value}"
        assert policy.config.server_uri == f"ws://127.0.0.1:{value}"
        assert _argv_port(policy.config, "--port") == str(value)

    def test_the_default_port_is_named_identically_too(self):
        """The un-supplied path is held to the same agreement."""
        policy = VeraPolicy(embodiment="pusht", auto_launch_server=False)
        assert policy._client.uri == "ws://127.0.0.1:8820"
        assert policy.config.server_uri == "ws://127.0.0.1:8820"
        assert _argv_port(policy.config, "--port") == "8820"


# --------------------------------------------------------------------------- #
# vis_port - the same range, with zero as a documented mode selector
# --------------------------------------------------------------------------- #
class TestViewerPortDomain:
    def test_zero_disables_the_viewer_rather_than_naming_a_port(self):
        """Documented: ``0`` omits ``--vis-port``, so it is not refused."""
        cfg = _config(embodiment="pusht", vis_port=0)
        assert cfg.vis_port == 0
        assert _argv_port(cfg, "--vis-port") is None

    def test_none_applies_the_default_and_enables_the_viewer(self):
        """``None`` resolves to the per-embodiment default, it does not disable."""
        cfg = _config(embodiment="pusht")
        assert cfg.vis_port == 8821
        assert _argv_port(cfg, "--vis-port") == "8821"

    @pytest.mark.parametrize("value", USABLE_PORTS)
    def test_a_usable_viewer_port_reaches_the_argv(self, value):
        cfg = _config(embodiment="pusht", vis_port=value)
        assert _argv_port(cfg, "--vis-port") == str(value)

    @pytest.mark.parametrize("value", [v for v in UNUSABLE_PORTS if v != 0 and v is not False])
    def test_an_unusable_viewer_port_is_refused_by_name(self, value):
        with pytest.raises(ValueError, match="vis_port"):
            _config(embodiment="pusht", vis_port=value)

    @pytest.mark.parametrize("value", [True, False])
    def test_a_boolean_viewer_port_is_refused_rather_than_read_as_the_zero(self, value):
        """``False == 0``, so a bare zero test would read it as "disable"."""
        with pytest.raises(ValueError, match="vis_port"):
            _config(embodiment="pusht", vis_port=value)

    def test_the_zero_exemption_is_the_only_difference_from_the_shared_rule(self):
        """Unit: the wrapper decides the floor and defers everything else."""
        assert _viewer_port_error(0, "vis_port", "VeraConfig") is None
        for value in UNUSABLE_PORTS:
            if isinstance(value, int) and not isinstance(value, bool) and value == 0:
                continue
            assert _viewer_port_error(value, "vis_port", "VeraConfig") is not None


# --------------------------------------------------------------------------- #
# The environment override goes through the same guard
# --------------------------------------------------------------------------- #
class TestEnvironmentOverride:
    @pytest.mark.parametrize("raw", REFUSED_ENV_SERVER_PORTS)
    def test_an_unusable_env_server_port_is_refused(self, monkeypatch, raw):
        """``VERA_SERVER_PORT`` writes the same field, so it is checked too.

        Only the truthy rows were reachable while the override was read with
        ``or``: ``"0"`` parsed to a falsy int, so the override was dropped and
        the per-embodiment default (8820 on pusht) applied under a success.
        """
        monkeypatch.setenv("VERA_SERVER_PORT", raw)
        with pytest.raises(ValueError, match="server_port"):
            _config(embodiment="pusht")

    @pytest.mark.parametrize("raw", REFUSED_ENV_VIEWER_PORTS)
    def test_an_unusable_env_viewer_port_is_refused(self, monkeypatch, raw):
        monkeypatch.setenv("VERA_VIS_PORT", raw)
        with pytest.raises(ValueError, match="vis_port"):
            _config(embodiment="pusht")

    def test_a_zero_env_viewer_port_disables_the_viewer_as_the_keyword_does(self, monkeypatch):
        """The zero exemption is a property of the value, not of the spelling.

        ``0`` selects "no viewer" through the environment too, and the runner
        omits the flag exactly as it does for ``vis_port=0`` - which is what
        reading the override for its presence preserves. Read for its truth,
        ``VERA_VIS_PORT=0`` would resolve to the default port 8821 and the
        viewer the caller switched off would be served.
        """
        monkeypatch.setenv("VERA_VIS_PORT", "0")
        cfg = _config(embodiment="pusht")
        assert cfg.vis_port == 0
        assert _argv_port(cfg, "--vis-port") is None

    @pytest.mark.parametrize("raw", ["0", "-1", "65536", "70000", "8899"])
    def test_one_port_gets_one_verdict_whichever_spelling_named_it(self, monkeypatch, raw):
        """A value the keyword refuses cannot be accepted from the environment.

        The two spellings write one field through one check, so the outcome is a
        property of the value: refused for the same reason, or resolved to the
        same port. ``"0"`` is the row that separated them - the keyword was
        refused by name while the environment reported success on 8820.
        """
        value = int(raw)
        monkeypatch.delenv("VERA_SERVER_PORT", raising=False)
        keyword = _server_port_outcome(server_port=value)
        monkeypatch.setenv("VERA_SERVER_PORT", raw)
        environment = _server_port_outcome()
        assert keyword == environment

    def test_a_usable_env_override_still_applies(self, monkeypatch):
        monkeypatch.setenv("VERA_SERVER_PORT", "9999")
        monkeypatch.setenv("VERA_VIS_PORT", "9998")
        cfg = _config(embodiment="pusht")
        assert (cfg.server_port, cfg.vis_port) == (9999, 9998)


# --------------------------------------------------------------------------- #
# Guard placement - nothing is built before the verdict
# --------------------------------------------------------------------------- #
class TestNothingIsBuiltBeforeTheVerdict:
    @pytest.fixture
    def no_side_effects(self, monkeypatch):
        """Make building a client or a server runner fatal."""
        import strands_robots.policies.vera.provider as provider

        def _fatal(*args, **kwargs):
            raise AssertionError("a refused port reached a client / runner build")

        monkeypatch.setattr(provider, "VeraWebsocketClient", _fatal)
        monkeypatch.setattr(provider, "make_server_runner", _fatal)

    def test_a_refused_port_builds_no_client_and_no_runner(self, no_side_effects):
        """The config raises before the provider dials or launches anything."""
        with pytest.raises(ValueError, match="server_port"):
            VeraPolicy(embodiment="pusht", server_port=70000)

    def test_a_refused_viewer_port_builds_no_client_and_no_runner(self, no_side_effects):
        with pytest.raises(ValueError, match="vis_port"):
            VeraPolicy(embodiment="pusht", vis_port=-1)

    def test_the_fixture_is_not_vacuous(self, no_side_effects):
        """A usable port really does reach the patched builders."""
        with pytest.raises(AssertionError, match="refused port reached"):
            VeraPolicy(embodiment="pusht", server_port=8820)
