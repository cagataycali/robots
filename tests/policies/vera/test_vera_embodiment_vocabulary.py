"""``VeraConfig.embodiment`` is the key every other per-embodiment default is looked up by.

Every other unusable value on this dataclass is refused by name at the config
funnel - both ports, ``render_width``, ``motion_plan_scale`` and
``server_ready_timeout`` each carry a domain and a written rationale for holding
it there. ``embodiment`` carried none, even though it is not one knob among
those: it is the field they are *selected by*. Six readers consume it, and each
does something different with a name no table has an entry for::

    _DEFAULT_PORTS.get(self.embodiment, (8800, 8801))      # bare-literal fallback
    _DEFAULT_RENDER_WIDTH.get(self.embodiment, 128)        # bare-literal fallback
    f"VERA_{self.embodiment.upper()}_CKPT_ROOT"            # probed variable name
    f"vera-server-{self.embodiment}"                       # container name
    ["--embodiment", str(cfg.embodiment)]                  # subprocess argv
    ["-e", f"VERA_EMBODIMENT={cfg.embodiment}"]            # container env

The two fallbacks are what make this silent rather than merely wrong.
``(8800, 8801)`` and ``128`` are byte-for-byte ``mimicgen``'s entries in the two
tables they are fallbacks for, so an unrecognised spelling does not resolve to a
degraded configuration - it resolves to one that cannot be told from a
deliberate ``embodiment="mimicgen"``.

Measured on ``6ce7facc0``, one ``VeraConfig(embodiment=X)`` per row, no ``VERA_*``
in the environment, no ``vera`` package, no server:

| ``embodiment=`` | ports | width | same triple as |
| --- | --- | --- | --- |
| ``"pusht"``     | 8820 / 8821 | 252 | itself (declared) |
| ``"mimicgen"``  | 8800 / 8801 | 128 | itself (declared) |
| ``"allegro"``   | 8802 / 8803 | 128 | itself (declared) |
| ``"droid"``     | 8804 / 8805 | 128 | itself (declared) |
| ``"PushT"``     | 8800 / 8801 | 128 | **mimicgen** |
| ``"pushT"``     | 8800 / 8801 | 128 | **mimicgen** |
| ``"pusht "``    | 8800 / 8801 | 128 | **mimicgen** |
| ``"push_t"``    | 8800 / 8801 | 128 | **mimicgen** |
| ``"Mimicgen"``  | 8800 / 8801 | 128 | **mimicgen** |
| ``"mimicgen2"`` | 8800 / 8801 | 128 | **mimicgen** |
| ``"franka"``    | 8800 / 8801 | 128 | **mimicgen** |
| ``""``          | 8800 / 8801 | 128 | **mimicgen** |

The consequence is not a failed launch. ``VeraServerRunner.start`` opens with a
port probe and returns early on a hit - "Already serving (ours or someone
else's) - reuse it" - so a mistyped ``pusht`` dialed 8800, found a running
mimicgen server, completed the metadata handshake and rolled out against the
wrong embodiment's planner/IDM pair, reporting success throughout. The
``--embodiment`` flag that would have carried the typo to a server that could
object was never used, because no server was launched.

Nor does the container's refusal cover it. ``docker/entrypoint.sh`` ends its
per-embodiment ``case`` with ``ERROR: unknown embodiment`` / ``exit 2`` over the
same four names, but that arm is reached only under ``server_mode="docker"``,
only once an image has started, and never at all for the subprocess runner or
for ``auto_launch_server=False``. The vocabulary was stated in shell and
enforced in shell; the Python that computes the ports, the width and the
``-e VERA_EMBODIMENT=`` value it passes in did not hold it.

With the vocabulary held at the funnel the two tables become the single
statement of what each embodiment defaults to, so both lookups index them
directly: the second copy of a default is what made "not a known embodiment"
and "mimicgen" the same request.

Everything here is offline - no server, no socket, no ``vera`` package, no GPU.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest

from strands_robots.policies.vera.config import (
    _DEFAULT_PORTS,
    _DEFAULT_RENDER_WIDTH,
    Embodiment,
    VeraConfig,
)

VOCABULARY = get_args(Embodiment)

# Spellings a caller plausibly reaches for that no table has an entry for: case
# variants of a real name, a separator variant, a trailing space, a
# near-miss, an unrelated robot name, and the empty string.
UNKNOWN_SPELLINGS = (
    "PushT",
    "pushT",
    "pusht ",
    "push_t",
    "Mimicgen",
    "mimicgen2",
    "franka",
    "",
)

# Non-string values the field could hold from a keyword. ``None`` is included
# because the field has a default rather than being optional: passing it
# explicitly is a request for an embodiment, not for the default.
UNKNOWN_TYPES = (None, 0, 1, True, 2.5, ["pusht"], {"pusht": 1})


@pytest.fixture(autouse=True)
def _no_vera_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every field from the keyword and the tables, never the environment."""
    for name in (
        "VERA_SERVER_PORT",
        "VERA_VIS_PORT",
        "VERA_RENDER_WIDTH",
        "VERA_SERVER_READY_TIMEOUT",
        "VERA_MOTION_PLAN_SCALE",
        "VERA_DOCKER_CONTAINER",
        "VERA_CKPT_ROOT",
        "VERA_SERVER_MODE",
    ):
        monkeypatch.delenv(name, raising=False)


def resolved(embodiment: str) -> tuple[int | None, int | None, int | None]:
    """The triple that decides which server is dialed and what is sent to it."""
    cfg = VeraConfig(embodiment=embodiment)  # type: ignore[arg-type]
    return cfg.server_port, cfg.vis_port, cfg.render_width


class TestAnUnknownEmbodimentIsRefused:
    """A spelling no table has an entry for cannot resolve to a configuration."""

    @pytest.mark.parametrize("spelling", UNKNOWN_SPELLINGS)
    def test_unknown_spelling_raises_naming_field_and_vocabulary(self, spelling: str) -> None:
        """The message names the field, the value and every accepted name."""
        with pytest.raises(ValueError) as excinfo:
            VeraConfig(embodiment=spelling)  # type: ignore[arg-type]
        message = str(excinfo.value)
        assert "embodiment" in message
        assert repr(spelling) in message
        for name in VOCABULARY:
            assert repr(name) in message, f"vocabulary member {name!r} missing from {message!r}"

    @pytest.mark.parametrize("value", UNKNOWN_TYPES)
    def test_non_string_value_raises_rather_than_resolving(self, value: object) -> None:
        """A value that is not a name at all is refused, not looked up."""
        with pytest.raises(ValueError, match="embodiment"):
            VeraConfig(embodiment=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("spelling", UNKNOWN_SPELLINGS)
    def test_unknown_spelling_no_longer_resolves_to_mimicgen(self, spelling: str) -> None:
        """The property that made the typo silent: it is refused, not aliased.

        Pre-fix every row of this parametrization returned ``mimicgen``'s
        triple, because the two lookups' fallbacks were ``mimicgen``'s entries.
        """
        with pytest.raises(ValueError):
            resolved(spelling)

    def test_the_refusal_reaches_the_provider_keyword(self) -> None:
        """``VeraPolicy(embodiment=...)`` funnels through the same check.

        The provider declares the field as a plain ``str`` and forwards it, so
        the config is the one place the vocabulary can be held for both the
        keyword and a pre-built config handed to it.
        """
        provider = pytest.importorskip("strands_robots.policies.vera.provider")
        with pytest.raises(ValueError, match="embodiment"):
            provider.VeraPolicy(embodiment="PushT", auto_launch_server=False)


class TestDeclaredEmbodimentsResolveFromTheirOwnTable:
    """Controls: every accepted name still gets its own entry, not a fallback."""

    @pytest.mark.parametrize("embodiment", VOCABULARY)
    def test_ports_and_width_come_from_the_owner_tables(self, embodiment: str) -> None:
        """Removing the fallbacks did not change what a known name resolves to."""
        expected_ports = _DEFAULT_PORTS[embodiment]
        expected_width = _DEFAULT_RENDER_WIDTH[embodiment]
        assert resolved(embodiment) == (*expected_ports, expected_width)

    def test_pusht_keeps_the_entry_the_fallback_used_to_hide(self) -> None:
        """``pusht`` is the one name whose entry differs from the old fallbacks.

        It is therefore the row that proves the tables, not the literals, are
        being read: ``(8820, 8821, 252)`` against the old ``(8800, 8801, 128)``.
        """
        assert resolved("pusht") == (8820, 8821, 252)

    @pytest.mark.parametrize("embodiment", VOCABULARY)
    def test_an_explicit_value_still_wins_over_the_table(self, embodiment: str) -> None:
        """The gate refuses names, it does not start overriding supplied values."""
        cfg = VeraConfig(embodiment=embodiment, server_port=9100, vis_port=0, render_width=64)  # type: ignore[arg-type]
        assert (cfg.server_port, cfg.vis_port, cfg.render_width) == (9100, 0, 64)


class TestEveryEmbodimentHasAnEntryInBothTables:
    """Indexing the tables directly is only safe while they cover the vocabulary."""

    @pytest.mark.parametrize(
        ("table", "name"),
        [(_DEFAULT_PORTS, "_DEFAULT_PORTS"), (_DEFAULT_RENDER_WIDTH, "_DEFAULT_RENDER_WIDTH")],
    )
    def test_table_keys_are_exactly_the_vocabulary(self, table: dict, name: str) -> None:
        """A fifth embodiment cannot arrive without its defaults, or vice versa."""
        assert set(table) == set(VOCABULARY), f"{name} does not cover {Embodiment}"

    def test_the_two_tables_agree_on_which_names_exist(self) -> None:
        """One table gaining a name without the other is the drift to catch."""
        assert set(_DEFAULT_PORTS) == set(_DEFAULT_RENDER_WIDTH)

    def test_every_declared_port_pair_is_distinct(self) -> None:
        """Two embodiments sharing a port is what let a typo attach to a server.

        The defaults must keep each embodiment on its own port, so a config
        resolved for one name can never dial the server another launched.
        """
        pairs = list(_DEFAULT_PORTS.values())
        assert len(set(pairs)) == len(pairs), f"duplicate default ports: {_DEFAULT_PORTS}"
        every_port = [port for pair in pairs for port in pair]
        assert len(set(every_port)) == len(every_port), f"a port is reused: {_DEFAULT_PORTS}"


class TestTheContainerAndTheConfigRefuseTheSameNames:
    """The vocabulary is stated twice - in Python and in shell - and must agree.

    ``docker/entrypoint.sh`` dispatches on ``VERA_EMBODIMENT``, which the docker
    runner sets from this field. Two enforcers of one vocabulary is exactly the
    shape that drifts, so the shell ``case`` is read here rather than trusted.
    """

    ENTRYPOINT = Path(__file__).resolve().parents[3] / "strands_robots/policies/vera/docker/entrypoint.sh"

    def _case_arms(self) -> set[str]:
        """Embodiment names the entrypoint's ``case`` has an arm for."""
        script = self.ENTRYPOINT.read_text(encoding="utf-8")
        block = script.split('case "${EMBODIMENT}" in', 1)
        assert len(block) == 2, "entrypoint.sh no longer dispatches on ${EMBODIMENT}"
        body = block[1].split("esac", 1)[0]
        names: set[str] = set()
        for line in body.splitlines():
            match = re.match(r"\s*([a-z0-9_|]+)\)\s*$", line)
            if match:
                names.update(match.group(1).split("|"))
        return names

    def test_the_entrypoint_exists_where_the_runner_expects_it(self) -> None:
        """The cross-language pin is only meaningful while the script is there."""
        assert self.ENTRYPOINT.is_file(), self.ENTRYPOINT

    def test_the_shell_case_covers_exactly_the_python_vocabulary(self) -> None:
        """Neither side may gain or lose a name alone."""
        assert self._case_arms() == set(VOCABULARY)

    def test_the_entrypoint_still_refuses_an_unmatched_name(self) -> None:
        """The config's refusal complements the container's; it does not replace it."""
        script = self.ENTRYPOINT.read_text(encoding="utf-8")
        assert "unknown embodiment" in script
        assert "exit 2" in script
