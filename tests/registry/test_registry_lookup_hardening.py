"""Registry lookups must not be brickable, silently misrouted, or name-blind.

Three read/write invariants of the robot + policy registry that previously
had gaps:

1. ``register_robot`` must reject an alias that collides with an existing
   canonical name (or another robot's alias) instead of persisting it with a
   warning. The loader RAISES on such collisions at every subsequent read, so a
   "successful" registration could otherwise brick every ``get_robot`` /
   ``resolve_name`` call process-wide until ``user_robots.json`` was hand-edited.
   Write-time validation must match the loader's read-time validation.
2. A ``vera://host:port`` policy URL must parse its host/port into kwargs. The
   ``^vera://`` pattern matched but had no parser branch, so
   ``create_policy("vera://gpu-box:9000")`` silently fell back to 127.0.0.1.
3. ``resolve_name`` must round-trip every canonical robot name, including the
   alias-less ones. The canonical check used ``alias_map.values()`` (only robots
   that declare an alias), so a normalized form like ``reachy-2`` -> ``reachy_2``
   never resolved to the alias-less canonical ``reachy2``.
"""

import json
from importlib.resources import files

import pytest

from strands_robots.registry.policies import resolve_policy
from strands_robots.registry.robots import get_robot, resolve_name
from strands_robots.registry.user_registry import register_robot


def _make_robot_asset(assets_dir, name="myarm", xml="myarm.xml"):
    """Create a minimal valid MJCF asset dir the registry will accept."""
    robot_dir = assets_dir / name
    robot_dir.mkdir(parents=True, exist_ok=True)
    (robot_dir / xml).write_text('<mujoco model="myarm"><worldbody/></mujoco>')
    return name, xml


class TestRegisterRobotFailsClosedOnAliasCollision:
    """A registration the loader would reject must be refused at write time."""

    def test_alias_colliding_with_canonical_name_raises(self, tmp_path):
        """register_robot(aliases=[<existing canonical>]) raises ValueError."""
        name, xml = _make_robot_asset(tmp_path / "assets")
        with pytest.raises(ValueError, match="so100"):
            register_robot(
                name=name,
                model_xml=xml,
                description="x",
                category="arm",
                joints=6,
                aliases=["so100"],
            )

    def test_registry_still_loads_after_rejected_registration(self, tmp_path):
        """A rejected registration must not be persisted; lookups keep working."""
        name, xml = _make_robot_asset(tmp_path / "assets")
        with pytest.raises(ValueError):
            register_robot(
                name=name,
                model_xml=xml,
                description="x",
                category="arm",
                joints=6,
                aliases=["so100"],
            )
        # The bricking bug: after a warn-and-save, get_robot itself would raise.
        assert get_robot("so100") is not None
        assert get_robot(name) is None  # never persisted

    def test_unique_alias_registration_succeeds(self, tmp_path):
        """A non-colliding alias still registers and resolves normally."""
        name, xml = _make_robot_asset(tmp_path / "assets")
        register_robot(
            name=name,
            model_xml=xml,
            description="x",
            category="arm",
            joints=6,
            aliases=["myleader"],
        )
        assert resolve_name("myleader") == name
        assert get_robot("myleader") is not None

    def test_reregistration_with_own_alias_and_overwrite_succeeds(self, tmp_path):
        """Re-registering the same robot with overwrite must not self-collide."""
        name, xml = _make_robot_asset(tmp_path / "assets")
        register_robot(
            name=name,
            model_xml=xml,
            description="v1",
            category="arm",
            joints=6,
            aliases=["myleader"],
            overwrite=True,
        )
        register_robot(
            name=name,
            model_xml=xml,
            description="v2",
            category="arm",
            joints=6,
            aliases=["myleader"],
            overwrite=True,
        )
        assert get_robot("myleader") is not None
        assert get_robot("so100") is not None


class TestVeraUrlParsing:
    """vera://host[:port] must populate connection kwargs, not fall to 127.0.0.1."""

    def test_vera_url_parses_host_and_server_port(self):
        """create_policy('vera://gpu-box:9000') targets gpu-box:9000."""
        provider, kwargs = resolve_policy("vera://gpu-box:9000")
        assert provider == "vera"
        assert kwargs["host"] == "gpu-box"
        assert kwargs["server_port"] == 9000

    def test_vera_url_without_port_leaves_embodiment_default(self):
        """Omitting the port keeps host but leaves server_port to the default."""
        provider, kwargs = resolve_policy("vera://gpu-box")
        assert provider == "vera"
        assert kwargs["host"] == "gpu-box"
        assert "server_port" not in kwargs


class TestCanonicalNameRoundTrip:
    """Every canonical robot name (alias-less included) must resolve to itself."""

    def _canonical_names(self):
        data = files("strands_robots.registry").joinpath("robots.json").read_text(encoding="utf-8")
        return list(json.loads(data)["robots"])

    def test_every_canonical_name_round_trips(self):
        """resolve_name(canonical) == canonical for all robots in robots.json."""
        for canonical in self._canonical_names():
            assert resolve_name(canonical) == canonical

    def test_alias_less_robot_resolves_from_normalized_form(self):
        """A hyphenated form of an alias-less robot resolves to its canonical."""
        # reachy2 is alias-less; "reachy-2" normalizes to "reachy_2" which only
        # matches after the underscore-stripping fallback checks canonical names.
        assert resolve_name("reachy-2") == "reachy2"
        assert get_robot("reachy-2") is not None
