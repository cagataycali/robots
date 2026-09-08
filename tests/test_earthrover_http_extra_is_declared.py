"""The rover driver's HTTP transport is declared where an install reaches it.

``strands_robots/drivers/earthrover.py`` speaks HTTP to the vendor's
earth-rovers-sdk: ``GET /data`` is the telemetry every read returns and
``POST /control`` is what drives the rover. ``requests`` is that transport, so it
is a requirement of using the driver at all -- and no requirement named it. The
only bound the project stated lived in ``[tool.hatch.envs.default].dependencies``,
a hatch development environment that no install reaches, alongside three more
runtime distributions in the same shape.

What that left, measured against the checked-in lock: ``requests`` is absent from
the base install and from 26 of the 32 extras other than ``[all]``, and the six
that do resolve to it -- ``[lerobot]``, ``[lerobot-async]``, ``[molmoact2]``,
``[smolvla]``, ``[kimodo]`` and ``[cosmos3-diffusers]`` -- reach it as a
transitive of an unrelated machine learning stack (``lerobot`` itself, or
``diffusers``). So a rover install
(``strands-robots``, or ``[mesh,device-connect]`` for a fleet) shipped the driver
and none of its transport: ``connect_eagerly()`` returned "requests is not
installed", every read answered its empty cache, every write refused "not
connected", and the remedy printed was a bare ``pip install requests`` -- the only
place in the drivers tree that hands a reader a distribution rather than an extra
whose bound the project owns.

The same gap reached the tests. ``tests/drivers/test_earthrover_base_url_addresses_the_host.py``
opens ``pytest.importorskip("requests")`` and drives a real ``http.server`` to
measure *which host* the driver dials -- the property behind the userinfo and
``ws://`` refusals. Thirty-four cells sit behind that gate, and they ran because
the development environment pinned the distribution and because ``[all]`` happens
to carry ``diffusers``. Neither is a declaration, and the failure mode is quiet:
with requests absent the whole module collapses to one ``1 skipped`` line, which
reads the same as a suite that has nothing to say.

So ``[earthrover]`` declares it, ``[all]`` folds it in (pure Python, no build
step, and CI installs that bundle), the refusal names the extra the way
``[crazyflie]``'s does, and the development environment is left with development
tooling only. The last cell here is the general rule rather than the four names:
**every distribution that environment declares must be tooling the ``[dev]``
extra also declares.** A runtime distribution there is a second copy of a bound
that can drift from the one an install resolves, with ``hatch run test`` -- the
environment CI runs the suite in -- staying green over the drift.
``tests/test_dashboard_extra_is_declared.py`` pins that shape for the five
dashboard distributions after it shipped once; this generalizes it, because the
next re-pin will not be one of those five.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from strands_robots.drivers.earthrover import EarthRoverDriver
from tests.uv_lock_closure import lock_closure

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

#: The extra that owns the transport, and the bundle that folds it in. Both must
#: resolve to it: the first is the remedy the driver prints, the second is what
#: CI installs and therefore what decides whether the URL-property cells run.
_EXTRAS_SHIPPING_THE_TRANSPORT = ("earthrover", "all")


def _manifest() -> dict:
    with _PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]


def _requirements(extra: str) -> list[Requirement]:
    return [Requirement(text) for text in _manifest()["optional-dependencies"][extra]]


def _development_environment_dependencies() -> list[Requirement]:
    with _PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    declared = data["tool"]["hatch"]["envs"]["default"].get("dependencies", [])
    return [Requirement(text) for text in declared]


def test_the_extra_declares_the_http_transport() -> None:
    """``[earthrover]`` names requests, with the bound the project owns."""
    declared = [
        requirement for requirement in _requirements("earthrover") if canonicalize_name(requirement.name) == "requests"
    ]
    assert declared, (
        "the earthrover extra does not name requests, so the driver's own remedy "
        "installs an environment where GET /data and POST /control still cannot be "
        f"spoken. Declared: {[str(r) for r in _requirements('earthrover')]}"
    )
    specifiers = SpecifierSet(str(declared[0].specifier))
    assert specifiers, f"requests is declared unbounded ({declared[0]}), so no install resolves a graded version"


def test_the_bundle_folds_the_extra_in() -> None:
    """``[all]`` declares the transport instead of inheriting it.

    ``[all]`` already resolved to requests, through ``diffusers`` and ``lerobot``
    -- an accident of an unrelated stack, and the reason the driver's URL-property
    cells ran at all. Folding the extra in makes that a declaration, so dropping a
    machine learning dependency cannot silently turn thirty-four measurements into
    one skip.
    """
    assert "strands-robots[earthrover]" in _manifest()["optional-dependencies"]["all"], (
        "[all] does not fold in [earthrover], so the bundle CI installs supplies "
        "the rover transport only as long as some unrelated dependency keeps "
        "pulling requests"
    )


@pytest.mark.parametrize("extra", _EXTRAS_SHIPPING_THE_TRANSPORT)
def test_an_install_of_the_extra_locks_the_transport(extra: str) -> None:
    """Each extra that must ship the transport resolves to it in the lock."""
    closure = lock_closure(extra)
    assert "requests" in closure, (
        f"[{extra}] locks {len(closure)} distributions and requests is not among "
        "them, so an install of it ships the EarthRover driver and no transport for it"
    )


def test_the_installed_transport_satisfies_the_declared_bound() -> None:
    """The requests an install resolves is inside the bound the extra states."""
    requests = pytest.importorskip("requests")
    declared = [
        requirement for requirement in _requirements("earthrover") if canonicalize_name(requirement.name) == "requests"
    ]
    assert declared, "the earthrover extra declares no requests, so there is no bound to grade"
    specifiers = SpecifierSet(str(declared[0].specifier))
    assert Version(requests.__version__) in specifiers, (
        f"the installed requests {requests.__version__} is outside the declared "
        f"{specifiers}, so the bound describes a version no install resolves"
    )


def test_the_absent_transport_refuses_by_naming_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without requests the driver reports what supplies it, not a distribution.

    Drives the real gate rather than reading the source: a ``None`` entry in
    ``sys.modules`` is what makes ``import requests`` raise ``ImportError`` for an
    installed module, which is the absence an install without the extra has.
    """
    monkeypatch.setitem(sys.modules, "requests", None)
    driver = EarthRoverDriver(port="http://10.0.0.9:8001")

    reason = driver.connect_eagerly()

    assert reason is not None, "connect_eagerly reported success with no transport to speak"
    assert "strands-robots[earthrover]" in reason, (
        "the refusal does not name the extra that supplies requests, so the reader "
        f"is handed a distribution whose bound this project does not own. Got: {reason!r}"
    )
    # The degrade is the documented contract, not a side effect: the driver stays
    # usable and answers from its empty cache.
    assert driver.read_state() == {}, "a driver that could not connect must read as empty, not raise"


def test_the_development_environment_names_only_development_tooling() -> None:
    """A runtime distribution in the hatch environment is a bound no install reaches.

    The general rule, and why this file is not four assertions about four names.
    That environment sets ``features = ["all"]``, so everything the suite needs at
    runtime already arrives through a declared extra; a literal pin beside it is a
    second copy that can drift from the one an install resolves, with
    ``hatch run test`` green over the drift. Tooling is exempt because ``[dev]``
    declares it and no install is expected to carry it.
    """
    tooling = {str(canonicalize_name(requirement.name)) for requirement in _requirements("dev")}
    declared = _development_environment_dependencies()

    # Non-vacuity: an environment declaring nothing would agree with everything.
    assert len(declared) >= 5, f"the hatch environment declares only {[str(r) for r in declared]}; the scan has drifted"

    runtime = sorted(
        str(requirement) for requirement in declared if str(canonicalize_name(requirement.name)) not in tooling
    )
    assert not runtime, (
        f"{runtime} are declared in [tool.hatch.envs.default].dependencies, which no "
        "install reaches. Declare each in [project.dependencies] or in the extra "
        "whose capability needs it -- otherwise the bound the suite is exercised "
        f"against can drift from the bound an install resolves. Tooling: {sorted(tooling)}"
    )
