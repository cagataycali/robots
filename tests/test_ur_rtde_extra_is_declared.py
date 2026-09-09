"""The UR driver's RTDE client is declared by an extra whose bound this project owns.

``strands_robots/drivers/ur.py`` drives a Universal Robots e-Series arm through
the controller's Real-Time Data Exchange interface, and the whole driver is an
adapter onto two importables: ``rtde_control`` (``servoJ``, ``speedJ``, the
dashboard verbs) and ``rtde_receive`` (joint positions and velocities, the TCP
pose, the safety and robot mode words). Both come from one distribution,
``ur_rtde`` - and no requirement named it. Measured against the checked-in lock,
it was absent from the base install and from every one of the 33 declared extras,
so no install of this project could move a UR arm.

What a reader got instead was ``pip install ur_rtde``: a bare distribution, the
last place in the drivers tree that handed a reader a package whose version this
project does not bound. The two remedies beside it name an extra
(``[crazyflie]``, ``[earthrover]``), which is the form that stays correct when the
requirement moves - the reader installs a capability and the manifest decides the
version.

So ``[ur]`` declares it at the range the driver's own calls need. It is
deliberately left out of ``[all]``: ur_rtde is a compiled Boost/pybind11 binding
whose wheels cover manylinux x86_64/i686 and win_amd64 only, up to cp312, so on a
Python 3.13 interpreter or an aarch64 host pip falls back to the sdist and needs a
C++ toolchain plus Boost. ``[all]`` is the bundle CI and exploration install, and
putting a compiler in front of it would cost every one of those installs. The
driver degrades cleanly without the SDK, so ``[all]`` still imports, registers and
tests it.

The last rule here is the general one rather than a fourth assertion about
``ur_rtde``: **no driver refusal may hand a reader a distribution when an extra
could supply it.** Two hints legitimately name a distribution and both say why -
``panda-py`` is not published on PyPI at all (the Franka binding ships as a
release asset, and a direct URL requirement is denylisted by
``tests/test_dependency_audit.py``), and ``booster_robotics_sdk_python`` is a
vendor wheel pinned to the robot's firmware, so the installed build's vocabulary
is an input rather than something this project bounds. Anything else is the defect
this file fixes, and the next native driver to land will be graded by it.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from strands_robots.drivers.ur import URDriver
from tests.uv_lock_closure import lock_closure

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_DRIVERS = _REPO_ROOT / "strands_robots" / "drivers"

#: The extra that owns the RTDE client, and the distribution it must declare.
_EXTRA = "ur"
_CLIENT = "ur-rtde"

#: A written install hint, and the target it hands the reader. Quoted forms are
#: read whole because the extra spelling needs them (a bare ``strands-robots[ur]``
#: is two shell words to zsh). A hint with no target - the prose "no ``pip
#: install`` supplies a module" - matches nothing, which is correct: it names no
#: package.
_HINT = re.compile(r"pip install\s+(?:'(?P<quoted>[^']+)'|\"(?P<dquoted>[^\"]+)\"|(?P<bare>[A-Za-z][A-Za-z0-9._-]*))")

#: Distributions a driver refusal may name directly, each with the reason no
#: extra can carry it. Not a waiver list: both reasons are properties of the
#: distribution, so a new entry has to state one.
_UNDECLARABLE = {
    "panda-py": (
        "not published on PyPI - the Franka binding ships as a release asset, and a direct "
        "URL requirement is denylisted by test_pyproject_has_no_direct_reference_dependency"
    ),
    "booster-robotics-sdk-python": (
        "a vendor wheel pinned to the robot's firmware, so the installed build's vocabulary is "
        "an input rather than a version this project bounds (strands_robots/drivers/booster.py)"
    ),
}


def _optional_dependencies() -> dict[str, list[str]]:
    with _PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]["optional-dependencies"]


def _requirements(extra: str) -> list[Requirement]:
    """Requirement objects *extra* declares, or a failure naming what is missing."""
    declared = _optional_dependencies()
    assert extra in declared, (
        f"the manifest declares no [{extra}] extra, so `pip install 'strands-robots[{extra}]'` "
        f"exits 0 and installs nothing. Declared: {sorted(declared)}"
    )
    return [Requirement(text) for text in declared[extra]]


def _declared_client() -> Requirement:
    """The ``[ur]`` requirement on the RTDE client, or a failure naming the extra."""
    declared = [req for req in _requirements(_EXTRA) if canonicalize_name(req.name) == _CLIENT]
    assert declared, (
        f"[{_EXTRA}] does not name {_CLIENT}, so the driver's own remedy installs an environment "
        f"where rtde_control and rtde_receive still cannot be imported. Declared: "
        f"{[str(req) for req in _requirements(_EXTRA)]}"
    )
    return declared[0]


def test_the_extra_declares_the_rtde_client_at_a_bounded_range() -> None:
    """``[ur]`` names ur_rtde, and states a range rather than "any release"."""
    specifier = SpecifierSet(str(_declared_client().specifier))
    assert specifier, (
        f"{_CLIENT} is declared unbounded in [{_EXTRA}], so an install can resolve a build with "
        "neither interface the driver calls"
    )
    assert any(clause.operator in (">=", "==", "~=") for clause in specifier), (
        f"{specifier} states no floor, so a release predating servoJ's signature satisfies it"
    )


def test_an_install_of_the_extra_resolves_the_client() -> None:
    """The lock agrees that ``[ur]`` ships the client - the manifest is not taken on trust."""
    closure = lock_closure(_EXTRA)
    assert _CLIENT in closure, (
        f"[{_EXTRA}] locks {len(closure)} distributions and {_CLIENT} is not among them, so an "
        "install of it ships the UR driver and no client for it"
    )
    assert _CLIENT not in lock_closure(None), (
        f"{_CLIENT} is in the base install, so the extra grades nothing - every install would "
        "carry a compiled binding nobody asked for"
    )


def test_the_bundle_puts_no_compiler_in_front_of_itself() -> None:
    """``[all]`` does not fold ``[ur]`` in, so the bundle stays wheel-only.

    ur_rtde publishes no aarch64 wheel and none for cp313, so on the hosts this
    project runs on an install of it builds from the sdist and needs a C++
    toolchain plus Boost. ``[all]`` is what CI and an exploring reader install;
    the driver reports a reason and stays usable without the SDK, so the capability
    is opt-in rather than a cost every one of those installs pays.
    """
    assert f"strands-robots[{_EXTRA}]" not in _optional_dependencies()["all"], (
        f"[all] folds in [{_EXTRA}], so every install of the bundle now needs a C++ toolchain on "
        "any host ur_rtde publishes no wheel for. If that is intended, this rule is the place to "
        "say so."
    )
    assert _CLIENT not in lock_closure("all"), (
        f"[all] resolves {_CLIENT} through some other requirement, so the bundle carries the "
        "compiled binding whether or not it names the extra"
    )


def test_the_absent_client_refuses_by_naming_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the SDK the driver reports the extra that supplies it, and stays usable.

    Drives the driver's own resolution rather than reading the source: a ``None``
    entry in ``sys.modules`` is what makes ``importlib.import_module`` raise
    ``ImportError`` for a name, which is the absence an install without the extra
    has.
    """
    monkeypatch.setitem(sys.modules, "rtde_control", None)
    monkeypatch.setitem(sys.modules, "rtde_receive", None)
    driver = URDriver(tool_name="ur5e", port="192.168.1.10")

    reason = driver.connect_eagerly()

    assert reason is not None, "connect_eagerly reported success with no RTDE client to speak"
    assert f"strands-robots[{_EXTRA}]" in reason, (
        "the refusal does not name the extra that supplies the client, so the reader is handed a "
        f"distribution whose bound this project does not own. Got: {reason!r}"
    )
    assert "pip install ur_rtde" not in reason, (
        f"the refusal still hands the reader the bare distribution. Got: {reason!r}"
    )
    # The degrade is the contract, not a side effect: no raise past the envelope.
    assert not driver.is_connected
    assert driver.send_action({"elbow_joint": 1.4})["status"] == "error"
    assert driver.state()["status"] == "error"


def test_no_driver_refusal_hands_a_reader_an_undeclarable_distribution() -> None:
    """Every written install hint in the drivers tree names an extra, or says why not.

    The general rule. An extra is a capability whose version this project owns, so
    a hint naming one keeps working when the requirement moves; a hint naming a
    distribution pins the reader to a decision the manifest never made.
    """
    extras = {re.sub(r"[-_.]+", "-", name).lower() for name in _optional_dependencies()}
    offenders: list[str] = []
    hints = 0
    modules: set[str] = set()
    for source in sorted(_DRIVERS.rglob("*.py")):
        rel = source.relative_to(_REPO_ROOT).as_posix()
        for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            for match in _HINT.finditer(line):
                target = match.group("quoted") or match.group("dquoted") or match.group("bare")
                hints += 1
                modules.add(rel)
                named = re.fullmatch(r"strands-robots(?:\[([a-z0-9,_-]+)\])?", target.strip())
                if named is None:
                    if canonicalize_name(target) not in _UNDECLARABLE:
                        offenders.append(f"{rel}:{lineno} names the distribution {target!r} -- {line.strip()[:90]}")
                    continue
                for part in (named.group(1) or "").split(","):
                    if part and re.sub(r"[-_.]+", "-", part).lower() not in extras:
                        offenders.append(f"{rel}:{lineno} names the undeclared extra [{part}]")

    # Non-vacuity: a sweep reading nothing would agree with every manifest.
    assert hints >= 6 and len(modules) >= 4, (
        f"the sweep found {hints} hints across {len(modules)} modules; the drivers tree or the hint pattern has drifted"
    )
    assert not offenders, (
        "these refusals hand a reader a distribution instead of an extra whose bound this project "
        f"owns. Declare it and name the extra, or record why no extra can carry it in "
        f"_UNDECLARABLE. Currently recorded: {sorted(_UNDECLARABLE)}\n" + "\n".join(offenders)
    )
