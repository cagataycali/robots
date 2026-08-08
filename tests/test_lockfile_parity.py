#!/usr/bin/env python3
"""``uv.lock`` must still resolve ``pyproject.toml``.

The lock is checked in and pins every version a locked install resolves, so it
reads as authoritative. Nothing compared it against the manifest: the manifest was
edited on three separate days without a relock, and CI stayed green because
nothing looked. The drift was not cosmetic -- the lock pinned ``lerobot 0.6.0``
against a declared ``>=0.6.1`` floor, and recorded no ``roslibpy`` at all while
``[rosbridge]`` declares it, so a locked install of that extra resolved without
the package ``use_rosbridge`` imports.

The dependency-audit suite next door is where this would have been caught, and it
shows why it was not: it names ``uv.lock`` twice in prose, as the artifact whose
regressions it protects ("it wedged ``uv lock``, freezing uv.lock at lerobot
0.5.1"), and every assertion in it reads only ``pyproject.toml``. Nothing in the
repository opened the lock.

These tests are the gate. They run in the required suite rather than as a separate
job, which is deliberate: ``call-test-lint / Test and Lint`` is the only required
status check, so a bespoke workflow would be advisory, and the comparison here is
offline, so it does not need a resolver or the network the way ``uv lock --check``
does.

Three groups:

* the live-tree gate -- the manifest and the lock must agree, and each pin must
  satisfy the specifier it was resolved for;
* the four repaired rows, asserted individually, so a future relock that drops one
  fails on the row rather than only on the aggregate;
* the encodings the comparison rests on, pinned as premises, plus planted defects
  proving the checker reports rather than merely passing.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_LOCK = _REPO_ROOT / "uv.lock"
_CHECK_PATH = _REPO_ROOT / "scripts" / "check_lockfile_parity.py"


def _load_check_module():
    spec = importlib.util.spec_from_file_location("check_lockfile_parity", _CHECK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["check_lockfile_parity"] = module
    spec.loader.exec_module(module)
    return module


parity = _load_check_module()


def _write_pair(
    tmp_path: Path,
    *,
    declared: dict[str, list[str]],
    recorded: list[dict[str, Any]],
    pins: dict[str, str | list[str]],
) -> tuple[Path, Path]:
    """Write a synthetic manifest/lock pair and return their paths.

    ``declared`` maps an extra name to its requirement strings (the key ``""`` is
    the core dependency list); ``recorded`` is what the lock transcribes into the
    root package's ``requires-dist``; ``pins`` is the version each package is
    locked at, or a list of versions when the case under test needs one name
    pinned in more than one resolution fork.
    """
    core = declared.get("", [])
    extras = {name: reqs for name, reqs in declared.items() if name}
    lines = ["[project]", 'name = "demo"', 'version = "0"', f"dependencies = {core!r}".replace("'", '"')]
    if extras:
        lines.append("[project.optional-dependencies]")
        for name, reqs in extras.items():
            lines.append(f"{name} = {reqs!r}".replace("'", '"'))
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def entry(item: dict[str, Any]) -> str:
        parts = [f'name = "{item["name"]}"']
        if item.get("extras"):
            rendered = ", ".join(f'"{e}"' for e in item["extras"])
            parts.append(f"extras = [{rendered}]")
        if item.get("specifier"):
            parts.append(f'specifier = "{item["specifier"]}"')
        if item.get("marker"):
            parts.append(f'marker = "{item["marker"]}"')
        return "{ " + ", ".join(parts) + " }"

    lock_lines = [
        "version = 1",
        'requires-python = ">=3.12"',
        "",
        "[[package]]",
        'name = "demo"',
        "",
        "[package.metadata]",
    ]
    lock_lines.append("requires-dist = [")
    lock_lines += [f"    {entry(item)}," for item in recorded]
    lock_lines.append("]")
    for name, versions in pins.items():
        for version in versions if isinstance(versions, list) else [versions]:
            lock_lines += ["", "[[package]]", f'name = "{name}"', f'version = "{version}"']
    lock = tmp_path / "uv.lock"
    lock.write_text("\n".join(lock_lines) + "\n", encoding="utf-8")
    return pyproject, lock


class TestTheLockResolvesTheManifest:
    """The live gate: a stale lock must fail here rather than ship silently."""

    def test_the_lock_records_the_live_manifest(self) -> None:
        """Every declared requirement must be recorded, and vice versa."""
        declared = parity.declared_requirements(_PYPROJECT)
        recorded = parity.recorded_requirements(_LOCK, "strands-robots")
        findings = parity.parity_findings(declared, recorded)
        assert findings == [], "uv.lock no longer resolves pyproject.toml; run `uv lock`:\n" + "\n".join(findings)

    def test_every_locked_pin_satisfies_its_declared_specifier(self) -> None:
        """A pin outside its own declared range defeats the range."""
        declared = parity.declared_requirements(_PYPROJECT)
        findings = parity.pin_findings(declared, parity.locked_versions(_LOCK))
        assert findings == [], "a locked pin violates its declared specifier:\n" + "\n".join(findings)

    def test_the_audit_reports_nothing_for_the_live_pair(self) -> None:
        """The reusable entry point agrees with the two checks run separately."""
        assert parity.audit(_PYPROJECT, _LOCK) == []


class TestTheRepairedRowsStayRepaired:
    """Each drifted dependency, pinned on its own so a relock cannot drop it."""

    def test_the_locked_lerobot_satisfies_the_bucket_streaming_floor(self) -> None:
        """The floor exists to guarantee bucket streaming; the pin must honour it."""
        pinned = parity.locked_versions(_LOCK)["lerobot"]
        assert len(pinned) == 1, f"lerobot is locked more than once: {pinned}"
        assert Version(pinned[0]) >= Version("0.6.1"), (
            f"uv.lock pins lerobot {pinned[0]}, which cannot serve stream_dataset(repo_type='bucket')"
        )

    def test_the_rosbridge_extra_locks_roslibpy(self) -> None:
        """use_rosbridge imports roslibpy, so [rosbridge] must resolve it."""
        recorded = parity.recorded_requirements(_LOCK, "strands-robots")
        names = {name for extra, name, *_ in recorded if extra == "rosbridge"}
        assert "roslibpy" in names, f"[rosbridge] locks {sorted(names)} without roslibpy"
        assert "roslibpy" in parity.locked_versions(_LOCK)

    @pytest.mark.parametrize("extra", ["sim-mujoco", "sim-newton"])
    def test_the_ik_solver_stack_is_locked(self, extra: str) -> None:
        """move_to needs mink and a qpsolvers backend, so both must be locked."""
        recorded = parity.recorded_requirements(_LOCK, "strands-robots")
        names = {name for entry_extra, name, *_ in recorded if entry_extra == extra}
        assert {"mink", "qpsolvers"} <= names, f"[{extra}] locks {sorted(names)}"

    def test_the_declared_hub_floor_is_locked(self) -> None:
        """The huggingface-hub floor was raised without a relock; it must hold."""
        declared = parity.declared_requirements(_PYPROJECT)
        specifiers = {SpecifierSet(",".join(key[3])) for key in declared if key[1] == "huggingface-hub" and key[3]}
        assert specifiers, "no huggingface-hub requirement is declared any more"
        pinned = parity.locked_versions(_LOCK)["huggingface-hub"]
        for specifier in specifiers:
            assert any(specifier.contains(v, prereleases=True) for v in pinned), (
                f"uv.lock pins huggingface-hub {pinned}, which does not satisfy {specifier}"
            )


class TestTheComparisonsPremises:
    """What the comparison assumes about the two file formats, measured.

    Each of these is inert today and documents when a modelled-but-unused path
    goes live: a requirement that grows an environment marker makes the residual
    comparison meaningful, and a name locked once in every fork would make the
    any-fork rule in :func:`pin_findings` unnecessary.
    """

    def test_no_declared_requirement_carries_an_environment_marker(self) -> None:
        """Today every marker is the extra alone, so the comparison is exact."""
        declared = parity.declared_requirements(_PYPROJECT)
        marked = {described for key, described in declared.items() if key[4]}
        assert marked == set(), f"these carry a residual marker, now compared: {sorted(marked)}"

    def test_every_recorded_marker_is_only_an_extra_clause(self) -> None:
        """uv records nothing but ``extra == '...'`` for this project today."""
        recorded = parity.recorded_requirements(_LOCK, "strands-robots")
        assert {key[4] for key in recorded} == {()}

    def test_a_forked_resolution_locks_some_name_more_than_once(self) -> None:
        """[tool.uv] conflicts is why a pin is satisfied by any fork, not all."""
        forked = {name: v for name, v in parity.locked_versions(_LOCK).items() if len(v) > 1}
        assert forked, "no name is locked twice; the any-fork rule is now untested by the live lock"

    def test_self_references_are_expanded_into_the_declared_set(self) -> None:
        """[all] inherits [rosbridge]'s requirements, exactly as uv records them."""
        declared = parity.declared_requirements(_PYPROJECT)
        assert len(declared) > 100, f"only {len(declared)} requirements reconstructed"
        inherited = {(key[0], key[1]) for key in declared}
        assert ("all", "roslibpy") in inherited, "the self-reference in [all] was not expanded"
        assert not any(key[1] == "strands-robots" for key in declared), "a self-reference leaked through"


class TestTheCheckerReportsPlantedDrift:
    """Planted defects, so an empty finding list means agreement, not blindness."""

    def test_a_matching_pair_reports_nothing(self, tmp_path: Path) -> None:
        pyproject, lock = _write_pair(
            tmp_path,
            declared={"": ["numpy>=1.24"], "extra": ["requests>=2.0"]},
            recorded=[
                {"name": "numpy", "specifier": ">=1.24"},
                {"name": "requests", "specifier": ">=2.0", "marker": "extra == 'extra'"},
            ],
            pins={"numpy": "1.26.0", "requests": "2.31.0"},
        )
        assert parity.audit(pyproject, lock) == []

    def test_a_raised_floor_the_lock_never_recorded_is_reported(self, tmp_path: Path) -> None:
        """The lerobot shape: the manifest raised a floor, the lock kept the old one."""
        pyproject, lock = _write_pair(
            tmp_path,
            declared={"": ["numpy>=1.26"]},
            recorded=[{"name": "numpy", "specifier": ">=1.24"}],
            pins={"numpy": "1.26.0"},
        )
        findings = parity.audit(pyproject, lock)
        assert any(f.startswith("NOT LOCKED") and ">=1.26" in f for f in findings), findings
        assert any(f.startswith("STALE LOCK") and ">=1.24" in f for f in findings), findings

    def test_a_requirement_absent_from_the_lock_is_reported(self, tmp_path: Path) -> None:
        """The roslibpy shape: an extra gained a dependency and was never relocked."""
        pyproject, lock = _write_pair(
            tmp_path,
            declared={"": ["numpy>=1.24"], "bridge": ["roslibpy>=1.7.0"]},
            recorded=[{"name": "numpy", "specifier": ">=1.24"}],
            pins={"numpy": "1.26.0"},
        )
        findings = parity.audit(pyproject, lock)
        assert any("NOT LOCKED [bridge] roslibpy>=1.7.0" in f for f in findings), findings

    def test_a_pin_below_the_declared_floor_is_reported(self, tmp_path: Path) -> None:
        """Parity can agree while the resolution is stale; the pin check catches it."""
        pyproject, lock = _write_pair(
            tmp_path,
            declared={"": ["numpy>=1.26"]},
            recorded=[{"name": "numpy", "specifier": ">=1.26"}],
            pins={"numpy": "1.24.0"},
        )
        findings = parity.audit(pyproject, lock)
        assert (
            parity.parity_findings(parity.declared_requirements(pyproject), parity.recorded_requirements(lock, "demo"))
            == []
        )
        assert any(f.startswith("PIN VIOLATES FLOOR") and "1.24.0" in f for f in findings), findings

    def test_a_pin_satisfied_by_one_fork_is_not_reported(self, tmp_path: Path) -> None:
        """A conflicting extra locks two versions; either one satisfying is enough."""
        pyproject, lock = _write_pair(
            tmp_path,
            declared={"": ["gymnasium>=1.1"]},
            recorded=[{"name": "gymnasium", "specifier": ">=1.1"}],
            pins={"gymnasium": ["0.29.1", "1.3.0"]},
        )
        assert parity.audit(pyproject, lock) == []

    def test_an_expanded_self_reference_is_not_reported_as_undeclared(self, tmp_path: Path) -> None:
        """uv records the closure, so the checker must expand rather than compare literally."""
        pyproject, lock = _write_pair(
            tmp_path,
            declared={"bridge": ["roslibpy>=1.7.0"], "all": ["demo[bridge]"]},
            recorded=[
                {"name": "roslibpy", "specifier": ">=1.7.0", "marker": "extra == 'bridge'"},
                {"name": "roslibpy", "specifier": ">=1.7.0", "marker": "extra == 'all'"},
            ],
            pins={"roslibpy": "1.8.1"},
        )
        assert parity.audit(pyproject, lock) == []


def test_the_scripts_entry_point_gates_on_the_finding_list() -> None:
    """The module is runnable as a check, so a contributor can verify before pushing."""
    source = _CHECK_PATH.read_text(encoding="utf-8")
    assert re.search(r"^\s*return 1$", source, re.M), "main() must report failure with a non-zero status"
    assert "uv lock" in source, "the remedy for a finding must name the command that repairs it"
