"""A locked C++ extension of torch resolves to one version, because torch does.

``torchvision`` and ``torchcodec`` both ship compiled libraries linked against
libtorch, and both are unusable when paired with a torch whose ABI they were not
built for. Only one of them says so: ``torchvision``'s lock entry declares
``torch`` as a dependency, so the resolver keeps the pair consistent by itself.
``torchcodec``'s entry declares nothing at all - no ``Requires-Dist: torch``
exists in any of its wheels - so its version is decided purely by the version
window the manifest allows, independently of the torch that is locked beside it.

That is how ``uv.lock`` came to record ``torchcodec`` twice, 0.11.1 on
linux-aarch64 and 0.10.0 on macOS-arm64 / linux-x86_64 / win32, next to a single
``torch`` 2.11.0. The pairing is not merely untidy: 0.10.0's
``libtorchcodec_custom_ops*.so`` imports
``_ZN3c1013MessageLoggerC1EPKciib`` - ``c10::MessageLogger::MessageLogger(char
const*, int, bool)`` - which libc10 exports at torch 2.10 and no longer exports
at 2.11 (2.11 takes a ``c10::SourceLocation``), so importing
``torchcodec._core.ops`` raised ``undefined symbol`` and every dataset open fell
back to the PyAV decoder.

Nothing in the resolution can catch that: ``uv lock --check`` passes on such a
lock, because a resolution inside the declared bounds is what it grades. So the
rule is pinned here instead - a distribution that links libtorch without
declaring it must carry as many locked versions as ``torch`` itself, which is
one.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_LOCK = Path(__file__).resolve().parents[1] / "uv.lock"

#: Distributions that ship compiled libraries linked against libtorch. A
#: mismatched pair fails at import (a missing C++ symbol), never at resolution,
#: so each one's locked version set is graded against torch's.
_TORCH_EXTENSIONS = ("torch", "torchvision", "torchcodec", "torchaudio")


def _locked_versions() -> dict[str, set[str]]:
    """Every locked version of every distribution, keyed by name.

    A name can hold several ``[[package]]`` entries - uv writes one per
    resolution fork - which is exactly the shape this module grades.
    """
    packages = tomllib.loads(_LOCK.read_text(encoding="utf-8"))["package"]
    versions: dict[str, set[str]] = {}
    for package in packages:
        version = package.get("version")
        if version is not None:
            versions.setdefault(package["name"], set()).add(version)
    return versions


def _declared_dependencies(name: str, versions: dict[str, set[str]]) -> set[str]:
    """Names the lock records as dependencies of every entry of *name*."""
    packages = tomllib.loads(_LOCK.read_text(encoding="utf-8"))["package"]
    assert name in versions, f"{name} is not in the lock"
    declared: set[str] = set()
    for package in packages:
        if package["name"] == name:
            declared |= {dep["name"] for dep in package.get("dependencies", ())}
    return declared


def test_torch_is_locked_at_one_version():
    """The premise every other check reads: one torch ABI is installed."""
    assert _locked_versions().get("torch") is not None, "torch is not locked"
    assert len(_locked_versions()["torch"]) == 1, (
        f"torch is locked at {sorted(_locked_versions()['torch'])}; a second torch "
        "means a second ABI, and the extension rule below has to be read per fork"
    )


def test_the_lock_holds_more_than_one_torch_extension():
    """Non-vacuity: an empty roster would make the rule below pass blind."""
    locked = _locked_versions()
    present = [name for name in _TORCH_EXTENSIONS if name in locked]
    assert len(present) >= 3, f"expected the torch extension stack in the lock, found {present}"


@pytest.mark.parametrize("name", _TORCH_EXTENSIONS)
def test_a_torch_extension_is_locked_at_one_version(name):
    """One locked torch ABI admits one build of anything linked against it."""
    locked = _locked_versions()
    if name not in locked:
        pytest.skip(f"{name} is not part of this resolution")
    assert len(locked[name]) == 1, (
        f"{name} is locked at {sorted(locked[name])} beside torch "
        f"{sorted(locked['torch'])}. A build linked against a different libtorch "
        "fails at import with a missing C++ symbol, not at resolution - relock it "
        f"with `uv lock --upgrade-package {name}`"
    )


def test_torchcodec_declares_no_torch_dependency():
    """Why the rule cannot be left to the resolver, stated as a fact of the lock.

    ``torchvision`` is coupled to ``torch`` by its own metadata and needs no
    rule; ``torchcodec`` publishes no torch requirement in any wheel, so its
    version is free of the locked torch and only this contract couples them.
    """
    versions = _locked_versions()
    assert "torch" in _declared_dependencies("torchvision", versions), (
        "torchvision no longer declares torch; the asymmetry this rule is built on has changed"
    )
    assert _declared_dependencies("torchcodec", versions) == set(), (
        "torchcodec now declares dependencies - if one of them is torch, the "
        "resolver couples the pair and this contract can be reconsidered"
    )
