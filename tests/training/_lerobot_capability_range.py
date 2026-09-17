# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grading a written policy-capability roster against an admitted lerobot.

Two written rosters name the lerobot policy types that expose a capability: the
``*_POLICY_TYPES_FALLBACK`` snapshots in
:mod:`strands_robots.training.lerobot` (the answer each gate gives when lerobot
cannot be imported), and the ``validate()`` paragraph of
``docs/training/overview.md``. Both are graded against the capability set
derived from the INSTALLED lerobot's config registry.

The manifest admits a range - ``lerobot>=0.6.1,<0.7.0`` - and lerobot moves a
capability inside it. lerobot's #4425 added ``use_relative_actions`` to
``VLAJEPAConfig`` after v0.6.1, so the floor accepts four policy types for
relative actions and the next release accepts five. Graded for equality, no
written roster can pass at both ends: the four fail on the newer lerobot, and
adding the fifth fails on the floor, which is the version the lockfile
resolves.

The two directions of that difference do not deserve the same verdict, so they
are graded separately:

roster-only
    The roster names a type the installed lerobot says has no such field.
    Offline, that gate ACCEPTS a run lerobot then silently no-ops or hard-errors
    on, so the roster states something untrue about a type the registry holds.
    A failure at every version in the range.

live-only
    The installed lerobot accepts a type the roster omits. A roster frozen in
    the package cannot know about a capability a NEWER lerobot added, so this is
    a failure only when the installed lerobot is not above the floor the
    manifest declares - the version the roster is written for and the one CI
    resolves. Above the floor it is reported and the suite stays green, and it
    becomes a failure the moment the floor is raised onto that release.
"""

from __future__ import annotations

import tomllib
import warnings
from collections.abc import Iterable
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def declared_lerobot_floor() -> Version:
    """The lowest lerobot release the ``[lerobot]`` extra admits.

    Read from the manifest rather than restated here, so raising the floor
    changes what the rosters are graded against without editing this module.

    Returns:
        The smallest version named by a ``>=`` clause of the extra's ``lerobot``
        specifier.
    """
    manifest = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    extra = manifest["project"]["optional-dependencies"]["lerobot"]
    requirement = next(r for r in map(Requirement, extra) if r.name == "lerobot")
    return min(Version(clause.version) for clause in requirement.specifier if clause.operator == ">=")


def installed_lerobot_version() -> Version | None:
    """The lerobot release importable in this environment, if any.

    Returns:
        Its version, or ``None`` when lerobot is absent or declares a version
        string that cannot be parsed - in which case the caller holds the roster
        to the floor rather than guessing.
    """
    try:
        from importlib.metadata import version

        return Version(version("lerobot"))
    except Exception:
        return None


def roster_problem(
    subject: str,
    written: Iterable[str],
    accepted: Iterable[str],
    *,
    written_label: str,
    accepted_label: str,
) -> str | None:
    """Grade a written roster against the capability set a live registry yields.

    Args:
        subject: What the message should call the roster being graded.
        written: The policy types the roster names.
        accepted: The policy types the gate accepts for the installed lerobot.
        written_label: Message label for types only the roster names.
        accepted_label: Message label for types only the gate accepts.

    Returns:
        The failure the drift deserves, or ``None``. A roster that lags an
        installed lerobot ABOVE the declared floor returns ``None`` and warns:
        the lag is expected of a frozen roster and becomes a failure when the
        floor is raised.
    """
    only_written = sorted(set(written) - set(accepted))
    only_accepted = sorted(set(accepted) - set(written))
    installed = installed_lerobot_version()
    floor = declared_lerobot_floor()

    if only_written:
        return (
            f"{subject} names types the installed lerobot ({installed}) has no such field on: "
            f"{written_label}={only_written}. Offline that gate accepts a run lerobot will not "
            f"honour, so drop them from the roster."
        )
    if not only_accepted:
        return None
    if installed is None or installed <= floor:
        return (
            f"{subject} drifted from the lerobot it is written for (the declared floor {floor}): "
            f"{accepted_label}={only_accepted}. Name them in the roster."
        )
    warnings.warn(
        f"{subject} omits {accepted_label}={only_accepted}: the installed lerobot {installed} "
        f"accepts them and the declared floor {floor} does not, and the roster tracks the floor. "
        f"Raising the floor onto {installed} makes this a failure until the roster names them.",
        stacklevel=3,
    )
    return None
