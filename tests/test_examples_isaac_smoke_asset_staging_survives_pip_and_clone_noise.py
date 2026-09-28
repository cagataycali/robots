# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Isaac-on-AWS smoke stages the unitree_g1 asset without a silent swallow.

``run_smoke.sh`` fetches the one registry asset the smoke drives on the instance
HOST (the NGC image has no git and runs as a non-root uid). That host block is
built into ``$STAGE`` and shipped to the instance as a single ``AWS-RunShellScript``
document, which runs it under ``/bin/sh`` (dash on Ubuntu) - not under
``run_smoke.sh``'s own ``#!/usr/bin/env bash`` with ``set -euo pipefail``, which
applies only on the operator's machine.

Two ways that staging silently misled on a fresh instance:

1. **A missing pip was swallowed.** The asset was installed with
   ``python3 -m pip install --target ... | tail -1``. Ubuntu's ``python3`` ships
   without pip and without ``ensurepip``, and dash has no ``pipefail``: the pipe
   returned ``tail``'s status (0), so ``python3 -m pip`` printing "No module named
   pip" neither aborted nor was seen, and the asset install was skipped.
   ``provision.sh``'s cloud-init never installed ``python3-pip``. Fixed by
   installing ``python3-pip`` in cloud-init AND replacing the swallowing pipe with
   a POSIX ``|| { ...; exit 1; }`` guard that aborts loudly if pip is ever absent.
   ``set -o pipefail`` is deliberately NOT used - dash rejects it ("Illegal
   option") and would abort the whole staged script.

2. **A clone notice on stdout corrupted the captured path.** ``robot_descriptions``
   1.23.0 prints ``Cloning ...``/``Found commit ...`` to STDOUT on the first import
   that triggers the clone (its ``_cache.py``). A bare ``PKG=$(python3 -c ...
   2>/dev/null)`` folded that notice into ``PKG``, and ``cp -rL "$PKG"`` then got a
   multi-line non-path. Fixed by emitting the real path behind a ``PKGPATH=``
   sentinel and pulling that one line back out with ``sed``.

These pins parse the shipped scripts and run the exact staged idioms under
``sh`` (dash) - against a fake ``pip`` failure and a fake ``robot_descriptions``
that prints the notice - so the fresh-AMI path is verified by construction, not
by provisioning an instance.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "examples" / "isaac_on_aws" / "run_smoke.sh"
PROVISION = REPO_ROOT / "examples" / "isaac_on_aws" / "provision.sh"


def _block(text: str, start_marker: str, end_marker: str) -> str:
    """The lines of a heredoc body between ``start_marker`` and ``end_marker``."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if start_marker in ln]
    assert starts, f"{start_marker!r} not found"
    begin = starts[0] + 1
    ends = [i for i, ln in enumerate(lines) if i > begin and ln.strip() == end_marker]
    assert ends, f"{end_marker!r} not found after {start_marker!r}"
    return "\n".join(lines[begin : ends[0]])


def _cloud_init() -> str:
    return _block(PROVISION.read_text(), "USER_DATA=$(cat <<'CLOUDINIT'", "CLOUDINIT")


def _stage() -> str:
    # The staged block is an unquoted heredoc, so its real ``$`` are written
    # ``\$``. De-escape so the extracted commands can be run as they run on the
    # instance. (``$URL``/``$RD`` etc. that the outer shell expands are left as
    # empty under dash's default; the tests do not depend on them.)
    return _block(SMOKE.read_text(), "STAGE=$(cat <<EOF", "EOF").replace("\\$", "$")


def _set_line(stage: str) -> str:
    return next(ln.strip() for ln in stage.splitlines() if ln.strip().startswith("set "))


def _logical_command(stage: str, needle: str) -> str:
    """The full backslash-continued logical command containing ``needle``."""
    lines = stage.splitlines()
    start = next(i for i, ln in enumerate(lines) if needle in ln)
    collected = [lines[start]]
    j = start
    while collected[-1].rstrip().endswith("\\") and j + 1 < len(lines):
        j += 1
        collected.append(lines[j])
    return "\n".join(collected)


def _pkg_capture(stage: str) -> str:
    """The ``PKG=$(...)`` command, matched over balanced parens."""
    anchor = stage.index("PKG=$(")
    open_paren = stage.index("(", anchor)
    depth = 0
    for i in range(open_paren, len(stage)):
        if stage[i] == "(":
            depth += 1
        elif stage[i] == ")":
            depth -= 1
            if depth == 0:
                return stage[anchor : i + 1]
    raise AssertionError("unbalanced PKG=$(...) capture")


def test_cloud_init_installs_pip_for_the_host_asset_fetch() -> None:
    """provision.sh's cloud-init installs python3-pip on the fresh instance."""
    cloud_init = _cloud_init()
    pip_lines = [
        ln
        for ln in cloud_init.splitlines()
        if "python3-pip" in ln and "apt-get install" in ln and not ln.lstrip().startswith("#")
    ]
    assert pip_lines, (
        "provision.sh cloud-init must `apt-get install -y python3-pip`: run_smoke.sh "
        "fetches the asset on the host with `python3 -m pip`, and Ubuntu's python3 "
        "ships without pip or ensurepip."
    )


def test_the_staged_block_stays_posix_sh_compatible() -> None:
    """The staged `set` line runs under /bin/sh (dash), so no `pipefail`.

    The block is shipped to the instance as an AWS-RunShellScript document and
    runs under dash, which rejects `set -o pipefail` ("Illegal option") and would
    abort the whole staged script. This guards against reintroducing it.
    """
    set_line = _set_line(_stage())
    assert "pipefail" not in set_line, (
        f"the staged `{set_line}` uses pipefail, which dash rejects; the SSM "
        "document runs under /bin/sh and this would abort the staged script"
    )
    result = subprocess.run(["dash", "-c", set_line], capture_output=True, text=True)
    assert result.returncode == 0, f"the staged shell options are not dash-safe: {result.stderr}"


def test_a_missing_pip_aborts_loudly_instead_of_being_swallowed(tmp_path) -> None:
    """The host pip install is guarded, so a missing/failing pip aborts loudly.

    Runs the exact staged install command under `sh` (dash) with a `python3` on
    PATH that exits non-zero - the way a missing pip does - and asserts it aborts
    with a diagnostic rather than continuing past a swallowed failure.
    """
    stage = _stage()
    install = _logical_command(stage, "python3 -m pip")
    assert "install" in install and "--target" in install, "the host pip-install line moved"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python3 = fake_bin / "python3"
    fake_python3.write_text('#!/bin/sh\necho "No module named pip" >&2\nexit 1\n')
    fake_python3.chmod(0o755)

    script = f"{_set_line(stage)}\n{install}\necho REACHED"
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    result = subprocess.run(["sh", "-c", script], capture_output=True, text=True, env=env)

    assert result.returncode != 0, (
        "a failing `python3 -m pip install` did not abort the staged block; the "
        "asset install would be silently skipped on a fresh instance"
    )
    assert "REACHED" not in result.stdout, "execution continued past the failed pip install"
    assert "FATAL" in result.stderr, "the failed install produced no diagnostic"


def test_pkg_capture_tolerates_a_clone_notice_on_stdout(tmp_path) -> None:
    """The PKG capture returns the package path, not the robot_descriptions notice.

    Runs the exact ``PKG=$(...)`` idiom from ``run_smoke.sh`` under `sh` against a
    fake ``robot_descriptions`` whose ``g1_mj_description`` prints the same stdout
    notice the real 1.23.0 clone emits on first import.
    """
    fake = tmp_path / "fakemod"
    (fake / "robot_descriptions").mkdir(parents=True)
    (fake / "robot_descriptions" / "__init__.py").write_text("")
    pkg_path = str(tmp_path / "g1_pkg")
    (fake / "robot_descriptions" / "g1_mj_description.py").write_text(
        # Mirrors robot_descriptions/_cache.py: the clone prints to STDOUT.
        'print("Cloning https://github.com/unitreerobotics/unitree_ros...")\n'
        'print("Found commit deadbeef successfully!")\n'
        f"PACKAGE_PATH = {pkg_path!r}\n"
    )

    pkg_cmd = _pkg_capture(_stage()).replace("PYTHONPATH=/opt/strands/rd", f"PYTHONPATH={fake}")
    script = f'{pkg_cmd}\nprintf %s "$PKG"'
    result = subprocess.run(["sh", "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout == pkg_path, (
        f"PKG captured {result.stdout!r}, not the package path {pkg_path!r}; the "
        "robot_descriptions clone notice on stdout was folded into the path and "
        '`cp -rL "$PKG"` would fail on a multi-line non-path'
    )
    assert "Cloning" not in result.stdout
