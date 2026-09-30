"""``doctor``'s serial row judges the devices that are connected, every one of them.

On a host with no serial device - a cloud GPU box, CI, a sim-only laptop - the
row used to be ``FAIL`` whenever the user was not in ``dialout``, so ``doctor``
exited 1 on every such machine although nothing it could use was broken. It is
now a ``WARN`` that says what to do before plugging an arm in. With devices
present, each one is checked (it used to be only the first), and a device this
process can open read/write passes however access was granted: the ``dialout``
group, a udev rule, or an ACL.
"""

from __future__ import annotations

import grp
import os
import platform
import types
from pathlib import Path

import pytest

from strands_robots import doctor
from strands_robots.doctor import check_serial_permissions


@pytest.fixture()
def linux_user(monkeypatch):
    def setup(*, member: bool, devices: list[str], openable: set[str]):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setenv("USER", "robotuser")
        members = ["robotuser"] if member else ["someone_else"]
        monkeypatch.setattr(grp, "getgrnam", lambda _n: types.SimpleNamespace(gr_mem=members, gr_gid=20))
        monkeypatch.setattr(os, "getgroups", lambda: [20] if member else [1000])
        monkeypatch.setattr(
            Path, "glob", lambda self, pat: iter([Path(d) for d in devices if Path(d).match("/dev/" + pat)])
        )
        monkeypatch.setattr(os, "access", lambda p, _m: str(p) in openable)

    return setup


def test_no_device_and_no_dialout_is_a_warning_that_says_what_to_do(linux_user):
    linux_user(member=False, devices=[], openable=set())
    result = check_serial_permissions()
    assert "  WARN  " in result and "  FAIL  " not in result
    assert "no devices connected" in result
    assert "usermod -aG dialout" in result


def test_the_no_device_warning_does_not_fail_the_run(linux_user, monkeypatch, capsys):
    linux_user(member=False, devices=[], openable=set())
    monkeypatch.setattr(doctor, "CHECKS", (("Serial", "check_serial_permissions"),))
    assert doctor.run_doctor() == 0


def test_every_connected_device_is_checked_not_only_the_first(linux_user):
    linux_user(member=True, devices=["/dev/ttyACM0", "/dev/ttyACM1"], openable={"/dev/ttyACM0"})
    result = check_serial_permissions()
    assert "  FAIL  " in result
    assert "/dev/ttyACM1" in result


def test_a_device_opened_through_a_udev_rule_passes_without_dialout(linux_user):
    linux_user(member=False, devices=["/dev/ttyUSB0"], openable={"/dev/ttyUSB0"})
    result = check_serial_permissions()
    assert "  PASS  " in result
    assert "/dev/ttyUSB0" in result


def test_not_in_dialout_with_a_device_it_cannot_open_still_fails(linux_user):
    linux_user(member=False, devices=["/dev/ttyACM0"], openable=set())
    result = check_serial_permissions()
    assert "  FAIL  " in result
    assert "usermod -aG dialout" in result
