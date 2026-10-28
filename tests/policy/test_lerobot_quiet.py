"""A checkpoint saved on another machine loads on the device asked for without LeRobot's noise about it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from strands_robots import Description, Policy, Robot
from strands_robots.policy.lerobot import server_config


@pytest.fixture(scope="module")
def foreign_act(tiny_act, tmp_path_factory):
    """The tiny ACT, as if trained on a device this machine does not have (a CUDA box, read on a Mac)."""
    from lerobot.utils.device_utils import is_torch_device_available

    path = Path(shutil.copytree(tiny_act, tmp_path_factory.mktemp("ckpt") / "foreign"))
    cfg = json.loads((path / "config.json").read_text())
    cfg["device"] = next(d for d in ("mps", "cuda", "xpu") if not is_torch_device_available(d))
    (path / "config.json").write_text(json.dumps(cfg))
    return str(path)


def _server_config(ckpt: str) -> None:
    assert server_config(Description.load("so101"), ckpt, device="cpu").model.device == "cpu"


def _run(ckpt: str) -> None:
    arm = Robot("so101")
    arm.sim.camera("front", size=(64, 48), attach=arm)
    arm.run(Policy(ckpt, device="cpu"), steps=2)


@pytest.mark.parametrize("load", [_server_config, _run], ids=["server_config", "Policy-run"])
def test_device_cpu_loads_on_cpu_without_a_saved_device_warning_or_a_print(foreign_act, load, caplog, capsys):
    with caplog.at_level("WARNING"):
        load(foreign_act)
    assert [r.getMessage() for r in caplog.records if "is not available" in r.getMessage()] == []
    assert "Loading weights" not in capsys.readouterr().out
