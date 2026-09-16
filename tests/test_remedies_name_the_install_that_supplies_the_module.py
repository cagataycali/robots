# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two refusals whose remedy the caller could follow to no effect now name the real install.

``require_optional`` renders whatever it is handed, so a remedy is only as good
as the call site's claim about where the module comes from. Two sites claimed
wrong:

* ``CuroboPolicy._build_motion_gen`` passed ``extra="curobo"``. The extra exists
  and is empty on purpose (cuRobo is not on PyPI; the ``nvidia-curobo`` package
  there is a squatter), so the refusal's first line, ``pip install
  'strands-robots[curobo]'``, exits 0 and changes nothing - the very
  anti-pattern ``require_optional`` documents ``system_install`` as replacing.
  The remedy is now the source checkout, as a ``system_install`` text, with the
  pip block gone.
* ``MotionPlayer._load_file`` passed ``extra="kimodo"`` for torch. ``[kimodo]``
  does carry torch, alongside diffusers, transformers and accelerate - a whole
  other policy's stack - and the ProtoMotions user who only wants to unpickle
  a ``.pt`` motion was told to install it. The remedy is now the package alone.

Both are graded through the public door, with the module blocked, so the text
a caller reads is the text pinned here.
"""

from __future__ import annotations

import pytest

from strands_robots.policies.curobo.policy import CUROBO_SYSTEM_INSTALL_HINT, CuroboPolicy
from strands_robots.policies.protomotions.motion_utils import MotionPlayer
from tests._blocked_module import blocked


def test_curobo_policy_without_curobo_names_the_source_checkout_and_no_pip_line() -> None:
    with blocked("curobo"), pytest.raises(ImportError) as info:
        CuroboPolicy(robot_config="franka.yml")

    text = str(info.value)
    assert info.value.name == "curobo", text
    assert "git clone https://github.com/NVlabs/curobo.git" in text, text
    assert "pip install -e ./curobo" in text, text
    assert "strands-robots[curobo]" not in text, text
    assert "Install with:" not in text, text
    assert CUROBO_SYSTEM_INSTALL_HINT in text, text


def test_the_curobo_remedy_says_why_the_extra_is_not_the_answer() -> None:
    """The text explains the no-op, so a reader who knows the extra exists does not reach for it."""
    assert "[curobo] extra is empty" in CUROBO_SYSTEM_INSTALL_HINT
    assert "squatter" in CUROBO_SYSTEM_INSTALL_HINT
    assert "docs/policies/curobo.md" in CUROBO_SYSTEM_INSTALL_HINT


def test_a_raw_pt_motion_without_torch_names_torch_and_no_other_stack(tmp_path) -> None:
    with blocked("torch"), pytest.raises(ImportError) as info:
        MotionPlayer(str(tmp_path / "motion.pt"))

    text = str(info.value)
    assert info.value.name == "torch", text
    assert "pip install torch" in text, text
    assert "strands-robots[" not in text, text
    assert "kimodo" not in text, text
