# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blocking the clip encoder leaves both of its registries as it found them.

:func:`tests._blocked_encoder.blocked_encoder` displaces two entries so an
optional dependency reads as absent. Deleting either on the way out does not
undo the block, it orphans it: the next import returns a *different* module
object, and every reference taken before - a sibling test module's
collection-time ``import imageio``, and the double it patched onto it - is left
pointing at a module nothing will look at again.

These two cells pin the exit, which is what the statements inside the block
cannot report on themselves.
"""

from __future__ import annotations

import sys

import pytest

from strands_robots import utils
from tests._blocked_encoder import ENCODER_MODULE, blocked_encoder

imageio = pytest.importorskip(ENCODER_MODULE, reason="imageio not installed - pip install imageio imageio-ffmpeg")


def test_the_block_puts_back_the_module_it_displaced() -> None:
    """The entry holds the same object afterwards, not a hole."""
    with blocked_encoder():
        with pytest.raises(ImportError):
            utils.require_optional(ENCODER_MODULE)

    assert sys.modules[ENCODER_MODULE] is imageio


def test_require_optional_still_answers_with_that_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a cold memo - the ordering the suite collects in - the import is not repeated.

    The memo is swapped for a copy so the cell can empty it without leaving the
    session's own memo short a module.
    """
    monkeypatch.setattr(utils, "_lazy_modules", dict(utils._lazy_modules))
    utils._lazy_modules.pop(ENCODER_MODULE, None)

    with blocked_encoder():
        with pytest.raises(ImportError):
            utils.require_optional(ENCODER_MODULE)

    assert utils.require_optional(ENCODER_MODULE) is imageio, (
        "a fresh module object answers after the block, so a patch a sibling "
        "applied to the module it imported at collection time is invisible here"
    )
