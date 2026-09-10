# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin that ``MotorSpec.norm_mode`` refuses values outside its two-mode vocabulary.

A ``Literal`` type annotation on a dataclass field checks nothing at runtime,
so without a ``__post_init__`` guard any value that is not exactly ``"degrees"``
silently selects the percent branch in ``to_value`` / ``to_counts`` - the wrong
unit on physical hardware, reported as success.

The retired pre-calibration constructor signature ``MotorSpec(1, -180, 180)``
positionally binds ``norm_mode=-180, resolution=180``, which is the exact
spelling this guard must refuse: ``-180`` is not a mode, and a bus built on it
reads and commands percent across a 0..180 pseudo-encoder.

Addresses the review thread on
``strands_robots/drivers/feetech/bus.py:138``.
"""

from __future__ import annotations

import pytest

from strands_robots.drivers.feetech.bus import MotorSpec


class TestNormModeDomain:
    """MotorSpec refuses norm_mode values outside the accepted vocabulary."""

    def test_degrees_accepted(self) -> None:
        spec = MotorSpec(motor_id=1, norm_mode="degrees")
        assert spec.norm_mode == "degrees"

    def test_range_0_100_accepted(self) -> None:
        spec = MotorSpec(motor_id=6, norm_mode="range_0_100")
        assert spec.norm_mode == "range_0_100"

    def test_default_is_degrees(self) -> None:
        spec = MotorSpec(motor_id=1)
        assert spec.norm_mode == "degrees"

    @pytest.mark.parametrize(
        "bad_mode",
        [
            -180,  # the retired positional signature
            "DEGREES",  # case slip
            "percent",  # vocabulary slip
            "range_0_100 ",  # trailing whitespace
            "",  # empty string
            None,  # None
            True,  # bool
            0,  # falsy int
        ],
        ids=[
            "retired-positional-neg180",
            "case-slip-DEGREES",
            "vocabulary-slip-percent",
            "trailing-whitespace",
            "empty-string",
            "none",
            "bool-true",
            "falsy-int-zero",
        ],
    )
    def test_bad_norm_mode_refused(self, bad_mode: object) -> None:
        with pytest.raises(ValueError, match="norm_mode must be one of"):
            MotorSpec(motor_id=1, norm_mode=bad_mode)  # type: ignore[arg-type]

    def test_retired_three_positional_args_refused(self) -> None:
        """``MotorSpec(1, -180, 180)`` is the pre-calibration constructor.

        It positionally binds ``norm_mode=-180, resolution=180``, which is a
        silent wrong motion (percent across 0..180) rather than an error.
        """
        with pytest.raises(ValueError, match="norm_mode must be one of"):
            MotorSpec(1, -180, 180)  # type: ignore[arg-type]
