# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Importing ``strands_robots.drivers.feetech`` must not pull ``scservo_sdk``.

A CI box that does not have the vendor SDK installed must be able to import
the driver package and grade the codec against the frames the datasheet
publishes. Any transitive import of ``scservo_sdk`` at module load - directly
or via ``pyserial`` on some hosts - defeats that.

This mirrors the module-load pin :mod:`strands_robots.drivers.dynamixel`
carries against ``dynamixel_sdk``.
"""

from __future__ import annotations

import importlib
import sys


def test_import_does_not_pull_scservo_sdk() -> None:
    # Drop any prior import so we can grade this one from scratch.
    for name in list(sys.modules):
        if name.startswith(("strands_robots.drivers.feetech", "scservo_sdk")):
            del sys.modules[name]

    importlib.import_module("strands_robots.drivers.feetech")
    assert "scservo_sdk" not in sys.modules, (
        "importing strands_robots.drivers.feetech pulled scservo_sdk into sys.modules; "
        "the codec is meant to be importable on a box without the vendor SDK"
    )
