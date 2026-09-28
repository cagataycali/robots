# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Dynamixel driver: the Feetech driver's verbs over a Protocol 2.0 bus.

:class:`DynamixelDriver` is :class:`~strands_robots.drivers.feetech.driver.FeetechDriver`
with :class:`~strands_robots.drivers.dynamixel.bus.DynamixelBus` behind it - the
same ``send_action`` (degrees, gripper percent open), ``sensors``,
``set_torque``, ``stop`` and 30 Hz :class:`~strands_robots.drivers.rollout.PolicyRollout`,
the same refusals. Only the serial transport is offered.

Only ``koch`` is served: it is the one Dynamixel arm whose motor map lerobot
declares. The ViperX/WidowX/ALOHA arms carry shadow motors on their shoulder
and elbow, and registering them against a six-servo map would command joints
they do not have. Pass ``calibration=`` the JSON ``lerobot-calibrate`` wrote for
the arm (``lerobot_calibration_path("koch_follower", <id>)``).
"""

from __future__ import annotations

from strands_robots.drivers.dynamixel.bus import KOCH_MOTORS, DynamixelBus
from strands_robots.drivers.feetech.driver import FeetechDriver

#: Canonical registry names this driver is registered for.
SUPPORTED_ROBOTS: tuple[str, ...] = ("koch",)


class DynamixelDriver(FeetechDriver):
    """Native Dynamixel driver for the arms in :data:`SUPPORTED_ROBOTS`; see the module."""

    SUPPORTED_ROBOTS = SUPPORTED_ROBOTS
    TRANSPORTS = ("serial",)
    MOTORS = KOCH_MOTORS
    BUS = DynamixelBus
    WIRE = "Dynamixel-native driver for {name} (Protocol 2.0, XL330/XL430)"
