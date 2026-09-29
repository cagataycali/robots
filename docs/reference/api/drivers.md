---
description: The HardwareDriver protocol a native driver satisfies and the functions that register and look drivers up.
---

# Drivers

A native driver speaks a robot's wire protocol without lerobot. `Robot(name, mode="real", driver="strands")` builds one through the registry below. After this page you know the `HardwareDriver` protocol a driver must satisfy and the functions that register and look drivers up.

## Protocol

::: strands_robots.drivers.base.HardwareDriver
    options:
      heading_level: 3
      show_root_heading: true
      members_order: source

## Helpers for driver authors

::: strands_robots.drivers.base
    options:
      heading_level: 3
      members:
        - constructor_keywords
        - missing_driver_members
        - drifted_driver_parameters
        - declared_verbs
        - undeclared_verb_error
        - policy_step
        - decode_motor_state

## Native driver registry

::: strands_robots.drivers.registry
    options:
      heading_level: 3
      members:
        - resolve_driver
        - register_native_driver
        - get_native_driver_class
        - list_native_drivers
        - list_driver_coverage
        - driver_choice_error

## Shipped drivers

::: strands_robots.drivers
    options:
      heading_level: 3
      members:
        - shipped_robot_names
