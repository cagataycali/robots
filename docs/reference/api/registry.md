---
description: Resolve an alias to a canonical robot, list robots by category, inspect a provider entry, add your own.
---

# Registry

The registry is `strands_robots/registry/robots.json` and `policies.json`, read through the functions below: resolve an alias to a canonical robot name, list robots by category, inspect a provider entry, add your own robot without editing the package.

## Robots

::: strands_robots.registry.robots
    options:
      heading_level: 3
      members:
        - resolve_name
        - get_robot
        - list_robots
        - list_robots_by_category
        - list_aliases
        - joint_labels
        - has_sim
        - has_hardware
        - get_driver
        - get_hardware_type
        - format_robot_table

## Loader

::: strands_robots.registry.loader
    options:
      heading_level: 3
      members:
        - normalize_robot_name
        - reload
        - invalidate_cache

## Discovery through robot_descriptions

::: strands_robots.registry.discovery
    options:
      heading_level: 3
      members:
        - is_discoverable
        - list_discoverable
        - discover_robot
        - descriptions_module

## User robots

::: strands_robots.registry.user_registry
    options:
      heading_level: 3
      members:
        - register_robot
        - unregister_robot
        - list_user_robots

## Policy providers

::: strands_robots.registry.policies
    options:
      heading_level: 3
      members:
        - get_policy_provider
        - list_policy_providers
        - list_policy_aliases
        - resolve_policy
        - build_policy_kwargs
