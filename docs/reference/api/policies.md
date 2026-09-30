---
description: The Policy contract, how a provider string resolves through create_policy, and the cache that keeps a loaded model.
---

# Policies

A `Policy` turns an observation into actions. Providers live in `registry/policies.json` and resolve through `create_policy`: the `Policy` contract, how a provider string resolves, and the persistent cache that keeps a loaded model between calls.

## Contract

::: strands_robots.policies.base
    options:
      heading_level: 3
      members:
        - Policy
        - ChunkedPolicy
        - resolve_chunk_length
        - align_action_values

## Factory

::: strands_robots.policies.factory
    options:
      heading_level: 3
      members:
        - create_policy
        - register_policy
        - list_providers
        - list_aliases
        - import_policy_class
        - preflight_policy
        - preflight_reason
        - UntrustedRemoteCodeError

## Built-in policies

::: strands_robots.policies.mock.MockPolicy
    options:
      heading_level: 3
      show_root_heading: true

::: strands_robots.policies.composite.CompositePolicy
    options:
      heading_level: 3
      show_root_heading: true

## Persistent cache

::: strands_robots.policies.persistent
    options:
      heading_level: 3
      members:
        - PersistentPolicy
        - preload
        - list_cached
        - evict

## LeRobot policy types

::: strands_robots.policies.lerobot_local.resolution.list_policy_types
    options:
      heading_level: 3
      show_root_heading: true
