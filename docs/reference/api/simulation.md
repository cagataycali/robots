# Simulation

`strands_robots.simulation` holds the engine contract, the backend factory and the data types a simulation returns. After this page you know how backends are created, what `SimWorld` carries, and how to register your own.

## Factory

::: strands_robots.simulation.factory
    options:
      heading_level: 3
      members:
        - create_simulation
        - list_backends
        - register_backend

## Engine contract

`stop_policy` returns a `json` block with `robot`, `was_running` and `exited`; `exited` is null when there was no worker to join, the first stop after a rollout ended on its own adds `last_result`, and an empty name means the only rollout in flight. The MuJoCo engine waits 1 s (`MuJoCoSimulation._POLICY_STOP_JOIN_TIMEOUT`) for the worker to exit first.

::: strands_robots.simulation.base.SimEngine
    options:
      heading_level: 3
      show_root_heading: true
      filters: ["!^_"]
      members_order: source

## Capabilities

::: strands_robots.simulation.capabilities
    options:
      heading_level: 3

## World model

::: strands_robots.simulation.models
    options:
      heading_level: 3
      members:
        - SimWorld
        - SimRobot
        - SimObject
        - SimCamera
        - SimStatus
        - TrajectoryStep

## Run-policy observers

::: strands_robots.simulation.observers
    options:
      heading_level: 3
      members:
        - RunPolicyObserver
        - RunPolicyEvent
        - RunPolicyStarted
        - RunPolicyStep
        - RunPolicyEnded
        - RunPolicyOutcome
        - StoppedReason
        - ActionResolution

## Models and assets

::: strands_robots.simulation.model_registry
    options:
      heading_level: 3
      members:
        - resolve_model
        - resolve_urdf
        - register_urdf
        - list_registered_urdfs
        - list_available_models

## Predicates and benchmarks

::: strands_robots.simulation.predicates
    options:
      heading_level: 3
      members:
        - make_predicate
        - register_predicate

::: strands_robots.simulation.benchmark
    options:
      heading_level: 3
      members:
        - register_benchmark
        - unregister_benchmark
        - get_benchmark
        - list_benchmarks
