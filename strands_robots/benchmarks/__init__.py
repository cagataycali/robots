"""Strands Robots Benchmarks - per-benchmark adapters layered on :mod:`strands_robots.simulation.benchmark`.

Adapters live in optional extras so the core package stays dependency-free.
Importing this namespace is cheap; the heavy work happens when a specific
adapter submodule is imported (e.g. ``from strands_robots.benchmarks.libero
import LiberoAdapter``).

Currently shipped adapters:

* ``strands_robots.benchmarks.libero`` - LIBERO (Panda-only, ~130 tasks).
  Install with ``pip install 'strands-robots[benchmark-libero]'``.

Tracked follow-ups: Meta-World (#108), RoboSuite (#109).
"""
