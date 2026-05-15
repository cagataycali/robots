#!/usr/bin/env python3
"""List all supported robots and their capabilities.

The registry contains 60+ robots with simulation assets and/or hardware
support. Use list_robots() to discover what's available.

Requirements:
    pip install strands-robots

Usage:
    python examples/06_list_robots.py
"""

from strands_robots import list_robots


def _flag(present: bool, label: str) -> str:
    return f"[{label}]" if present else f"[{' ' * len(label)}]"


print("=== All Robots ===")
for r in list_robots(mode="all"):
    sim_flag = _flag(r.get("has_sim", False), "sim")
    real_flag = _flag(r.get("has_real", False), "real")
    print(f"  {sim_flag} {real_flag}  {r['name']:25s} {r.get('description', '')}")

print(f"\n=== Sim-only ({len(list_robots(mode='sim'))} robots) ===")
for r in list_robots(mode="sim")[:5]:
    print(f"  {r['name']}")

print(f"\n=== Real hardware ({len(list_robots(mode='real'))} robots) ===")
for r in list_robots(mode="real"):
    print(f"  {r['name']}")
