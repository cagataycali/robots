#!/usr/bin/env python3
"""List all supported robots and their capabilities.

The registry contains 38+ robots with simulation assets and/or hardware
support. Use list_robots() to discover what's available.

Requirements:
    pip install strands-robots

Usage:
    python examples/06_list_robots.py
"""

from strands_robots import list_robots

print("=== All Robots ===")
for r in list_robots(mode="all"):
    sim = "🎮" if r.get("has_sim") else "  "
    real = "🔧" if r.get("has_real") else "  "
    print(f"  {sim} {real}  {r['name']:25s} {r.get('description', '')}")

print(f"\n=== Sim-only ({len(list_robots(mode='sim'))} robots) ===")
for r in list_robots(mode="sim")[:5]:
    print(f"  {r['name']}")

print(f"\n=== Real hardware ({len(list_robots(mode='real'))} robots) ===")
for r in list_robots(mode="real"):
    print(f"  {r['name']}")
