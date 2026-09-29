"""Minimal repro: create_policy("gr00t") returns unhelpful 404 with no "Did you mean" hint.

Users read NVIDIA's docs (which spell it "GR00T", with numeric zeros) and reasonably
guess the provider name is 'gr00t'. The registry key is 'groot' (letter o's).

Compare with Robot('duck') which helpfully suggests 'microduck'.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import create_policy, Robot

print("=== Robot factory gives helpful 'Did you mean' hints ===")
try:
    Robot("duck")
except Exception as e:
    print(f"Robot('duck') -> {type(e).__name__}: {e}")
    print()

print("=== create_policy factory does NOT ===")
# Common typos a real user would make after reading NVIDIA docs:
for typo in ["gr00t", "GR00T", "Groot", "GROOT", "gr0ot", "grOOt"]:
    try:
        create_policy(typo)
    except Exception as e:
        msg = str(e).split("Available:")[0].strip()
        print(f"create_policy({typo!r:10}) -> {type(e).__name__}: {msg}")

# EXPECTED: "Unknown policy provider: 'gr00t'. Did you mean: 'groot'? Available: [...]"
# ACTUAL:   "Unknown policy provider: 'gr00t'. Available: [...]"  (no hint, user re-reads long list)
