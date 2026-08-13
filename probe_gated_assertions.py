"""Execute the newton-gated assertions of the new test module under a real newton install."""
import pathlib, sys
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1], flush=True)
from strands_robots.simulation.newton.backend import articulated_solver_error, articulated_solvers
from strands_robots.simulation.newton.simulation import NewtonSimEngine

CANNOT = ("mpm", "semi_implicit", "style3d", "vbd", "xpbd")
ok, fail = [], []


def check(label, fn):
    try:
        fn()
        ok.append(label)
        print(f"  PASS {label}", flush=True)
    except BaseException as exc:  # noqa: BLE001 - report every failure mode
        fail.append((label, f"{type(exc).__name__}: {exc}"[:200]))
        print(f"  FAIL {label}: {type(exc).__name__}: {exc}"[:220], flush=True)


for s in CANNOT:
    def one(s=s):
        try:
            NewtonSimEngine(solver=s)
        except ValueError as exc:
            assert str(exc) == articulated_solver_error(s), f"{exc!s}"
            return
        raise AssertionError("DID NOT RAISE")
    check(f"refuses_with_shared_verdict[{s}]", one)


def unknown():
    try:
        NewtonSimEngine(solver="not_a_solver")
    except ValueError as exc:
        assert "Unknown Newton solver" in str(exc), str(exc)
        return
    raise AssertionError("DID NOT RAISE")


check("unknown_still_reports_unknown", unknown)


def describe_truthful():
    sim = NewtonSimEngine(solver="mujoco")
    try:
        adv = sim.describe()["available_solvers"]
    finally:
        sim.destroy()
    assert adv == sorted(articulated_solvers()), adv
    for r in CANNOT:
        assert r not in adv, r


check("describe_advertises_only_accepted", describe_truthful)
print(f"\nRESULT: {len(ok)} passed, {len(fail)} failed", flush=True)
sys.exit(1 if fail else 0)
