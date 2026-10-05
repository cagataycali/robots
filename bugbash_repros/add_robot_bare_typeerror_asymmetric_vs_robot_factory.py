"""
Repro: sim.add_robot(<typo>) raises bare CPython TypeError

Asymmetry: Robot(..., positon=[0,0,0]) is caught by reject_misspelled_kwargs()
in simulation/base.py:190 and refused with a sentence that names the parameter
and offers 'did you mean position?'. The SAME typo on sim.add_robot(name="arm",
positon=[0,0,0]) hits the engine's non-**kwargs signature and the user gets:

    TypeError: MuJoCoSimEngine.add_robot() got an unexpected keyword argument 'positon'

- no did-you-mean,
- names the backend class (an implementation detail) instead of 'add_robot',
- not the structured {"status":"error",...} tool envelope the sibling
  world-mutators return.

Covered call sites of reject_misspelled_kwargs (base.py:190):
  * Robot()                       -> robot.py:864
  * MuJoCoSimEngine.__init__      -> mujoco/simulation.py:1005
  * NewtonSimEngine.__init__      -> newton/simulation.py:225
  * MjlabEngine.__init__          -> mjlab/simulation.py:153
  * create_policy's _MISPELLING_RATIO sibling   -> policies/factory.py:593

MISSING on *every* backend's add_robot:
  * simulation/base.py:1456  (abstract)
  * simulation/mujoco/simulation.py:2338
  * simulation/newton/simulation.py:600
  * simulation/isaac/simulation.py:3346
  * simulation/mjlab/simulation.py:615

Reproduces on strands-labs/robots @ d57bd71f0 (main, 2026-10-05) via
`pip install -e .` + MUJOCO_GL=egl.
"""
import sys
from strands_robots import Robot


def main() -> int:
    sim = Robot("so101", mesh=False)

    # ---- Half A: the sibling guard that works ---------------------------
    try:
        Robot("so101", mesh=False, positon=[0, 0, 0])
    except TypeError as e:
        got_a = str(e)
        print("[A] Robot() typo refusal:")
        print("    ", got_a.splitlines()[0][:200])
        assert "did you mean 'position'" in got_a, "Robot() guard regressed"
        assert "Robot(mode='sim')" in got_a
        print("    ✓ names parameter, offers did-you-mean, names caller")

    # ---- Half B: the guard on add_robot (post-fix) -----------------------
    try:
        sim.add_robot(name="arm2", positon=[0.1, 0, 0])
    except TypeError as e:
        got_b = str(e)
        print("\n[B] sim.add_robot() typo refusal:")
        print("    ", got_b[:200])

        # Properties the fix delivers (symmetric with the Robot() sibling):
        has_hint = "did you mean 'position'" in got_b
        names_add_robot = "add_robot" in got_b
        leaks_backend_class = "MuJoCoSimEngine" in got_b

        print(f"    did-you-mean 'position':  {has_hint}")
        print(f"    names 'add_robot':        {names_add_robot}")
        print(f"    leaks backend class:      {leaks_backend_class}  (should be False)")

        if not has_hint:
            print("\nFAIL (defect present): sim.add_robot() misspelled kwarg "
                  "is a bare CPython TypeError. Expected the sibling "
                  "'did you mean' refusal that Robot() produces for the "
                  "identical typo.")
            return 1
        if leaks_backend_class:
            print("\nFAIL (partial fix): message still names the backend "
                  "implementation class instead of 'sim.add_robot'.")
            return 1

        print("    ✓ PASS: symmetric with Robot() sibling guard.")

        # Half C: a non-typo TypeError raised from INSIDE the method body
        # should propagate unchanged (reshape is pattern-matched).
        try:
            # 'position' is accepted; passing it as a string triggers an
            # internal validation TypeError, not CPython's unknown-kw guard.
            sim.add_robot(name="arm3", position="not a vector")
        except TypeError as inner:
            msg = str(inner)
            if "got an unexpected keyword argument" in msg:
                print("    ✗ reshape over-reached: swallowed an internal TypeError")
                return 1
            print(f"    ✓ internal TypeError propagates unchanged: {msg[:120]}")
        except Exception as inner:
            # Any other refusal (ValueError, etc.) is fine - reshape didn't fire.
            print(f"    ✓ internal {type(inner).__name__} propagates unchanged")

        return 0

    print("\nFAIL: sim.add_robot() accepted the typo silently.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
