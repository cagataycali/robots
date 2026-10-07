"""
robot_observation_keys asymmetry repro
=======================================

Public introspection on SimEngine ships `robot_action_keys(name)` and
`robot_joint_names(name)` — both are taught by every per-robot hero page in
docs/robots/*.md (162 refs) and both appear in error-message hints like
"use action='robot_joint_names' to see one robot's joints"
(strands_robots/simulation/isaac/simulation.py:10188,10203).

There is NO public counterpart for observation keys, even though:

  (a) `Policy.preflight(observation_keys: set[str], ...)` is the documented
      pre-construction hook (strands_robots/policies/base.py:288-317) and its
      docstring says the keys are "as returned by SimEngine.get_observation"
      (base.py:306-308);
  (b) Three internal call sites literally build the set this way
      (strands_robots/simulation/base.py:1981, hardware_robot.py:1664,
      tools/run_policy.py:812) and name the local `observation_keys`.

A policy author reading those contracts naturally reaches for
`robot.robot_observation_keys("microduck")` and hits:

    AttributeError: 'MuJoCoSimEngine' object has no attribute 'robot_observation_keys'

The AttributeError has no "did you mean set(robot.get_observation(name).keys())"
hint and the method is absent from the class, so `dir()` and tab-completion
drop silently.

Reproduces on both microduck (14-DoF biped, floating base) and so101 (6-DoF arm,
fixed base), i.e. with and without a free joint — so it isn't a per-robot
accident, it's a missing public method.

Verified on strands-robots @ v0.5.3 (0.1.dev1+g5e48817d9, editable).
"""

from strands_robots import Robot


def probe(name: str, **kwargs) -> None:
    r = Robot(name, **kwargs)
    try:
        aks = r.robot_action_keys(name)
        jns = r.robot_joint_names(name)
        obs = r.get_observation()
        obs_keys = set(obs.keys()) if isinstance(obs, dict) else set()

        print(f"\n== {name} ==")
        print(f"  robot_action_keys({name!r})   -> {len(aks)} keys (public method OK)")
        print(f"  robot_joint_names({name!r})   -> {len(jns)} names (public method OK)")
        print(f"  set(get_observation().keys()) -> {len(obs_keys)} keys (what preflight() receives)")
        try:
            oks = r.robot_observation_keys(name)
            print(f"  robot_observation_keys({name!r}) -> {oks!r}  (should not reach here)")
        except AttributeError as e:
            print(f"  robot_observation_keys({name!r}) -> AttributeError: {e}")
            print(f"    (user is stuck: no 'did you mean' hint, method absent from dir())")
    finally:
        r.cleanup()


if __name__ == "__main__":
    probe("microduck")        # floating-base biped
    probe("so101", mesh=False)  # fixed-base manipulator
