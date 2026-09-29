### Fixed: the sim-isaac extra installs the IK stack move_to solves on

Isaac's `move_to` solves its IK with MuJoCo and mink, but `[sim-isaac]` declared neither, so `move_to` failed in an environment installed with that extra alone. The extra now declares `mujoco`, `mink` and `qpsolvers[daqp]` with the same bounds as `[sim-mujoco]`.
