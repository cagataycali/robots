# Isaac action-controller install: probing both kinematics reads

`capture.py` drives the production `IsaacDeltaEEFController` through
`LiberoAdapter._install_action_controller` over an engine that exposes the Isaac
action seam backed by a real MuJoCo Franka, so the real install decision runs and
the controller's joint targets have a real physical consequence. Isaac Sim is not
required to reach the install decision (the adapter's probe is duck-typed).

Run once per tree, then compose:

    MUJOCO_GL=egl python3 _art/capture.py /tmp/art-main      # in an upstream/main worktree
    MUJOCO_GL=egl python3 _art/capture.py /tmp/art-branch    # in the PR tree
    python3 _art/compose.py

`compose.py` asserts every number it renders against the two dumps, asserts the
two captures ran in different trees, and refuses to save unless the honored
render is identical across trees and the controller measurably moves the arm.
