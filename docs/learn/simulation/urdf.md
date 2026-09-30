---
description: How Robot("<name>") compiles a robot_descriptions URDF into a MuJoCo asset on first use, what the loader adds, what it repairs, and what it refuses.
---

# URDF robots

By the end of this page you know which robots `Robot("<name>", mode="sim")` builds from a URDF, what the loader adds to make the URDF a MuJoCo model, and the sentence you get when a description cannot be compiled.

```bash
pip install 'strands-robots[sim-urdf]'   # trimesh, pycollada, xacrodoc, plus [sim-mujoco]
```

## Which robots

`robot_descriptions` ships {{n:urdf_robots}} descriptions with a URDF, no MJCF sibling and no curated entry in `robots.json`. They are the `source: "urdf"` rows of `list_robots()`, listed by `strands_robots.list_urdf_only()`, and appear in the [catalog](../../robots/index.md) with a thumbnail and the upstream commit. {{n:urdf_robots_sim}} of them build; a description that does not is listed with `has_sim` false and the refusal on its page.

A curated name always wins. `panda` has both a `panda_description` URDF and a `panda_mj_description` MJCF, and the registry serves the curated entry; the URDF path never shadows a name you already know.

```python
from strands_robots import Robot, list_urdf_only

print(len(list_urdf_only()))          # the URDF long tail
robot = Robot("atlas_v4", mode="sim")  # first call compiles the asset
```

## What the loader does

`strands_robots/assets/urdf.py` runs once per robot, on the first `Robot()` resolving the name, and writes a Menagerie-shaped directory under `~/.strands_robots/assets/<module>/` (`robot.urdf`, `robot.xml`, `scene.xml`, `meshes/`, `urdf_asset.json`). Later calls reuse it.

1. **Meshes.** Every `<mesh filename>` is resolved: `package://<pkg>/...` against the description's package and repository, relative paths against the URDF, package and repository, and must stay under them. STL and OBJ are copied. Collada (`.dae`), PLY and glTF are converted with trimesh to binary STL, or to OBJ above MuJoCo's 200,000-face STL limit. A Collada file with a missing texture image is read anyway.
2. **Repairs.** An inertial that is zero, not positive definite or under MuJoCo's `mjMINVAL`, or a link with no inertial, gets a small positive one. A primitive with a zero dimension is dropped. A visual with several `<material>` children keeps the first. A xacro prefix left unbound is declared so the file parses. Gazebo, transmission and `ros2_control` elements are removed.
3. **Compiler block.** `<mujoco><compiler meshdir="." discardvisual="false" fusestatic="false" balanceinertia="true"/></mujoco>` keeps the visual meshes and every link name, so `move_to` and cameras address the frames the URDF names.
4. **Base.** Descriptions tagged humanoid, biped, quadruped, wheeled, mobile manipulator or drone get a free joint on the root unless the URDF already has one; arms, hands and educational rigs stay bolted to the world. Every root is lifted until its lowest geometry clears the floor by a centimetre: a URDF has no floor and its zero pose often sits below it.
5. **Actuators.** One position actuator per hinge and slide joint, named after the joint. The gain is the URDF `effort` limit (clamped to 5..2000), the force range is plus or minus that effort, the control range is the joint range. Finger, thumb, knuckle, jaw and gripper joints are capped at 20 so arm effort does not crush the hand. Damping defaults to gain over 20 for hinges and over 10 for slides; hinges get 0.01 armature.
6. **Scene.** `scene.xml` includes `robot.xml` and adds a checker floor, a sky and one directional light, so cameras, renders and thumbnails behave as on every other robot.

`urdf_asset.json` records the module, the upstream repository and commit, each mesh's source and format, the actuated joint list and `nu`. `scripts/build_urdf_registry.py` reads those files into `strands_robots/registry/urdf_robots.json`, so the joint count a page prints is the compiled model's.

## Xacro-only descriptions

Twenty-three descriptions (the Universal Robots family, Kinova Jaco, xArm, Franka FER and FR3 v2, Stretch SE3) ship xacro and no URDF. `robot_descriptions` renders and caches them with `xacrodoc`, which `[sim-urdf]` installs; without it the loader refuses with the sentence that names the package.

## Refusals

Each failure class has one sentence, returned by `Robot()` and recorded in the registry:

| class | sentence |
|---|---|
| clone | upstream clone failed or URDF_PATH missing after import |
| mesh missing | a mesh the URDF names is not in the package or repository |
| mesh outside tree | a mesh the URDF names resolves outside the description tree (an absolute or `~` path, or a symlink out of the URDF, package and repository directories) |
| mesh format | mesh format `<ext>` is not loadable by MuJoCo and has no converter |
| mesh convert | trimesh could not read `<file>` |
| no trimesh | mesh format `<ext>` needs trimesh: `pip install 'strands-robots[sim-urdf]'` |
| compile | MuJoCo refused the compiled spec: `<first line>` |
| no joints | the compiled model has no actuated joint |
| xacro | the description ships xacro only; rendering it needs xacrodoc |

At this commit the one refused description is `eve_r3`: its upstream repository was deleted.

## Viewer

The compiled MJCF exists only on your machine, so URDF robots have no in-browser 3D view. Their pages show a local render made by `docs/hooks/render_thumbnails.py --offscreen`, which loads the same `robot.xml` with MuJoCo's offscreen renderer.

## Your own URDF

The same loader serves a file of yours through `build_from_urdf`:

```python title="sketch"
from strands_robots.assets.urdf import build_from_urdf

info = build_from_urdf("my_robot.urdf", "~/.strands_robots/assets/my_robot", name="my_robot", tags={"arm"})
print(info.nu, info.joints)
```

`add_robot(urdf_path=...)` still loads a URDF directly, without actuators or repairs; use the builder for the treatment the catalog robots get.
