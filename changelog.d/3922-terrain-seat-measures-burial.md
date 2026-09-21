### Fixed: a floating base spawned on terrain is seated by measurement, not by a height sample

`create_world(terrain=...)` states that a floating-base robot spawns SEATED on
the local surface. The seat offset the base by the heightfield height beneath its
own `(x, y)` and assumed the model's flat pose already cleared `z=0`. Neither
holds for a real asset: the surface under a foot 0.3 m out is not the surface
under the base, LeKiwi's wheels sit 34.6 mm below its root body (its own scene
recesses the floor to -0.1 m) and a straight-legged quadruped's feet sit 120 mm
below `z=0`. Every floating-base asset in the registry therefore spawned buried
on `terrain="rough"` - measured 8.8 mm (`unitree_g1`) to 39.5 mm (`lekiwi`) -
and the episode opened with the contact solver ejecting the robot, the one state
this seat exists to prevent.

The seat now lifts the base by the depth its own kinematic tree is MEASURED to be
inside the ground, read off MuJoCo's contact solve: the penetration depth for a
contact whose normal points up, and the surface height above the contact point
for a horizontal one - a shin inside the heightfield prism is pushed sideways, so
its `dist` says nothing about how far down it is (the A1's calves measure -24.3 mm
at `normal_z` 0.000 while the surface stands 84.5 mm above the contact point).
Ownership is by `body_rootid`, so a free-jointed task object carried inside the
robot's own MJCF is not mistaken for the base's own burial. The lift is iterated
to a tenth of a millimetre; a robot already clear of the surface keeps its pose
exactly, and flat ground is untouched.
