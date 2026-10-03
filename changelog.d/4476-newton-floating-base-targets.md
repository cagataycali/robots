### Fixed: a floating-base robot on Newton drives the joints its actions name

On the Newton backend a robot with a free base (a humanoid, a legged robot, the
Microduck) wrote every position target one joint late: Newton lays its target
array out per DOF, and the engine indexed it per coordinate, where the base
spans 7 coordinates but 6 DOFs. Each joint received its neighbour's command and
the last joint received none, so a walking policy fell over within half a
second whatever velocity it was given. Targets now follow the layout Newton
reports, and the same Microduck walk travels on Newton as it does on MuJoCo.
Arms with a fixed base were unaffected.
