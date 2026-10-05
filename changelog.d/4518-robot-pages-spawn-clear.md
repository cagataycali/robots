### Fixed: a robot page's first fence spawns its robot resting on the ground

Thirteen sim robots (LeKiwi, Unitree A1/Go1/Go2, Aliengo, ANYmal C, UR10e,
Open Duck Mini, Asimov v0, RB-Y1, and the Aero, Allegro and Shadow hands) are
authored with part of the model below `z=0`, so the `Robot("<name>")` fence on
their docs page logged `starts N mm inside the ground ... Pass position=[...]`
on the reader's very first run. The registry now carries that position as an
optional `spawn_position`, and `docs/hooks/robot_pages.py` writes it into the
fence (`Robot("lekiwi", position=[0.0, 0.0, 0.0346])`), so the copy-paste runs
clean. `Robot(name)` itself still spawns where it always did.
