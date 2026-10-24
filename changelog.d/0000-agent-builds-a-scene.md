### Added: an example where one agent builds a world and drives the arm in it

`notebooks/examples/agent_builds_a_scene.py` hands `sim.tool` and `arm.tool` to one strands `Agent`: it places three
cubes and a camera, moves `shoulder_pan` by 0.3 rad, renders the camera and prints every tool call it made. A test runs
it with a scripted model, so the generated tools and their strands wiring are checked on every PR without a network.
