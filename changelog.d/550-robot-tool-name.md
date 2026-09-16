### Added: `Robot(tool_name=...)` - two robots of one type can share an Agent

`Robot()` hard-coded the name the agent sees the robot under: `"<name>_sim"` in
simulation, the canonical robot name on hardware. So
`Agent(tools=[Robot("so101"), Robot("so101")])` - a bimanual pair, or a real
arm beside its sim twin - died in the Strands tool registry with
`Tool name 'so101_sim' already exists` (naming an object repr), and the
obvious escape, `Robot("so101", tool_name="left_arm")`, raised a `TypeError`
from the factory's own internal forward.

`tool_name` is now a keyword-only factory parameter on every path (simulation,
lerobot hardware, native driver); the default is unchanged. A name the
registry would refuse (empty, spaces, slashes, a non-string) is refused at the
call site with the remedy, before any backend is built.
