### Fixed: README mesh row names the full `robot.mesh.tell(...)` reach-through

The "What you get" Mesh row at `README.md:97` printed `tell()` as if it were a
method on the Robot the hero snippet above constructs, but `tell` lives on
`strands_robots.mesh.core.Mesh` and is only reachable through
`Robot(..., mesh=True).mesh.tell(peer, instruction, policy_provider=...)`.
A reader following the row literally hit `AttributeError: 'MuJoCoSimEngine'
object has no attribute 'tell'`; the hero `Robot("so100")` form also has
`.mesh = None`, so even the half-right `.mesh.tell` reach-through failed with
`AttributeError: 'NoneType' object has no attribute 'tell'`. The row now names
`Robot(mesh=True)` and the fully-qualified call so the one line a reader skims
compiles. No code change.
