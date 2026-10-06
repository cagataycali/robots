### Fixed: the README's Mesh row names the call that works

The "What you get" Mesh row said every robot could `tell()` another what to
do, which reads as a method on the `Robot` the quickstart builds. `Robot` has
no `tell`, and a plain `Robot("so100")` has `.mesh = None`. The row now reads
`Robot(mesh=True)` and `robot.mesh.tell(peer, instruction, policy_provider=...)`,
the call the wire accepts.
