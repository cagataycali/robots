### Fixed: the quickstart tells a mesh peer which policy to run

The fleet step of Getting Started called `mesh.tell(peer, "hold the tray
steady")` with no policy, which the wire boundary refuses before the command
leaves the sender. The step now names the policy and its checkpoint as the
boundary takes them (a Hub id, not a local path), and `Mesh.tell`'s
docstring says what it sends and why `policy_provider` is required.
