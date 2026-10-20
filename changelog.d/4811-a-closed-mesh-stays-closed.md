### Fixed: a closed `Mesh` refuses every verb instead of starting threads that outlive it

After `Mesh.close()`, `mount`, `connect`, `invoke`, `broadcast` and `subscribe` raise `RuntimeError` naming the
closed mesh, the way a closed `Simulation` already does. Before, a mount on a closed mesh was accepted and its
watchdog and state-stream threads stayed alive after a second `close()`, which returns early.
