### Tests: the Device Connect suite shrinks ahead of the package's 0.7 removal

Eight test files that pin only Device Connect plumbing are deleted: the lazy
package init, the heartbeat schema, the sync bring-up loop, the state RPC, sim
joint addressing, and a per-robot matrix of 1,102 cells that ran the same two
mocked drivers once per registry entry. The pins on its safety paths stay until
the package itself goes: caller authorization and the e-stop allowlist,
operator consent on `rpc`, the plaintext-transport refusal, and the stop
verdict. The mesh (`strands_robots.mesh`, the `robot_mesh` tool) and the native
Reachy driver are what replace Device Connect.
