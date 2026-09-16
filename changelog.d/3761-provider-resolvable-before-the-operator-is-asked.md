### Fixed: a policy provider the library cannot resolve is refused before the arm is energized, and the approval prompt stops naming a server that does not exist

`policy_provider` names what a rollout's policy is built from, so it is the
value the other pre-flight checks read the registry *about*: `_policy_port_error`
reads its `requires` entry to decide whether a port is mandatory,
`_policy_requires_error` reads it for the checkpoint keywords. Both deferred an
unresolvable name to `create_policy`, which raises from `_get_policy` - after
`_connect_robot` has energized the arm, and, on the agent tool, after the
operator has approved the rollout. That is the shape both of those checks exist
to close, left open for the argument that names what is being built.

Deferring also made the port check answer for it. With no registry entry to
read, `requires` cannot say a port is optional, so
`start_task(policy_provider="grooot")` was refused as `policy_port is required` -
a port problem reported for a provider problem, the wording
`tests/test_hardware_policy_port_domain.py` already forbids, and the same
sentence a correctly spelled `groot` gets, so the two were indistinguishable.
Following the remedy it prescribed was worse than the refusal: with a port
supplied, `grooot` passed the pre-flight entirely, spent the bring-up window and
an operator's approval, and failed on the executor thread.

`Robot._policy_provider_error` now refuses an unresolvable provider in both task
entry points and in the pre-gate check, before the port is judged, naming the
provider and the ones that resolve. Resolution is asked of
`registry.policies.policy_provider_resolves`, the registry's account of what
`import_policy_class` accepts, so a declared alias (`lerobot`, `random`, `c3`)
and an auto-discovered module (`composite`, `persistent`) are not refused for
being absent from `list_policy_providers()`. A pre-built `policy_object` makes
the provider inert, exactly as it does the port.

The approval prompt described every policy as `at {host}:{port}`, so the 9
registered providers that declare no port - the in-process ones - were announced
to the operator as a server `at localhost:None`, an endpoint that does not
exist. It now says which of three things is true, from the registry rather than
a guess: `policy groot at localhost:5555`, `policy mock built in this process,
no server`, or, for `cosmos3`/`lerobot_async`/`remote`, which dial a server
while defaulting the port, `policy cosmos3 at localhost, on the provider's
default port`.
