### Fixed: `DeclarativeBenchmark` now applies its string checks on both construction paths

`DeclarativeBenchmark` has two construction paths, and they applied different
checks to the same values. `from_dict` refused a `name`, `default_robot`, `scene`
or `instruction` that was not a string; the constructor stored each one raw, so a
directly constructed benchmark could carry a value the evaluation loop -- or the
policy it drives -- then had to deal with. Only `max_steps` and
`supported_robots` were mirrored across the two.

Two of the four were silent rather than loud. `instruction=42` was accepted and
handed to the policy verbatim as its task command, because `PolicyRunner` falls
back to `spec.instruction` when the caller passes none and cannot tell an `int`
from an instruction; a language-conditioned policy received a value its tokenizer
cannot take, and the evaluation reported `status="success"`. A falsy non-string
`scene` such as `[]` was skipped by the truthiness test in `on_episode_start`, so
a declared scene was never loaded, also under `status="success"`. A non-string
`name` was stored and then advertised as the benchmark's id by
`sim.list_benchmarks()`.

All four now go through one module-level string domain, invoked from both paths
with their own context, so the key a spec file sets and the keyword a direct
construction passes are held to the same rule. `default_robot` is checked ahead
of the `supported_robots` membership test, so a non-string is reported as the
wrong type rather than as a robot missing from the supported set. The accepted
side is unchanged: `instruction=""` and `scene=""`/`scene=None` are the
documented ways to declare neither.
