### Fixed: a benchmark declaring a per-episode horizon the evaluation cannot honor is refused, not run as an empty evaluation

`evaluate_benchmark` documents that `max_steps` "comes from the benchmark (not a
parameter here)", so the benchmark object is the sole authority for the
per-episode horizon. It was also the one bound of the evaluation's nested loop
with no domain of its own: `n_episodes`, `action_horizon` and `control_substeps`
are all checked at the public entry point, and `eval_policy` checks its own
`max_steps` there too, but a benchmark's horizon was validated on only one of
the four paths that can set it.

`DeclarativeBenchmark.from_dict` (and so `register_benchmark_from_file`) rejected
a non-positive-integer horizon. The constructor it feeds coerced with a bare
`int()`, so `2.7` became `2` and `True` became `1`;
`LiberoAdapter(max_steps=...)`, its `from_file` / `from_text` classmethods and
`load_libero_suite` did the same, one line below a validated `init_jitter`; and a
plain `BenchmarkProtocol` subclass setting the documented `max_steps` attribute -
the extension point the base class invites - was not checked at all, nor was an
assignment to it after construction.

The result was the failure `SimEngine._validate_positive_int` already names for
this parameter: "episodes of zero length, that fabricate a 0% success rate". A
benchmark declaring `max_steps=0` returned `status="success"` reporting
`success_rate: 0.0` and `Avg steps: 0/0` over episodes that applied no action -
carrying the same `success_rate` field a genuine 0% result does - and a
fractional or NaN horizon raised a bare `TypeError` out of `range()`, past the
agent-tool envelope.

The horizon now shares the count domain `positive_count_error` already documents
it under. Each creation path raises `ValueError` where the value was supplied,
and the evaluation loop checks it where it reads it, so a subclass attribute or a
post-construction assignment - neither of which any constructor can see - is
refused with a structured error naming the parameter and the benchmark. The check
runs before `set_eval_seed`, so a rejected evaluation no longer reseeds the
process-global RNG. Omitting the horizon still applies the documented default.
