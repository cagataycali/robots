### Fixed: a wrong or missing predicate keyword is refused by name, on every DSL surface

`make_predicate` now reads the factory's signature before calling it. A keyword
the predicate does not take is refused with `predicate 'contact_between' takes
geom_a, geom_b; got body_a, body_b` and a missing required keyword with
`predicate 'grasped' is missing gripper_prefix`, where before the factory's own
`TypeError` (`got an unexpected keyword argument 'body_a'`), which names no
accepted keyword, reached `run_policy(stop_when=...)`,
`eval_policy(success_when=...)` and benchmark files verbatim. The DSL compiler
keeps the clause context (`stop_when: ...`) in front of every value refusal.
Closes #4145.
