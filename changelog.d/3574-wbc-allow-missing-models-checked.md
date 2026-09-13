### Fixed: `WBCPolicy(allow_missing_models=...)` is checked, not read by truthiness

`WBCPolicy.__init__` checked `walk` with `boolean_flag_error` and read
`allow_missing_models` beside it by truthiness. Every non-empty string is
truthy, so `"false"` - the spelling a JSON `policy_config` reaches for to ask
for the eager ONNX load - selected the test seam instead: no session was
loaded, construction succeeded, and the missing checkpoint surfaced at the
first `get_actions` as a refusal advising `allow_missing_models=False`, the
value the caller believed they had passed. `None`, `0` and `[]` took the
loading branch while spelling neither posture. A non-boolean is now refused at
construction, naming the parameter, ahead of the load it gates; `WBCGaitPolicy`
forwards the flag to the base class, so one check covers both providers.
