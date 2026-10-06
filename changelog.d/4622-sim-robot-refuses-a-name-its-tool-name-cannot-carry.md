### Fixed: a simulated `Robot` refuses a name its default tool name cannot carry

The registry forgives whitespace around a robot name, but the default sim tool
name is built from the name as typed. `Robot("so100 ")` therefore built under
`status="success"` as the tool `"so100 _sim"`, and the refusal only came from the
model provider on the first Agent call. The derived tool name is now screened
with the same rule as an explicit `tool_name=`, and the refusal names the clean
spelling: `write the name as Robot('so100') or pass tool_name=`. Aliases are
unchanged - `Robot("h1")` is still the tool `h1_sim`.
