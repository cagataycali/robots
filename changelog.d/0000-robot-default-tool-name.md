### Fixed: a sim robot's default tool name is one a model provider accepts

`Robot(" so100 ")` finds the so100 (names are looked up whitespace- and
case-tolerantly), but its default tool name was built from the raw string,
`" so100 _sim"`, which registers with an Agent and then fails the first Bedrock
call with a `ValidationException` on `toolSpec.name`. The default is now the
name with every run of characters outside `[A-Za-z0-9_-]` turned into one `_`,
capped at 64 characters (`so100_sim`, `my_arm_v2_sim`). An explicit
`tool_name=` is still refused, not rewritten.
