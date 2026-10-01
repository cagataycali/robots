### Fixed: the dashboard voice console works on strands-agents 1.57.2

strands-agents 1.57.2 moved the bidirectional agent from `strands.experimental.bidi`
to `strands.bidi` and removed its built-in `stop_conversation` tool. The voice
console imported both, so on 1.57.2 opening the microphone raised
`ModuleNotFoundError` and `mypy` reported four missing modules. The console now
imports `strands.bidi`, ships its own `stop_conversation` tool that ends the
session through `BidiAgent.cancel()`, and the `strands-agents` floor (core,
`[ollama]` and `[voice]`) is 1.57.2.
