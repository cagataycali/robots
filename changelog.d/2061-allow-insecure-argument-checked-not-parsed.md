### Fixed: `allow_insecure="false"` enabled insecure Device Connect transport instead of refusing it

`resolve_allow_insecure` resolves one security posture from two sources and
documents the argument as outranking `DEVICE_CONNECT_ALLOW_INSECURE`. The
environment value was parsed - only `("true", "1", "yes")` opt in - while the
argument was returned as given. A non-empty string is truthy, so the two sources
disagreed about the same value: `resolve_allow_insecure("false")` resolved to
insecure while `DEVICE_CONNECT_ALLOW_INSECURE=false` resolved to secure, and
every *off* spelling (`"false"`, `"no"`, `"0"`, `"off"`) inverted on the
higher-precedence path. `init_device_connect(allow_insecure="false")` logged the
prominent INSECURE-mode warning and handed the string through to the runtime.

Each source is now held to its own declared type: the environment value is still
parsed, and the argument - declared `bool | None` - is checked, with a non-boolean
refused before any runtime is constructed. A numpy boolean is accepted and
normalized, so the declared `bool` return is real for a caller's comparison
result. A non-string `env_value` is refused too, rather than raising
`AttributeError` from `.lower()` out of a function declared to return `bool`.

The argument is checked rather than parsed with the same vocabulary because
parsing would move which spellings invert rather than remove the inversion:
`"on"`, `"enabled"` and `"y"` are absent from that vocabulary, so each would
silently resolve to secure while reading as an opt-in.
