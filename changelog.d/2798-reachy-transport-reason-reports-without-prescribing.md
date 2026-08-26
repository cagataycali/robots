### Fixed: the Reachy transport refusal no longer prescribes an install it cannot establish

`ReachyDriver` resolves its daemon transport lazily and reports a reason rather than
raising when that import fails. The reason named the `[device-connect]` extra as both
the cause and the remedy - correct once, when the transport's parent package imported
`device_connect_edge` at module scope and the extra's absence really did make the leaf
unimportable. That import is now lazy, and the leaf itself imports nothing outside the
standard library, so no `pip install` supplies a module whose absence can reach that
branch. Every cause that still can - a shadowing module, a partial wheel, a corrupt
install - is untouched by installing an extra, so the remedy was advice that cannot
help offered with the confidence of a diagnosis, which is the failure mode the named
refusal exists to prevent, pointed the wrong way.

The reason now reports the module and the underlying `ImportError` and stops there.
That is the shape `drivers.g1._resolve_message_class` already uses, which this
function's own docstring documents itself as sharing, and it is the position
`utils.require_optional` refuses to print a pip line in when it is told a module
arrives from somewhere pip cannot reach: such a line "would hand the caller an
instruction that reports success without supplying the module".

`docs/getting-started/robot-factory.md` quoted the old reason verbatim, including the
remedy and a `ModuleNotFoundError` naming a package that can no longer be the cause.
Nothing graded that block, so it is corrected against the resolver and pinned to it.
The one other site that names the same extra is left alone: it reports a Device
Connect bring-up failure, on a path whose modules do import `device_connect_edge` at
module scope, so there the extra is a real cause and installing it is a real remedy.
