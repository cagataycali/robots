### Fixed: a Microduck command-width refusal names where the width came from

`MicroduckPolicy` sizes the command block from the graph's declared `obs` input,
because `command_names` names which slots a skill reads and is not a width -
seven of the ten weights Pollen ships name fewer slots than their graph consumes.
Both width refusals credited `command_names` regardless, so `roulade.onnx` with a
three-wide `command=` was refused as `expected 13 (from command_names=['twist'])`
when those names sum to 3, and a session declaring no names at all was told the
width came `from command_names=None`. Nothing either message pointed at yields
the number it asked for. Each now names the quantity it measured against, with
the arithmetic a caller can check (`the graph's obs input declares 61 - 48 fixed
blocks`), and keeps naming `command_names` where the sum really is the authority.
`docs/policies/microduck.md`, which still taught the summed rule, is corrected
with it.
