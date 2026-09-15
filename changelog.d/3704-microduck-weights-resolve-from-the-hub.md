### Fixed: a bare Microduck weight name is fetched from Pollen's Hub repository, where the weights now live

`docs/policies/microduck.md` said the shipped weights "ship in Pollen's
`microduck` repository under `policies/*.onnx`" and every example under
`examples/microduck/` defaulted `--onnx` to `../microduck/policies/...`.
Pollen removed that directory upstream; the ten weights are published on
the Hub at `pollen-robotics/microduck-policies`, so a reader following the
page hit `no such ONNX` on the first command of the release's headline feature.

`MicroduckPolicy(onnx_path=...)` now accepts a local file or a bare weight
name: a bare name that is not in the working directory is fetched from
`MICRODUCK_POLICIES_HF_REPO` through `huggingface_hub` on first use and cached,
so the page's `MicroduckPolicy(onnx_path="alpha_walking.onnx")` works on a
fresh install. A path with directories that does not exist is refused naming
the repository and the bare name that would fetch it, never answered with a
download. `resolve_microduck_weight()` is exported for callers who want the
path themselves; the four examples default to bare names and share it, and
the `[microduck]` extra carries `huggingface_hub`.
