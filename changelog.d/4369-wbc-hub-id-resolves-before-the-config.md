### Fixed: a `wbc` checkpoint given as a HuggingFace id builds instead of failing after the download

`WBCPolicy(checkpoint="<org>/<repo>")` downloaded the ONNX files and then
failed `main ONNX checkpoint not found (resolved: <snapshot>/<org>/<repo>)`,
because the config was resolved before the id became a local directory and
read the id as the main ONNX file's path. The constructor now resolves the id
to its snapshot before the config reads it, so the snapshot's `config.json` or
its canonical ONNX names are found the way they are for a local checkout; the
session loader downloads nothing more and the `allow_missing_models` stub seam
still makes no network call. (#4161)
