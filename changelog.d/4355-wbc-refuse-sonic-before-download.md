### Fixed: `wbc` refuses the SONIC inference stack before downloading it

A HuggingFace checkpoint id whose file list is the SONIC VLA inference stack
(`nvidia/GEAR-SONIC`) is now refused from a metadata-only listing, before
`snapshot_download` fetches its 1.26 GB of ONNX. A local directory holding a
SONIC file next to the canonical `GR00T-WholeBodyControl-Balance.onnx` is no
longer refused.
