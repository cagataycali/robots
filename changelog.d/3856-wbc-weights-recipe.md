### Docs: the WBC weights recipe fetches the two controllers, not a 4.6 GB tree

`docs/policies/wbc.md` obtained the two decoupled-WBC G1 controllers -- 1.8 MB of
ONNX each -- by cloning the whole `NVlabs/GR00T-WholeBodyControl` git-LFS tree,
so the first command a WBC user runs downloaded 4.6 GB for 3.6 MB of weights.
The recipe now fetches the two artifacts directly from the LFS media host, and
`tests/policies/wbc/test_docs_weights_recipe_fetches_the_two_controllers.py`
pins it: the fence that populates the checkpoint directory must not clone a
repository, and every `.onnx` name it states must be one
`WBCPolicy._default_onnx_paths` resolves, read from the module's constants
rather than transcribed.
