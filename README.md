# gripper_dim_index domain — artifact

`capture.py` runs one real decode per index spelling onto a MuJoCo Panda through the
shipped `MinkIKBridge` and dumps facts + renders; run once per tree.
`compose.py` builds the figure and asserts every number it prints against the two dumps.
`probe_gripper_dim_index.py` is the standalone measurement table.
