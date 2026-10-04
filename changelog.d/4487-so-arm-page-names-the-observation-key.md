### Docs: the SO-arm pages say which joint name a read returns

The SO-100 and SO-101 pages listed each joint as "Model joint" beside an
"Action key" label, so a reader who wrote `send_action({"shoulder_pan": v})`
expected `get_observation()["shoulder_pan"]` back and got `KeyError`: the
observation keeps the model's joint name (`"1"`), which keeps recorded datasets
and trained checkpoints at one column per joint. The table now heads its columns
"Observation key" and "`send_action` label", and the `SimEngine.get_observation`
schema no longer cites `"shoulder_pan"` as an observation key.
