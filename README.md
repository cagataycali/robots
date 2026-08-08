# Evidence: PPO optimization-epoch domain

`measure.py` runs a real 60-step PPO run on the SO-100 reach task once per
candidate `num_learning_epochs`, counting `torch.optim.Adam.step` calls and
fingerprinting the actor-critic parameters of the checkpoint each run writes. It
is executed unchanged in two trees (`upstream/main` and the branch) and dumps
`A_main.json` / `B_branch.json`. `compose.py` reads both dumps, asserts every
claim it is about to draw, and renders `epochs_evidence.png`.

The `NEVER-TRAINED` row is the control: `num_learning_epochs=5` with
`PpoTrainer.update` replaced by a no-op. Its parameter sum is identical to the
`0`- and `-3`-epoch runs to 16 digits, which is what establishes that those
checkpoints contain the untrained network.
