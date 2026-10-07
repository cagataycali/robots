### Fixed: `flux3_action` without torch names the whole install

`Flux3ActionPolicy()` on an interpreter without torch now refuses with the same
remedy as a missing `flux-action`: lerobot's torch, the git-only inference
library and the NATTEN wheel for that torch/CUDA build, in one message. Before,
it named only `[lerobot]` and `pip install torch`, and the NATTEN pairing
surfaced as a second refusal after that install.
