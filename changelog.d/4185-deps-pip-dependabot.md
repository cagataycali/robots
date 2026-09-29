### Fixed: `uv.lock` closes the three open Dependabot alerts (transformers, setuptools, torch)

lerobot 0.6.1, the newest release, caps setuptools below 82, transformers
below 5.6 and torch below 2.12, so `uv lock --upgrade-package` could not move
any of them. The floors now live in `[tool.uv] override-dependencies`, next
to the existing diffusers pin: setuptools 84.0.0 (GHSA-h35f-9h28-mq5c),
transformers 5.17.0 (GHSA-xrqw-3rrv-vx5w), torch 2.14.0 with torchvision
0.29.0 and torchcodec 0.16.0 (GHSA-rrmf-rvhw-rf47). Verified against the
lerobot stack: SmolVLA and ACT checkpoints drive the MuJoCo so101, a recorded
dataset reads back through torchcodec and replays, and the lerobot_local,
robot factory and dataset suites pass.

torch 2.14 also reshapes two surfaces the doctor's `torch arch` check reads.
`PYTORCH_RELEASES_CODE_CC` is now keyed by platform under each CUDA release
(the aarch64 wheel carries Thor's `sm_110`, the x86_64 wheel does not); the
remedy reads the row for this machine instead of raising a `TypeError` on the
platform name. And the compatibility predicate consults a table only on a CUDA
build (a second table from CUDA 13.2 admits an Orin on `sm_80` code), so the
test that grades the written-out rule now grades it against the table it
transcribes rather than against the predicate, which on the CPU wheel CI
installs falls back to a plain major-version interval.
