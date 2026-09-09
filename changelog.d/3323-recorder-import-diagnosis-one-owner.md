### Fixed: `DatasetRecorder.create` names which install fixes a failed lerobot import

`lerobot.datasets.lerobot_dataset` fails to import for four unrelated reasons,
and `dataset_recorder` already owns a diagnosis that says which one happened:
lerobot absent (install `strands-robots[lerobot]`), lerobot present but a
package its dataset stack needs absent (install `lerobot[dataset]`), lerobot
present but not providing that module (install the supported range), or a
binary conflict between installed packages that no install fixes. Every
backend's `start_recording` reports it.

The documented direct creation API imported the same module for the same
reasons and composed a second answer -- "lerobot not available. Install with:
pip install lerobot" -- for all four. Three of them have lerobot installed
already, so that command named a package that was already there and changed
nothing; for the conflict case no install is the fix at all.

`DatasetRecorder.create` and `.resume` now raise the diagnosis the probe
reports, so the two surfaces cannot disagree and the contract pinned on the
probe holds for the creation API too.
