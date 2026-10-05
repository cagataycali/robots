### Fixed: `verify-dataset` tells a wrong path apart from an unfinished recording

A dataset root that does not exist (a typo, a dangling symlink) or that is a
file instead of the dataset folder used to report "The dataset is empty or was
never finalized", sending the user back to `stop_recording`. The episode reader
now says the path does not exist or is not a directory; only a real but empty
dataset folder keeps the finalize advice.
