### Fixed: the lock resolves a `torchcodec` built against the `torch` it locks

`uv.lock` carried two `torchcodec` entries: 0.11.1 on linux-aarch64 and 0.10.0
on macOS-arm64, linux-x86_64 and win32, both beside a single `torch` 2.11.0.
torchcodec links against libtorch but declares no `torch` requirement of its
own -- its wheel metadata carries no `Requires-Dist: torch` at any version --
so the resolver is free to pair any release in lerobot's
`torchcodec>=0.3.0,<0.12.0` window with any torch, and a mismatched pair is
only discovered when the extension is imported:

    from torchcodec._core import ops
    ImportError: undefined symbol: _ZN3c1013MessageLoggerC1EPKciib

That is what an installed environment saw on linux-x86_64 and macOS-arm64:
every `LeRobotDataset` open logged the load failure and fell back to the PyAV
decoder, so the video path was slower than the lock describes and the
`[lerobot]` extra's stated purpose -- "the read-back stack the streaming data
loop needs" -- was met by the fallback rather than by the codec that was
locked. The pairing is the one the two pyproject files already assume: this
repository's `[lerobot]` extra documents "torch 2.11 + torchcodec 0.11.x +
torchvision 0.26 on linux x86_64/aarch64 and macOS arm64", and lerobot's own
matrix notes torchcodec 0.11 needs torch>=2.11 and 0.12 needs torch==2.12.

Relocking `torchcodec` collapses the two entries into one 0.11.1 for every
platform, which is the version that has wheels for all four and matches torch
2.11. No `pyproject.toml` requirement changes: this is a lock-only correction
of a resolution that was inside the declared bounds and still unimportable.
