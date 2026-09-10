### Fixed:
The lock pinned `torchcodec` 0.10.0 next to `torch` 2.11.0 on macOS-arm64 and Linux-x86_64, an ABI mismatch (`Symbol not found: __ZN3c1013MessageLoggerC1EPKciib`) that made every `LeRobotDataset` open print a torchcodec traceback and fall back to pyav. `torchcodec` is now 0.11.1 on every platform, matching the linux-aarch64 entry.
