### Fixed: a torchcodec that cannot load costs one warning line, not a 150-line wall

When torchcodec is installed but its native library cannot load (a notebook,
REPL or `python -c` on macOS where the dyld shim cannot re-exec), recording
and dataset read-back now pass `video_backend="pyav"` themselves and log one
warning naming the remedy, instead of letting LeRobot's default resolver
print its full loader exception before falling back to pyav anyway. A
backend the caller passes still wins; a loading torchcodec is untouched.
