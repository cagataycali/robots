### Fixed: `import strands_robots` no longer warns about ffmpeg in a REPL / `-c`

On macOS with Homebrew ffmpeg and torchcodec installed, `python -c "import
strands_robots"`, the REPL and Jupyter printed a `RuntimeWarning` about the
dyld path for video decode on every import, because the shim cannot re-exec
those hosts. The first thing a new user typed answered with a warning about
video they had not asked for.

The import is now silent there (the remedy is logged at debug level) and the
warning moves to the first video-decoding open: `stream_dataset` /
`StreamingDatasetReader.open` with video keys and without `drop_videos=True`
warns with the same `export DYLD_FALLBACK_LIBRARY_PATH=…` line, via the new
`strands_robots._dyld.video_decode_hint()`. Script runs still re-exec as
before; proprio-only streaming is not told about ffmpeg.
