### Fixed: an unknown Reachy Mini move library is refused, not collapsed to dances

`ReachyMiniDriver.playMove` and `.listMoves` resolved the HuggingFace dataset id
with `'emotions' if library == 'emotions' else 'dances'`, so every other
spelling - `"emotion"`, `"dance"`, `"Emotions"`, `""` - silently addressed the
*dances* library and reported success. Both RPCs now resolve the library through
the admitted set and refuse an unknown name, matching the gate the native
`ReachyDriver` applies to the same two daemon endpoints.
