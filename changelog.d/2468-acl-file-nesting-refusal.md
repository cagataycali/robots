### Fixed

- **mesh**: an ACL file too deeply nested for the JSON5 parser to read is now refused as an unloadable ACL instead of escaping the loader's fail-closed boundary as a `RecursionError`, so the start-time gate still refuses to bring the wire up and the operator gets the path and a remedy.
