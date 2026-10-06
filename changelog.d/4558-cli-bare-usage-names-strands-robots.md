### Fixed: a bare `strands-robots` prints the same usage line as `--help`

Running `strands-robots` with no command printed
`Usage: python -m strands_robots <command>`, naming an invocation the user had
not typed, while `--help` on the same dispatcher printed
`Usage: strands-robots <command> [options]`. Both branches now print one shared
usage line naming the documented console script; the bare call still exits 1.
