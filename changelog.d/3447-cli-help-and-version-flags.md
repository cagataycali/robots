### Fixed: `strands-robots --help` and `--version` answer instead of "Unknown command"

On a fresh install the first two things typed at the console script -
`strands-robots --help` and `strands-robots --version` - printed
`Unknown command: --help` and exited 1, which reads as a broken install.
`-h/--help` now prints the usage line and the command list (exit 0);
`-V/--version` prints the installed version.
