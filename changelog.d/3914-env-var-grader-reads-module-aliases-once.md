### Tests: the env-var grader reads a module's aliases once, not once per function

`test_env_vars_the_package_reads_are_documented` derived its resolver set by
evaluating a module's import aliases and string constants beside each function
the module defines, so every module was walked once per function - 4,684 walks
over 311 files - before a single resolver was looked for, and each function's
once-bound locals were re-derived on every pass of the fixed-point loop. Each is
now read once and carried beside the function. Same population, same rule, same
message; the file runs in 6 s where it ran in 26 s. Towards #3869.
