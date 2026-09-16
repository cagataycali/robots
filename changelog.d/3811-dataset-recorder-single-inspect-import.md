### Fixed: `dataset_recorder.py` imports `inspect` once

#3792 gave `dataset_recorder.py` a module-scope `import inspect`; #3787, drafted
before it, landed the grader that refuses a function that re-imports a name the
module already binds. Green apart, red together: `main` failed
`test_no_function_scope_import_repeats_a_module_scope_binding[dataset_recorder.py]`
on the two old function-scope `import inspect` lines in `create()` and
`_resume_existing()`. Both are deleted; the module-scope binding is the one
that stays. No behaviour change.
