### Fixed: a UTF-8 file on disk reads the same whatever the process locale is

`open()`, `Path.read_text`/`write_text`, `os.fdopen` and
`tempfile.NamedTemporaryFile` fall back to `locale.getencoding()` when no
`encoding` is given, so 42 call sites decoded and encoded with whatever codec the
process happened to have. None of the files involved is a locale-encoded
document: a benchmark spec or policy config is authored in an editor, and lerobot
writes `meta/info.json` with `encoding="utf-8"` and `ensure_ascii=False`.

Measured under an ASCII locale with UTF-8 mode off, on files that are valid
UTF-8: `register_benchmark_from_file` raised `UnicodeDecodeError` instead of
registering a benchmark whose `instruction` is Spanish; `WBCConfig.from_file`
raised instead of returning a `policy_path` that names a non-ASCII directory; and
a three-task dataset's `total_tasks` was read as absent, which
`validation_split_error` honours as single-task and so admits the per-task
holdout it exists to refuse. Every site now states `encoding="utf-8"`, and the
answers under a UTF-8 locale are unchanged.
