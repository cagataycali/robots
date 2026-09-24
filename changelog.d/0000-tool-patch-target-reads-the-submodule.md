### Fixed: a tool patch target reads the submodule, not the package slot

`strands_robots.tools` caches each tool object into its own namespace under the
name of the submodule that defines it, so `strands_robots.tools.<name>` holds
either the tool or the module - whichever was read first in the process. Both
patchers resolve a *string* target by walking that slot, so 152 targets over 12
test modules (`monkeypatch.setattr("strands_robots.tools.serial_tool.time.sleep",
...)`, `patch("strands_robots.tools.gr00t_inference.subprocess.run")`) raised
`ImportError`/`AttributeError`, or patched the tool object rather than the module
the code under test reads, depending on which file ran first: 9 failures and 3
errors in one such order on `main`, and 8 under a 4-worker parallel run. Each now
passes the module object it means, obtained with `importlib.import_module`.

`tests/tools/test_tools_lazy_import.py` was what flipped the slot mid-run: it
drops a cached binding to force a fresh lazy resolution, which leaves the *tool*
there for the rest of the process. It now restores the binding it found. The
derived rule in `tests/tools/test_lazy_tool_name_is_not_read_as_a_module.py`
grades the string spelling too, and pins the flip it rests on.
