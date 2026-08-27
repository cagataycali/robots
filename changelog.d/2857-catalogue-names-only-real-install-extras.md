### Fixed: the policy catalogue named an install extra that cannot exist, and the guard for that class could not see the column

``docs/policies/overview.md`` is the page a reader lands on to choose a
``policy_provider``, and its Providers table gave VERA's install extra as the
bare name ``vera``. There is no such extra. Three places in the tree already say
so: ``pyproject.toml`` carries a comment beginning "There is intentionally no
``vera`` extra", ``docs/policies/vera.md`` repeats it in its install section, and
the README's VERA row reads "Git-only (not on PyPI, no extra)". The catalogue was
the one surface that disagreed, and the page's own preamble claimed the table
"can never silently drift".

pip does not refuse an extra a distribution never declared, and on a current pip
it does not even warn. Measured on pip 26.0.1 against a throwaway project whose
only extra is ``real``, both ``pip install --dry-run --no-deps '.[real]'`` and
the same command for ``'.[nope]'`` answer ``Would install extra-probe-0.0.1`` and
exit 0, with no mention of the bad name anywhere in the output -- the
``WARNING: ... does not provide the extra ...`` line older pip printed is gone. So
a reader following that row installed the base package, received none of VERA's
client dependencies, and got no signal at all; the only remaining evidence is an
ImportError met later, somewhere unrelated-looking. The cell now says there is no
extra and leaves the install to the provider's page, which carries the two
commands the client actually needs.

``tests/test_dependency_audit`` already owns this rule, and its history section
records fixing this same name once before -- on the VERA page. It sweeps the
qualified ``strands-robots[NAME]`` form, and its stated reason for sweeping only
that form is that a bare ``[wbc]`` in prose is ambiguous with lerobot's own
extras, which are written identically. That is true of prose and it does not
survive a table column whose header says what the column holds: a cell under
``| Install extra |`` in this project's own docs names one of this project's
extras by construction. The catalogue is exactly such a column, which is why the
name survived the page fix, so the column becomes the third graded surface
alongside the written command and the runtime ``require_optional(extra=...)``
message.

Both cell forms in use are read, because they split the population in half: a
bare name in the README and policy tables, and the bracket a reader types in the
architecture and installation matrices. Reading only the bare form left two of
the four columns ungraded. The sweep covers 4 columns and 46 names, of which the
one violation was VERA's; the remaining 45 are unchanged, and floors under both
counts make a scan that stops matching fail rather than report a clean tree. A
narrowed file walk is caught by those floors, and a narrowed header vocabulary by
constructed exemplars that grade the reader on markdown built in the test, since
a clean tree can no longer exercise a rejection.
