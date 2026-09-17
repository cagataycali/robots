### Added: the agent tool reference is generated from the tool functions

`docs/reference/tools.md` is a provenance line and one `{{tool_reference}}` token.
`docs/hooks/tool_reference.py` expands it at build into a section per agent-callable
`@tool` under `strands_robots/tools/` - 68 of them - carrying the one-line summary, the
`@tool(context=True)` operator seam where there is one, the `Returns:` clause each tool
states its refusals in, and a table of every parameter with its annotation, default and
the description the agent is shown. Nothing on the page is typed by hand, so a tool added
to the package documents itself at the next build and a renamed parameter cannot linger.
Like the two hooks beside it the reader is `ast` and the filesystem: the docs environment
installs mkdocs alone, so importing the package there is not available.

The page is graded against the schema an agent is really handed.
`tests/test_docs_tool_reference_hook.py` builds every tool's `tool_spec` and compares
parameter names and order, which parameters are required, every default and every
description, tool by tool - so the page cannot claim a signature the decorator does not
publish, and a tool the hook stops looking for fails the sweep instead of quietly
shortening the page.
