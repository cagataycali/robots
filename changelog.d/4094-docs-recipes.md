### Documentation: a Recipes section - ten runnable stories, each one shipped example

The site nav gains `Recipes`: the Quickstart plus nine pages, each showing one
`examples/` script (included by snippet, so the page cannot drift from the file),
the command that runs it, its real output and a rendered lead image, then links
into the reference page it depends on. `reference/examples/overview.md` is folded
into `recipes/index.md` (both old URLs redirect there), and the installation page
drops the dependency-floor derivations `pyproject.toml` already carries and the
env-var table `reference/configuration.md` already lists, so the site total
still goes down.
