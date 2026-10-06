### Fixed: README links to the docs open the built site, with its tables and figures

The README linked docs pages by their Markdown source (`docs/<page>.md`). Six
of those pages are filled in by MkDocs hooks at build time, so on github.com a
reader saw `{{env_vars}}`, `{{robot_cards}}` or `{{extras:table}}` where the
table should be, and on PyPI the relative links were dead. Every README link to
a docs page now points at `https://strands-labs.github.io/robots/<page>/`.
