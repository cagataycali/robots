### Docs: the site splits into a lane a reader browses and a reference lane

The sidebar listed eight sections and ninety entries, so the table of contents
was longer than most of the pages it indexed. `docs/index.md`,
`getting-started/` and `robots/` stay in the nav - three sections - and every
page that states a contract moved under `docs/reference/`, published by an index
generated from the tree rather than by ninety nav rows.

Each of the 92 URLs the move retired is served a redirect, so a link published
before the move still resolves to the page it named.
