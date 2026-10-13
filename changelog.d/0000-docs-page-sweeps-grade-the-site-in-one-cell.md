### Tests: four docs-page sweeps grade the site in one cell, not one cell per page

Four docs graders (Hub ids named on a page, front matter that YAML can read,
stated Simulation action counts, `:::` API directives and their members) used
to parametrize over every page or directive, so each page was its own test
cell: 667 cells in all. Each sweep now reads the same pages in one cell and
lists every offending page in one message. The four files go from 700
collected cells to 37; the non-vacuity cells beside them stay, and the
front-matter and Hub-id sweeps now also fail when they find no page to read.
