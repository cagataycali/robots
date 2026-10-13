### Docs: the site-wide word ceiling is gone; the 900-word page budget stays

`_SITE_BUDGET` was one number in one test file that every docs change had to
edit, so any two open docs pull requests conflicted on it the moment either
merged (#4736 against #4735, on a ceiling each had moved by under 100 words).
It was meant to only move down and was raised more often than it was lowered.
Every page is still graded against the 900-word budget that
`docs/hooks/word_budget.py` states and the mkdocs build enforces.
