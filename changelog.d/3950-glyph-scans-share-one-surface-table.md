### Fixed

- The Unicode-dash scan now reads every surface its emoji sibling reads. Both
  rules were separate modules over separate surface sets: emoji graded the
  package, the test tree and `changelog.d/*.md`, while the dash rule graded the
  package alone. `scripts/assemble_changelog.py --apply` folds a fragment body
  verbatim into `CHANGELOG.md`, so the 127 em dashes sitting in 42 pending
  fragments - plus 28 in 9 test modules - would have reached the released notes
  on the next release, on a rule the package itself had already been swept for.
  `tests/test_source_strings_no_emoji_or_unicode_dash.py` replaces both modules
  with one rule table over one surface table, so a rule can no longer read one
  set of files while its sibling reads another; the 155 offender lines are
  rewritten to the ASCII hyphen the rule already names as the remedy.
