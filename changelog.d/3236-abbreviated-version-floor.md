### Fixed

- The dependency-audit sweep no longer reports a version floor written without
  its trailing zeros as undercutting itself. PEP 440 reads an absent release
  component as zero, so `numpy>=1.21` and `numpy>=1.21.0` are one release and
  pip resolves them identically; the comparison ordered the shorter key first
  because a tuple that is a prefix of another sorts below it, which made the
  abbreviation of every declared floor a reported offence.
