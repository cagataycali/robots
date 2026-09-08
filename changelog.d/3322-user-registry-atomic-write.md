### Bug Fixes

- **registry**: a `user_robots.json` write now stores the whole overlay or
  leaves it exactly as it was. Both `register_robot` and `unregister_robot` are
  read-modify-writes of the *whole* document, and the write encoded straight
  into the destination it had already truncated - so a failure partway through
  did not lose the entry being changed, it lost every robot the overlay held.
  A value JSON cannot represent reached that point routinely: `hardware` is
  typed `dict[str, Any]` and stored verbatim, so a `Path` port or a numpy joint
  count raised `TypeError` after a prefix of the new document had replaced the
  old one, and `parse_user_robots` reports an unparseable overlay as *no user
  robots* - a warning, then the package registry alone. A refused registration
  therefore destroyed the registry it was refused from, and the previously
  registered robots came back as `Unknown robot` from `get_robot`,
  `resolve_name` and `sim.add_robot`. The document is now serialized in full
  before the destination is touched, and the text is committed through a temp
  sibling plus `os.replace`, so a rejected entry raises a `ValueError` that
  names the store and states it is unchanged, and an I/O error during the
  commit leaves the previous document intact with no temp file behind. A
  successful write produces the same bytes as before.
