### Fixed: an unknown policy provider is refused naming the close spelling

`create_policy("gr00t")`, `create_policy("WBC")` and `create_policy("cosmos")` now raise `Unknown policy provider: 'gr00t'. Did you mean: 'groot'? Available: [...]`, matching the hint `Robot()` gives for a robot name. Case and dashes are folded before matching, against every registered name and alias.
