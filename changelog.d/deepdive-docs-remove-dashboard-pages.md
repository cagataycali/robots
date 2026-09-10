### Removed: the six `docs/dashboard/` pages

They documented `python -m strands_robots dashboard`, a command the package
does not ship (`Unknown command: dashboard`; available: doctor, verify-dataset),
and were not in the site nav. 984 lines of prose about an unshipped UI are gone;
the `dashboard` extra and `strands_robots/dashboard/` helpers are unchanged.
