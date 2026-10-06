### Fixed: a misspelled Isaac `add_object` shape suggests the shape it misspells

`add_object(shape="spehre")` on the Isaac backend now answers `Unknown shape:
'spehre'. Did you mean 'sphere'? Valid: (...)`, as MuJoCo and Newton already
did for the same typo.
