### Fixed: `add_object(material={"builtin": ...})` suggests the builtin a typo was close to

A near-miss such as `"chekker"` or `"flatt"` now reads `unknown builtin 'chekker'.
Did you mean 'checker'? Supported: checker, flat, gradient.`, matching the
shape and material-key refusals in the same builder. A name close to nothing
still gets the plain supported list.
