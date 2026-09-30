### Fixed: Isaac colour randomization says which robots it cannot recolour

On Isaac, converted robot visuals are read-only USD instances, so `randomize(colors=True)` recoloured only objects while reporting success. The envelope now names the robots it left untouched and why, and the docs say colours are objects-only on Isaac.
