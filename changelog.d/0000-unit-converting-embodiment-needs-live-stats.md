### Fixed: a unit-converting embodiment is refused when the normalizer's stats are inert

`state_units`/`action_units` and the normalizer's stats are two halves of one
declaration, and declaring only the units is worse than declaring neither. The
conversion is written into the very tensor an inert normalizer then leaves
alone, so the full so101 joint range reached the model at 160.0 where packing it
natively reaches 2.79 and the checkpoint was trained on ~1 sigma - and the
load-time diagnostic prescribed exactly the unit half the caller had already
declared. That pairing is now refused at load, naming the inert features, the
units the map converts between, and both ways out (supply the training
dataset's stats for the normalizer and unnormalizer, or drop the conversion).
A native map, or a converting map whose stats cover its features, loads
unchanged.
