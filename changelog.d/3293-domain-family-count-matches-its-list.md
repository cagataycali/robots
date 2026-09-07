### Docs: a shared numeric domain that counts its families enumerates that many

`non_negative_count_error` opened with "Shared domain for two families" above a
list of three, and then explained that refusing `0` "would reject the common
configuration for both". The third family - the episode counts of the
dataset-integrity gate, spent by `verify_dataset`'s `expected` / `min_frames`
and by `SimEngine.verify_dataset_episodes` - had been added to the list without
the count above it or the sentence below it following. That count is what tells
a reader arriving from one caller how far the domain reaches, and what tells the
next author that a caller in a new family widens the list rather than just
calling the function, so a count that disagrees with the list misstates both.
The seed bullet named `TrainSpec.seed` alone and now also names
`StreamingDatasetReader.open`, the other surface measured to spend it.

`positive_count_error` drifted the same way earlier, saying "three" while
listing four, so the claim is now graded against the list it sits above: a
stated family count must have one top-level bullet per family. That also
forbids the shape which made the drift invisible in the first place - a count
enumerated in prose with nothing countable beneath it, which is how
`non_negative_whole_number_error` stated "two families"; its two families are
now bullets, matching its three siblings in the same file.
