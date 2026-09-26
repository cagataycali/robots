### Docs: the Training page keeps the abstraction, the provider knobs get their own

`docs/training/overview.md` was 1,788 words carrying two subjects. The page is
1,158 words and keeps the `Trainer` abstraction, the record-train-deploy loop,
the `TrainSpec` table and the per-provider installs; what each backend accepts
through `TrainSpec.extra` - lerobot's config tree, reward models and sample
weighting, the GR00T and Cosmos3 knobs - is a new
`docs/training/provider-knobs.md` at 729 words, moved verbatim with its anchors
intact.
