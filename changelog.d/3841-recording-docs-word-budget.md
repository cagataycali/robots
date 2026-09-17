### Docs: the recording reference is four pages, each inside a 1,500-word budget

`docs/recording.md` carried recording, verification, replay and the
`DatasetRecorder` API in one 9,924-word page, most of the volume prose restating
refusals the code already words. It is now four pages - `recording.md` (session
verbs, rates, camera selection, `root`/`overwrite`),
`data/dataset-recorder.md` (the writer, the domains it refuses, MP4 clips,
codecs, publishing), `data/verifying-datasets.md`
(`verify_dataset_episodes`, `verify-dataset`, dropped frames and poisoned
flushes) and `data/reading-back.md` (replay, streaming, streamed training) -
totalling 5,267 words, with option lists rendered as tables and no fact dropped.
Inbound links and the `#where-the-dataset-is-written-root-overwrite` anchor are
unchanged, so no documented URL moves.

`tests/test_docs_pages_are_within_the_word_budget.py` holds the budget as a
ratchet: a page over 1,500 words fails unless `_OVER_BUDGET` names it, and an
entry that no longer names an over-budget page fails too, so the list of pages
still owing a split can only shrink.
