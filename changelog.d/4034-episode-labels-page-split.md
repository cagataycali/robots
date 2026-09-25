### Docs: the episode-labels page keeps the sidecar, the judge gets its own page

`docs/data/episode-labels.md` had grown to 2,283 words carrying two subjects: the
label sidecar - where it lives, its schema version, the field domains held in both
directions, and the filtered re-training step - and the judge agent that writes
into it.

Split at that H2 boundary. `data/episode-labels.md` (1,030 words) is the sidecar;
the new `docs/data/episode-judge.md` (964 words) is the judge: the four tools
`create_judge_agent` assembles, the per-camera image blocks `sample_frames` emits,
and measuring judge/human agreement against its baseline before filtering training
data. Review measurements that the CHANGELOG and the PRs already hold went first;
every contract sentence, refusal and field domain survives. Both pages are inside
the 1,500-word budget and the new page has a nav row.
