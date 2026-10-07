### Docs: README "What you get" names the one supported bucket flavour

README L94's Teleop/Record row read "stream to HF datasets or Storage Buckets":
the capitalized plural "Storage Buckets" after the comma-separated alternative
"HF datasets" let a new reader bind the second item to AWS S3 / GCS / Azure /
R2 - four cloud blob stores that the project does not integrate with.
`strands_robots.dataset_transfer.sync_dataset_to_bucket` is the single sync
entry point and its allowlist is HF-repo-id-shaped (`"name"` or `"org/name"`,
`[A-Za-z0-9._-]` only); any URL-shaped bucket is refused in the input gate,
before the `hf` CLI is located. No `boto3`, `google.cloud.storage` or
`azure.storage` is imported in the module.

The docs page the row links to
([docs/learn/data/stream-and-sync.md](../docs/learn/data/stream-and-sync.md))
is unambiguous - it writes "HF Storage Bucket" (singular, prefixed) in every
occurrence. The README is the only surface that writes "Storage Buckets" bare
and plural after "HF datasets". Row now mirrors the docs scoping: "sync to a
Hugging Face dataset repo or an HF Storage Bucket".
