### Docs: the README recording row names the HF Storage Bucket it syncs to

The "What you get" row said recordings "stream to HF datasets or Storage
Buckets", which reads as generic cloud storage (S3, GCS, Azure).
`sync_dataset_to_bucket` only accepts an HF bucket name (`name` or `org/name`)
and refuses every `s3://`, `gs://`, `azure://` or `https://` target. The row now
says what the code does: push to a Hub dataset repo or sync to an HF Storage
Bucket, the same two targets `docs/learn/data/stream-and-sync.md` lists.
