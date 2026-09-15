### Fixed: `use_rosbridge(action="list_topics")` lists every topic rosapi reports

The listing paired rosapi's `topics` and `types` arrays by index, so any topic
past the end of a short `types` went unreported - and a reply carrying no
`types` at all listed nothing, answering an empty graph with `status="success"`.
Only `topics` is guaranteed by that service: roslibpy's own client asserts
`"topics" in result` and reads that array alone. Every topic is now listed, a
topic rosapi named no type for reads as `[type not reported by rosapi]`, and a
count mismatch is named in a warning.
