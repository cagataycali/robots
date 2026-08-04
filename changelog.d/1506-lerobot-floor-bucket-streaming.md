### Fixed: the lerobot floor now guarantees bucket streaming instead of documenting it

`stream_dataset(repo_type="bucket")` needs a `StreamingLeRobotDataset` that
accepts a `repo_type` parameter. lerobot 0.6.0's constructor has none; 0.6.1
added `repo_type: Literal["dataset", "bucket"]`. Every lerobot-bearing extra
floored at `>=0.6.0`, so the resolver installed a lerobot that could not serve
the flagship streaming path at all - the `>=0.6.1` requirement lived only in
prose. All three extras (`[lerobot]`, `[lerobot-async]`, `[molmoact2]`, the last
of which carries its own pin rather than inheriting one) now floor at
`>=0.6.1,<0.7.0`, so 0.6.0 is unresolvable.

Raising the floor does not remove a pre-existing older lerobot from an
environment, so the runtime guard is retained - but its message no longer says
`repo_type='bucket' is not supported by any released lerobot`. That was true
when written and named no remedy; it now reports the version that serves bucket
streaming and the install command that gets it. The version is declared once, as
`strands_robots.streaming_dataset.BUCKET_STREAMING_MIN_LEROBOT`, so the remedy
the error advertises cannot drift from what the resolver installs.
