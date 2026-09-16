### Security: the gr00t_inference container mounts the checkpoint directory read-only

`_download_checkpoint` writes the checkpoint on the host through
`snapshot_download` and the inference server only reads it back, but every
default bind mount was emitted as a bare `-v host:container`. A checkpoint that
executes on load - a torch pickle is arbitrary code - could therefore rewrite
the checkpoint corpus the operator trusts, or drop a new file into a directory
the host reads. The checkpoint mount is now `:ro`, completing the narrowing in
#3755: that change decided which directories may be mounted, this decides how.

The Hugging Face cache mount stays read-write, because the container's
`huggingface_hub` writes into the cache it reuses, and a caller-supplied
`volumes` map is emitted exactly as the operator wrote it. One exception keeps
the checkpoint mount writable: a TensorRT engine cache the operator pointed
inside it, since `--trt-engine-path` is written on first compile so subsequent
runs can load it, and that mount is the only writable place in the default
layout where an engine persists across container recreation. The default engine
path is relative, so it resolves against the server's working directory and the
default mount is read-only.
