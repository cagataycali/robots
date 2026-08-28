### Fixed: the ProtoMotions missing-checkpoint refusal names a step instead of the caller's argument

`ProtoMotionsPolicy` takes `onnx_path` as a *local* file and resolves no
HuggingFace model id. `WBCPolicy`, the sibling ONNX policy in the same package,
does resolve one - `_maybe_download_checkpoint` fetches an `org/repo` checkpoint
through `huggingface_hub` - and the `[protomotions]` extra declares that same hub
client, which no module in `policies/protomotions/` imports. The install docs
attributed the fetch to it ("`huggingface_hub` (fetches a checkpoint from a model
id)"), so the documented next step for a reader with no weights on disk was to
pass the model id the module docstring names.

Doing that produced a refusal whose remedy was the argument itself:

```
ONNX artifact not found: cagataydev/protomotions-gtp-unitree-g1. Download from
cagataydev/protomotions-gtp-unitree-g1 on HuggingFace.
```

The caller is told to download the string they just supplied. It is sound advice
for a mistyped local path and a dead end for the value the docs pointed at, since
no part of it turns a model id into the local file the parameter wants.

The refusal now names that step - the canonical repo *and* file name, as a command
whose output is the path to pass:

```
ONNX artifact not found: cagataydev/protomotions-gtp-unitree-g1. This parameter
takes a local file, and this policy does not download one. Fetch the checkpoint
first:
  python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('cagataydev/protomotions-gtp-unitree-g1', 'unified_pipeline.onnx'))"
then pass the path it prints as onnx_path.
```

That command was run against the live checkpoint: it exits 0 and prints a path to
a 22,606,590-byte `unified_pipeline.onnx` in the local hub cache, which is the
argument the constructor accepts. The install section now says the hub client is
one the caller calls, rather than one the extra calls for them.

The repo id and filename become module constants so the message cannot drift from
the checkpoint the module docstring, `ProtoMotionsConfig` and this remedy all name;
a test pins that the raised message interpolates them rather than spelling them a
second time. Nothing else about the guard changes: a missing local path is still
reported by that path, and an existing file still reaches the session build.

Whether this policy *should* resolve a model id the way its sibling does is left
alone - that would add a repo-id spelling to a `*_path` parameter and decide
whether `yaml_path` follows, which is a public-surface choice rather than a
wording one. This change makes the refusal actionable either way.
