### Docs: the post-tune example names what a switch to `cosmos3` needs

`examples/07_post_tune_any_policy.py` said that switching `PROVIDER` to
`"cosmos3"` changes only the provider string. The Cosmos 3 trainer refuses the
example's spec three ways, so the header now names all three: `base_model`,
`extra["sft_toml"]` and `COSMOS_ROOT`.
