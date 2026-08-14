# push_to_hub allowlist key — measurement artifacts

Reproduces the figure and the mutation table in strands-labs/robots#2259.

| file | what it is |
|---|---|
| `push_to_hub_allowlist.png` | the composed figure |
| `capture.py` | drives `lerobot_train` over 6 (flag, operator) cases and records the argv, the ask and the store; run once per tree, each dump records its own tree |
| `compose.py` | builds the figure; every drawn number is asserted against the two dumps |
| `dump-main.json` | capture on `upstream/main` |
| `dump-branch.json` | capture on the branch |
| `mutate.py` | applies the 6 regressions, AST-scoped to the enclosing function, and prints `in_fn` vs `in_file` per anchor |
| `parser.py` | the argv/consent probe used during triage |

## Reproducing

```
PYTHONPATH=<tree> python3 capture.py            # once per tree
python3 compose.py                              # asserts, then writes the png
python3 mutate.py                               # 6 rows x 2 arms
```

`capture.py` pops `BYPASS_TOOL_CONSENT` before driving the tool: the harness
environment sets it, and with it set every consent gate is bypassed in-process,
so a naive probe reports no defect at all.
