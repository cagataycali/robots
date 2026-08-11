# use_rosbridge connect() state coverage

- `capture.py` - measures the descriptor-shadowing effect on both descriptor kinds in-tree
- `mutate.py` - the mutation table (4 production regressions + 1 test-side), two arms
- `compose.py` - builds the figure; asserts every rendered number against `facts.json`
- `facts.json` - the measured data behind every cell
