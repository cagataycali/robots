# Measurement for the cosmos3 native-stack-absent coverage change

`capture.py` measures every number the figure shows (two real subset coverage
runs, then the behavioural outcomes on a torch-free install); `compose.py`
draws it and asserts each rendered cell against `facts.json` before saving;
`mutate.py` produces the mutation matrix (six regressions, two arms, source
restored byte-identically).

Reproduce from a checkout of the branch:

    python3 mutate.py <tag>     # writes /tmp/mut-<tag>.json
    python3 capture.py <tag>    # writes /tmp/art-<tag>.json
    python3 compose.py <tag>    # writes /tmp/fig-<tag>.png
