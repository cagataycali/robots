# use_rosbridge: the two refusals nothing reached

`capture.py` measures, in one pass: every `_err(...)` refusal `use_rosbridge`
makes (by AST), which of them the suite reaches (two coverage arms, the
before-arm deselecting exactly the 15 new cases), what reaches the roslibpy
double for a refused service name / an incomplete publish / a complete publish,
the topic-vs-service parity table, and a 6-row mutation matrix run against both
arms. `compose.py` draws only what the dump contains and asserts every number.

    python3 capture.py <tag>     # writes /tmp/art-rosbridge-<tag>.json
    python3 compose.py <tag>     # writes /tmp/rosbridge-refusals.png

`facts.json` is the dump the published figure was drawn from.
