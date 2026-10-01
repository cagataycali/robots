### Changed: docs code fences the site does not run render as plain Python

A fence marked `python title="sketch"` no longer carries a "python, not run on
this page: needs a robot on USB" chip. The docs hook strips the marker, so the
fence renders like any other highlighted Python block, and the surrounding
prose says what it needs (an SO-101 on USB, the `groot` extra). The marker
itself stays: `docs/hooks/check_sketches.py` and the docs tests still key on it.
