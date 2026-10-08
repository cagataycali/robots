### Tests: a distributed run rewrites each test module's asserts once, not once per worker

On a fresh checkout every `pytest-xdist` worker rewrote the asserts of all
~2,200 test modules itself before its first test, because the workers import
them in the same order at the same moment. The controller now fills pytest's
own rewrite cache before it starts the workers, split across one process per
worker, and the workers read it. A cold `-n 2` collection on two cores takes
102 s instead of 144 s; a warm one is unchanged.
