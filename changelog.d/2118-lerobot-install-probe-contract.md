### Quality: drive the lerobot install probe that decides which remedy is printed

``dataset_recorder._describe_lerobot_import_failure`` picks between four install
instructions and branches first on ``_lerobot_installed()``, so that probe decides
whether a caller is told to install lerobot or told lerobot is already present and
something else is wrong. Nothing drove it: all five of its references replaced it
with a lambda, and the branch the suite took was not the branch a caller takes -
importing the recorder does not import lerobot, so a spec lookup answers in a real
process while under pytest the ``sys.modules`` fast path answered every time.

Pins the probe's three documented claims - the spec lookup answers, answering
imports nothing, and a lookup that raises is answered rather than propagated out of
an error path - plus the link from each outcome to the remedy printed. Four
plausible regressions in the probe are caught by these and invisible to the 155
pre-existing tests over the same module.
