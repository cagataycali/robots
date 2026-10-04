### Fixed: a robot whose lerobot type is only on lerobot main names the from-source install

`Robot("rebot_b601", mode="real")` (and `bi_rebot_b601`) on a PyPI lerobot used
to answer with the installed lerobot's `Known lerobot robot types` listing, which
cannot contain the robot. The registry already marks these entries
`hardware.requires_lerobot_from_source`; the refusal now reads that flag and
names the entry, `pip install 'git+https://github.com/huggingface/lerobot'` and
the robot's docs page.
