### Docs: the install and start pages no longer state a doctor probe count

`docs/start/install.md` and `docs/start/index.md` said "fifteen probes" while `strands-robots doctor --list` prints 17. Both now name the probes without a count, and a test refuses any docs page that states one, so the pages cannot drift again when `CHECKS` grows.
