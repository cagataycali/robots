# serial_tool register-domain artifact

Generated on a real pty (a serial device), so every number is measured rather
than reconstructed.

* `measure_serial.py` - run inside a checkout; writes the measured JSON.
  Prints the tree it resolved so each run is attributable.
* `make_fig.py` - composes the figure from the two JSON dumps and asserts
  every fact it draws before saving.
* `main_measured.json` / `branch_measured.json` - the two runs.

Reproduce:

    git worktree add -f --detach /tmp/pre upstream/main
    cp measure_serial.py /tmp/pre/ && (cd /tmp/pre && python3 measure_serial.py /tmp/before.json)
    python3 measure_serial.py /tmp/after.json    # from the branch checkout
    python3 make_fig.py
