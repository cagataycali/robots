### Fixed: a reader looping on `sim.lock` no longer starves `act()`

`sim.lock` was a plain `threading.RLock`, which lets the releasing thread take it straight back. Beside a reader
that loops on `with sim.lock: observe(); describe()`, 5 `act()` calls took 7.1 s on aarch64. The world lock is now
fair (FIFO, still re-entrant): a waiting `act()` gets the world at the next release, and the same 5 calls take 4 ms.
