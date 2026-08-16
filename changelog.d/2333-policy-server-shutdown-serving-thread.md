### Fixed

`PolicyServer.stop()` no longer kills the thread serving the server. `stop()`
closes the listening socket from the caller's thread while the accept loop is
still waiting on it, and the websockets sync server raises out of
`serve_forever()` when that happens - `ValueError: Invalid file descriptor: -1`
on websockets 12.0, `OSError: [Errno 9] Bad file descriptor` on 13.0 through
17.x. The serving thread is a daemon, so its death was reported nowhere: `stop()`
returned normally and the server looked cleanly shut down (measured 12 of 20
`start()`/`stop()` cycles on websockets 12.0, 3 of 12 on 17.0.1). The accept loop
now runs through a helper that absorbs the failure only while a stop is in
progress, so a socket failure with no stop pending still propagates.
