### Fixed: a redacted log record is still renderable, so the access log keeps every request

`RedactingFilter` baked the redacted text into `record.msg` and cleared
`record.args` - the stock way to freeze a redacted message. uvicorn's
`AccessFormatter`, the formatter this module exists for, never reads that
message: `formatMessage` unpacks five values out of `record.args` and builds the
request line from them. Clearing the args left it nothing to unpack, so it raised
`ValueError` inside `Handler.emit` for exactly the records that carried a
credential. Measured through uvicorn 0.41's own formatter, six requests: three
reached the access log, and all three authenticated WebSocket handshakes were
dropped, each leaving a `--- Logging error ---` traceback on stderr. Because
every camera and mesh socket carries its JWT in the query string, the log kept
every ordinary request and lost every authenticated one.

`args` is a rendered part, so it is now redacted in place with its arity
preserved, and a non-`str` value is left alone (an `int` under a `%d` is not a
credential, and a fingerprint there would leave a format string its own args
cannot render). The message is baked and the args dropped only when the
credential is not visible in any single arg - the case no per-arg redaction can
reach, where failing closed under `Handler.emit`'s guard is the right answer.
That verdict is taken by asking which credential values survive rather than by
re-running the redaction, whose own fingerprint matches the value pattern and
would report a wrong length for text that is already clean.
