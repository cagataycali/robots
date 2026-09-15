### Security: a request header cannot forge a dashboard log entry

`Host`, `Origin` and the forwarded address are chosen by whoever sends the
request, and the dashboard's refusals quote them so the operator can see what
was refused. A log file is parsed by line, so a carriage return and a line
feed inside one of those values forged a second entry that never happened -
`passkey enrolled for root`, spelled by a stranger, indistinguishable from the
real line above it.

Every dashboard log statement that quotes a caller-supplied value now goes
through `log_redaction.one_line`: control characters become their escapes
(`\r\n` reads as `\r\n`, so the bytes that arrived are still legible), and a
value longer than 200 characters is cut with an ellipsis so a header cannot
fill the file. Four sites in `auth.py` and `settings.py`; the messages
themselves are unchanged.
