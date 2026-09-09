### Fixed: the EarthRover SDK base URL addresses the host the caller wrote

`base_url_error` graded the *shape* of `port=` and never read the host out of it.
Every endpoint the driver speaks is built from that one string, so a value whose
authority names one host and resolves to another sent `POST /control` - a drive
command - to a rover the caller never named, with `connect_eagerly()` reporting
success whenever something answered there and `get_status()` reporting the base
URL back verbatim, so the address an operator reads still contained the host they
configured. Measured against a real HTTP server: a driver configured for
`bot.local@127.0.0.1:<port>` connected and delivered `POST /control` to
`127.0.0.1`. Two spellings do this and the transport refuses neither, reporting
only the host it ended up with: userinfo (`bot.local@10.0.0.9:8001` dials
`10.0.0.9`), and a foreign scheme (`ws://10.0.0.9:8001` was prefixed into
`http://ws://10.0.0.9:8001`, whose authority is `ws:`, so the request went to the
host `ws` on port 80 and the configured port was discarded). Both are now refused
at construction, naming the host that would have been dialled. The
case-sensitivity behind the second also broke `HTTP://10.0.0.9:8001`, a valid URL
- a URI scheme is case-insensitive and `requests` fetches it identically - so the
scheme test moved into one owner, `_declared_scheme`, that the validator and the
constructor's normalisation both ask; that value now reaches the host it names.
`dial_host_error` is deliberately not asked here, and a test pins why: it grades
a bare host destined for `ws://{host}:{port}`, where an IPv6 literal needs its
brackets, so it refuses the `::1` that `urlsplit` reports for the legitimate
`http://[::1]:8001`. A URL that cannot be used at all is still left to the
transport, which already names it.
