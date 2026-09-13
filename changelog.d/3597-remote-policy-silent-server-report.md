### Fixed: a policy server that accepted the connection and went quiet is reported as quiet, not as absent

`RemotePolicy` bounds both of its wire reads - the `ready` handshake with
`connect_timeout`, every reply with `request_timeout` - so a server that
accepted the connection and then said nothing returns control to the caller
instead of holding it. Bounding a read decides *that* the caller gets control
back, not *what* it is told, and these two reads were the only wire failures
this client did not report: each sat in a `try`/`finally` with no `except`, so
the expiry travelled out as a bare `TimeoutError`, which `websockets` raises
with no arguments at all:

    TimeoutError: ''

That names neither the endpoint, nor which of the two reads expired, nor the
budget that expired, nor the parameter carrying it. The connect side of the
same method has always reported actionably ("could not reach a PolicyServer
... Start one first"), and that is the wrong report here - this server is
listening, so starting one names the only thing that is not wrong, while the
remedy for a checkpoint still loading onto the GPU is to read the server's log
or raise the budget.

Both reads now raise a `ConnectionError` carrying the report already shared on
this wire (MODULE `strands_robots.policies._ws_wire`), whose `budget_param`
exists so each client names its own knob:

    PolicyServer at ws://gpu-box:8765 accepted the connection but sent no
    'ready' handshake within connect_timeout=10s. The server is listening, so
    it is still loading (a large checkpoint takes minutes) or it is wedged ...

`Cosmos3WebsocketClient` already drew that distinction through the same helper;
`RemotePolicy` is its second caller, so the two clients on this wire cannot
describe one condition two ways. An absent server keeps its start-the-server
hint, pinned beside the new reports as a control because `TimeoutError` is a
subclass of `OSError` - a clause order that let the absent-server handler see
an expiry first would silently collapse the two cases back into one report.

The connection bookkeeping is unchanged: the discard stays in the `finally`, so
a cancellation between the send and the receive still discards, and the
structural pin that protected that now asserts the invariant directly - the
`finally` performs the discard and no handler does - rather than the absence of
handlers that stood in for it.
