### Removed: the `lerobot_async` provider, and a URL scheme no provider declares is refused

lerobot ships the replacement itself - `lerobot.async_inference.policy_server`
and `robot_client` - so this provider duplicated it, and `pip install
"strands-robots[lerobot-async]"` no longer resolves; drive a remote policy
through lerobot's own client, or through the `remote` provider over
`ws://`/`wss://`.

Its removal left `^grpc://` declared by no provider, which exposed the
resolver's fall-through: every stage after the URL match reads the string as a
*name* - a shorthand, a Hub repo id, a provider - so an address reaching them is
forwarded to `lerobot_local` as a checkpoint id. Measured before this change,
`resolve_policy("grpc://gpu-box:8080")` returned `("lerobot_local",
{"pretrained_name_or_path": "grpc://gpu-box:8080"})` under a warning, so the
caller's next report was a HuggingFace lookup failure for a repo id nobody
wrote, nowhere near the server they named. The same held for `http://`,
`tcp://`, or any scheme this package does not ship.

Stage 1 now refuses a `scheme://` outside the declared set, naming the schemes
that do resolve (`cosmos3://`, `ws://`, `wss://`, `zmq://`). That list is read
back out of the `url_patterns` entries the stage matches on and filtered by
those same patterns, so a scheme this package does not ship is never advertised.
A string with no leading scheme is untouched: a shorthand, a repo id and a bare
`host:port` resolve exactly as before.
