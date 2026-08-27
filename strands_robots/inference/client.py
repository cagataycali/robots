"""Remote policy inference client.

The ``remote`` provider's client half.  Wraps a live WebSocket to a
:class:`~strands_robots.inference.PolicyServer` and presents the full
:class:`~strands_robots.policies.base.Policy` ABC locally, forwarding
every observation to the server and returning the action chunk.
"""
