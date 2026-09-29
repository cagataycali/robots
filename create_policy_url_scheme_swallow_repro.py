"""Repro: create_policy swallows the good URL-scheme error and re-raises a generic 404.

Symptom
-------
An end-user calls ``create_policy("http://localhost:5555")`` (a shape that looks
like it should hit the ``remote`` provider). They get:

    ValueError: Unknown policy provider: 'http://localhost:5555'. Available:
    ['cosmos3', 'curobo', 'groot', 'kimodo', 'lerobot_local', 'microduck',
     'mock', 'moveit2', 'protomotions', 'remote', 'rl', 'wbc', 'wbc_gait']

That message names 13 providers and none of them is a hint that the URL
scheme is the problem. The user sits there wondering which provider handles
HTTP — but *no provider* does, only ``ws://``, ``wss://``, ``zmq://``,
``cosmos3://``.

Meanwhile ``strands_robots/registry/policies.py:459-464`` already raises the
exact error the user needs:

    No policy provider handles the URL scheme 'http://' (from
    'http://localhost:5555'). Declared schemes: cosmos3://, ws://, wss://,
    zmq://.

But ``strands_robots/policies/factory.py:365-368`` catches it as a bare
``Exception``, logs it at ``logger.warning`` (invisible in the quickstart's
default logging config), and falls through to stage 3 which re-raises the
generic 404. So the answer exists, is generated, and is thrown away.

Reproduces on d01d382 (v0.5.3 candidate).
"""

import logging
import io
import sys

sys.path.insert(0, "/home/cagatay/bugbash-strands-robots-1790650987/robots")

# End-user quickstart: no explicit logging config, warnings go to stderr but
# not to stdout, and the exception is what they act on.
buf = io.StringIO()
handler = logging.StreamHandler(buf)
handler.setLevel(logging.WARNING)
logging.getLogger("strands_robots.policies.factory").addHandler(handler)

from strands_robots.policies import create_policy  # noqa: E402

for provider_str in [
    "http://localhost:5555",
    "grpc://localhost:9000",
    "tcp://localhost:1883",
]:
    print(f"--- create_policy({provider_str!r}) ---")
    try:
        create_policy(provider_str)
    except ValueError as e:
        # What the user sees at the prompt:
        print(f"  User sees ValueError: {e}")
    # What was actually logged at WARNING (invisible unless logging is configured):
    logged = buf.getvalue()
    if logged.strip():
        print(f"  Silently WARNING-logged (invisible to end-user):")
        for line in logged.strip().splitlines():
            print(f"    {line}")
    buf.truncate(0)
    buf.seek(0)
    print()

# EXPECTED: the ValueError names the URL scheme and the declared schemes list
# (which is what registry/policies.py:459 already raises), so the user knows to
# swap http:// for ws://.
#
# ACTUAL: the ValueError names 13 provider names and swallows the useful
# scheme information into a WARNING that most callers never see.
