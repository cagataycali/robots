### Security: urllib3 is floored at 2.8.0, the first release clearing GHSA-8988-9cw3-xx77, GHSA-vxq7-64xx-v4gw and GHSA-gh4c-6fx4-qh6g

urllib3 reaches the tree only through `botocore` and `requests`. The lock
moves to 2.8.0 and `[tool.uv] constraint-dependencies` names the floor, so a
later `uv lock` that would resolve below it fails loudly instead of silently
reopening the HTTPS-proxy TLS, unbounded chunk-size and chunked-Deflate
advisories.
