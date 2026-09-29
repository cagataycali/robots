### Docs: the Isaac Sim install has a uv spelling that resolves

The only documented Isaac Sim install line was pip's, and it does not resolve under uv. The install hints and docs now also give the uv line, with the two flags it needs (`--index-strategy unsafe-best-match --prerelease=allow`), both measured necessary.
