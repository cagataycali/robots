### Security: the dashboard frontend lock takes source-map-js 1.2.2 (GHSA-68fv-2mgg-jv7q)

`source-map-js` reaches the frontend only as a build-time dependency of
`postcss` (under Vite). 1.2.1 lets a crafted source map stall the event loop;
`package-lock.json` now resolves 1.2.2, the first release that clears it.
Rebuilding the bundle from the new lock produces the same `static/app.js` as
the old one on the same toolchain, so nothing shipped changes.
