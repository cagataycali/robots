### Fixed: docs content images are deferred by the HTML, not by a script that runs too late

`docs/assets/docs.js` assigned `img.loading = "lazy"` from a page-ready handler, which
runs after the parser has already started every image request, so the site read as lazy
while each below-the-fold clip still downloaded. `docs/hooks/media.py` now writes the
attribute at build time. Measured on a 390x844 viewport over an emulated 4G connection
with no scrolling, `policies/wbc-rollouts` fetched 664 KB and `hardware/universal-robots`
751 KB before the change, and 0 KB after it. A tag that states its own strategy - the
robot cards emit `loading="lazy"` themselves - is left untouched.
