### Docs: the SO-101 prints in Strands green on the site

The docs viewer showed the SO-101 in the menagerie model's yellow. Its printed
parts are whatever colour the kit was printed in, so the site now prints them in
the Strands green and keeps the STS3215 servos black, in the 3D viewer (both
colour schemes, following the palette toggle) and in the catalog thumbnail. The
mechanism is a `palette` per robot in the viewer manifest (`docs/hooks/manifest.py`
`_BRAND_PALETTES`): a model colour as the MJCF spells it and a theme token the
viewer resolves. `tests/test_docs_viewer_brand_palette.py` pins that every token
exists in the viewer's theme, that every painted robot is one the viewer renders,
and that no other robot is repainted.
